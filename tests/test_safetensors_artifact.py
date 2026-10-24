"""Adapter artifact format 2.0: safetensors only (ARTIFACT_SPEC SER7, SER10, SER11)."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import ASTConfig, ASTForAudioClassification

from ast_audio_classification_pipeline import (
    ADAPT_CLASSES,
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    MODEL_REVISION,
    ASTAudioClassificationPipeline,
)
from ast_audio_classification_pipeline.pipeline import ARTIFACT_METADATA_KEY, read_artifact

ROOT = Path(__file__).resolve().parents[1]

# A tiny AST so a full CPU forward pass is cheap; built the same way in the fresh interpreter below.
TINY = (
    "ASTConfig(num_labels=527, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, "
    "intermediate_size=32, max_length=16, num_mel_bins=16, patch_size=4, frequency_stride=4, time_stride=4)"
)
BUILD = f"""
import numpy as np
import torch
from transformers import ASTConfig, ASTForAudioClassification
from ast_audio_classification_pipeline import ADAPT_CLASSES, ASTAudioClassificationPipeline, rehead_model

def build():
    torch.manual_seed(0)
    model = ASTForAudioClassification({TINY}).eval()
    rehead_model(model, ADAPT_CLASSES, seed=7)
    pipe = ASTAudioClassificationPipeline(
        _runner=lambda _: np.zeros(3, dtype=np.float32), labels=list(ADAPT_CLASSES), device="cpu",
        model=model, extractor=object(), activation="softmax",
    )
    pipe.adaptation_config = {{"mode": "classifier-head-gradient-adaptation", "rehead_seed": 7,
                               "history": [{{"epoch": 1, "train_loss": 0.5}}], "learning_rate": 0.001}}
    return pipe
"""
namespace: dict = {}
exec(BUILD, namespace)
build = namespace["build"]


def _fresh_pipe() -> ASTAudioClassificationPipeline:
    torch.manual_seed(123)
    model = ASTForAudioClassification(eval(TINY, {"ASTConfig": ASTConfig})).eval()
    return ASTAudioClassificationPipeline(
        _runner=lambda _: np.zeros(527, dtype=np.float32),
        labels=[f"label-{i}" for i in range(527)],
        device="cpu",
        model=model,
        extractor=object(),
    )


def _fixed_input() -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, 16 * 16).reshape(1, 16, 16)


def _split(data: bytes) -> tuple[dict, bytes]:
    (n,) = struct.unpack("<Q", data[:8])
    return json.loads(data[8 : 8 + n]), data[8 + n :]


def _join(header: dict, body: bytes) -> bytes:
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + body


def test_round_trip_into_fresh_model_gives_identical_tensors_and_outputs(tmp_path):
    pipe = build()
    saved = pipe.save_artifact(tmp_path / "ast-audio-adapter-v2.safetensors")
    fresh = _fresh_pipe()
    fresh.load_artifact(saved)

    original = pipe.model.classifier.state_dict()
    reloaded = fresh.model.classifier.state_dict()
    assert set(original) == set(reloaded)
    for key, tensor in original.items():
        assert torch.equal(tensor, reloaded[key]), key
    assert fresh.labels == list(ADAPT_CLASSES)
    assert fresh.adaptation_config == pipe.adaptation_config

    # Same backbone weights, then identical logits on a fixed input through the full model.
    fresh.model.audio_spectrogram_transformer.load_state_dict(
        pipe.model.audio_spectrogram_transformer.state_dict()
    )
    with torch.inference_mode():
        expected = pipe.model(_fixed_input()).logits
        actual = fresh.model(_fixed_input()).logits
    assert torch.equal(expected, actual)


def test_saved_file_is_plain_safetensors_with_one_json_metadata_key(tmp_path):
    saved = build().save_artifact(tmp_path / "a.safetensors")
    data = saved.read_bytes()
    assert not data.startswith(b"PK")  # not a torch.save zip archive
    assert b"\x80\x02" not in data[:16] and b"torch._utils" not in data and b"collections" not in data
    header, body = _split(data)
    assert set(header["__metadata__"]) == {ARTIFACT_METADATA_KEY}
    text = header["__metadata__"][ARTIFACT_METADATA_KEY]
    metadata = json.loads(text)
    assert text == json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert metadata["format"] == ARTIFACT_FORMAT
    assert metadata["format_version"] == ARTIFACT_FORMAT_VERSION == "2.0"
    assert read_artifact(saved)[0] == metadata
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    assert set(tensors) == {"dense.bias", "dense.weight", "layernorm.bias", "layernorm.weight"}
    # Every byte after the header belongs to a declared tensor: nothing else is stored.
    assert max(v["data_offsets"][1] for v in tensors.values()) == len(body)
    assert sum(v["data_offsets"][1] - v["data_offsets"][0] for v in tensors.values()) == len(body)


def test_two_saves_and_a_fresh_interpreter_give_identical_bytes(tmp_path):
    first = build().save_artifact(tmp_path / "one.safetensors").read_bytes()
    second = build().save_artifact(tmp_path / "two.safetensors").read_bytes()
    assert first == second
    other = tmp_path / "fresh.safetensors"
    script = BUILD + f"\nbuild().save_artifact({str(other)!r})\n"
    env_path = str(ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {env_path!r})\n" + script],
        check=True,
        timeout=300,
    )
    assert other.read_bytes() == first


def test_save_refuses_non_safetensors_suffix_and_non_plain_metadata(tmp_path):
    pipe = build()
    with pytest.raises(ValueError, match=r"\.safetensors"):
        pipe.save_artifact(tmp_path / "adapter.pt")
    pipe.adaptation_config["path"] = tmp_path  # a live pathlib object is not plain data (SER11)
    with pytest.raises(TypeError):
        pipe.save_artifact(tmp_path / "adapter.safetensors")
    pipe.adaptation_config["path"] = float("nan")
    with pytest.raises(ValueError):
        pipe.save_artifact(tmp_path / "adapter.safetensors")


def test_loader_refuses_pickle_truncated_and_tampered_files(tmp_path):
    saved = build().save_artifact(tmp_path / "good.safetensors")
    data = saved.read_bytes()
    target = _fresh_pipe()
    original_classifier = target.model.classifier

    pickled = tmp_path / "old-format.pt"
    torch.save({"format": ARTIFACT_FORMAT, "format_version": "1.0"}, pickled)
    with pytest.raises(ValueError, match="not a .*safetensors"):
        target.load_artifact(pickled)

    for cut in (4, 40, len(data) - 3):
        truncated = tmp_path / f"cut-{cut}.safetensors"
        truncated.write_bytes(data[:cut])
        with pytest.raises(ValueError, match="safetensors"):
            target.load_artifact(truncated)

    header, body = _split(data)
    metadata = json.loads(header["__metadata__"][ARTIFACT_METADATA_KEY])

    def tampered(name: str, new_header: dict, new_body: bytes = body) -> Path:
        path = tmp_path / f"{name}.safetensors"
        path.write_bytes(_join(new_header, new_body))
        return path

    wrong_base = {**metadata, "base_model_revision": "0" * 40}
    assert wrong_base["base_model_revision"] != MODEL_REVISION
    wrong_header = {**header, "__metadata__": {ARTIFACT_METADATA_KEY: json.dumps(wrong_base)}}
    with pytest.raises(ValueError, match="base_model_revision"):
        target.load_artifact(tampered("base", wrong_header))

    extra_key = {**header, "__metadata__": {**header["__metadata__"], "format": "pt"}}
    with pytest.raises(ValueError, match="exactly"):
        target.load_artifact(tampered("extra", extra_key))

    bad_json = {**header, "__metadata__": {ARTIFACT_METADATA_KEY: "{not json"}}
    with pytest.raises(ValueError, match="plain JSON"):
        target.load_artifact(tampered("json", bad_json))

    nan_body = bytearray(body)
    start = header["dense.bias"]["data_offsets"][0]
    nan_body[start : start + 4] = struct.pack("<f", float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        target.load_artifact(tampered("nan", header, bytes(nan_body)))

    assert target.model.classifier is original_classifier
    assert target.labels[0] == "label-0"
