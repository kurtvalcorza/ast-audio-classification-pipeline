"""The adapter is exported as a safetensors state dict plus a plain-JSON sidecar, never as a pickle."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from transformers import ASTConfig, ASTForAudioClassification

from ast_audio_classification_pipeline import (
    ADAPT_CLASSES,
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    ASTAudioClassificationPipeline,
    artifact_sidecar_path,
    rehead_model,
)

_CONFIG = dict(
    num_labels=527,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=32,
    max_length=16,
    num_mel_bins=16,
    patch_size=4,
    frequency_stride=4,
    time_stride=4,
)


def _adapted_pipe(seed: int = 0) -> ASTAudioClassificationPipeline:
    torch.manual_seed(seed)
    model = ASTForAudioClassification(ASTConfig(**_CONFIG))
    rehead_model(model, ADAPT_CLASSES)
    with torch.no_grad():
        for parameter in model.classifier.parameters():
            parameter.copy_(torch.randn_like(parameter))
    pipe = ASTAudioClassificationPipeline(
        _runner=lambda _: np.zeros(len(ADAPT_CLASSES), dtype=np.float32),
        labels=list(ADAPT_CLASSES),
        device="cpu",
        model=model,
        extractor=object(),
        activation="softmax",
    )
    pipe.adaptation_config.update(
        {"epochs": 2, "learning_rate": np.float32(1e-3), "history": [{"loss": np.float64(0.5)}]}
    )
    return pipe


def _fresh_pipe() -> ASTAudioClassificationPipeline:
    torch.manual_seed(123)
    return ASTAudioClassificationPipeline(
        _runner=lambda _: np.zeros(527, dtype=np.float32),
        labels=[f"label-{i}" for i in range(527)],
        device="cpu",
        model=ASTForAudioClassification(ASTConfig(**_CONFIG)),
        extractor=object(),
    )


def test_round_trip_restores_identical_tensors_and_outputs(tmp_path):
    pipe = _adapted_pipe()
    saved = pipe.save_artifact(tmp_path / "adapter.safetensors")
    fresh = _fresh_pipe()
    fresh.load_artifact(saved)
    original = pipe.model.classifier.state_dict()
    restored = fresh.model.classifier.state_dict()
    assert set(original) == set(restored)
    for name, tensor in original.items():
        assert torch.equal(tensor, restored[name]), name
    assert fresh.labels == list(ADAPT_CLASSES)
    assert fresh.activation == "softmax"
    features = torch.randn(4, _CONFIG["hidden_size"])
    pipe.model.eval()
    with torch.no_grad():
        assert torch.equal(pipe.model.classifier(features), fresh.model.classifier(features))


def test_sidecar_is_pure_json_with_lineage_and_digest(tmp_path):
    saved = _adapted_pipe().save_artifact(tmp_path / "adapter.safetensors")
    sidecar_path = artifact_sidecar_path(saved)
    assert sidecar_path.name == "adapter.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["format"] == ARTIFACT_FORMAT
    assert sidecar["format_version"] == ARTIFACT_FORMAT_VERSION == "2.0"
    assert sidecar["weights_format"] == "safetensors"
    assert sidecar["weights_file"] == "adapter.safetensors"
    assert sidecar["weights_sha256"] == hashlib.sha256(saved.read_bytes()).hexdigest()
    assert sidecar["base_model_id"] == MODEL_ID and sidecar["base_model_revision"] == MODEL_REVISION
    assert sidecar["class_names"] == list(ADAPT_CLASSES)
    # numpy scalars in the adaptation settings are written as plain JSON numbers.
    assert isinstance(sidecar["adaptation"]["learning_rate"], float)
    assert sidecar["adaptation"]["history"] == [{"loss": 0.5}]


def test_digest_mismatch_is_refused_before_the_weights_are_read(tmp_path, monkeypatch):
    saved = _adapted_pipe().save_artifact(tmp_path / "adapter.safetensors")
    other = _adapted_pipe(seed=7).save_artifact(tmp_path / "other.safetensors")
    saved.write_bytes(other.read_bytes())  # substitute weights; the sidecar still names the old digest
    import safetensors.torch

    def _must_not_load(*args, **kwargs):
        raise AssertionError("load_file was reached before the digest check")

    monkeypatch.setattr(safetensors.torch, "load_file", _must_not_load)
    fresh = _fresh_pipe()
    original_classifier = fresh.model.classifier
    with pytest.raises(ValueError, match="digest mismatch"):
        fresh.load_artifact(saved)
    assert fresh.model.classifier is original_classifier


def test_flipped_weight_byte_is_refused(tmp_path):
    saved = _adapted_pipe().save_artifact(tmp_path / "adapter.safetensors")
    data = bytearray(saved.read_bytes())
    data[-1] ^= 0x01
    saved.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="digest mismatch"):
        _fresh_pipe().load_artifact(saved)


def test_missing_or_digestless_sidecar_is_refused(tmp_path):
    saved = _adapted_pipe().save_artifact(tmp_path / "adapter.safetensors")
    sidecar_path = artifact_sidecar_path(saved)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    del sidecar["weights_sha256"]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="records no weights_sha256"):
        _fresh_pipe().load_artifact(saved)
    sidecar_path.unlink()
    with pytest.raises(ValueError, match="sidecar 'adapter.json' is missing"):
        _fresh_pipe().load_artifact(saved)


def test_no_pickle_or_partial_file_is_written(tmp_path):
    _adapted_pipe().save_artifact(tmp_path / "adapter.safetensors")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["adapter.json", "adapter.safetensors"]


@pytest.mark.parametrize("name", ["adapter.pt", "adapter.pth", "adapter.bin", "adapter.ckpt", "adapter"])
def test_pickle_paths_are_refused_on_save_and_on_load(tmp_path, name):
    with pytest.raises(ValueError, match="must end in .safetensors"):
        _adapted_pipe().save_artifact(tmp_path / name)
    assert not any(tmp_path.iterdir())
    legacy = tmp_path / name
    torch.save({"format": ARTIFACT_FORMAT}, legacy)
    with pytest.raises(ValueError, match="pickle"):
        _fresh_pipe().load_artifact(legacy)
