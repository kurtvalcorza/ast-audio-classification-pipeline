import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from ast_audio_classification_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    MAX_AUDIO_SECONDS,
    MAX_INPUT_SECONDS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    NUM_LABELS,
    SAMPLE_RATE,
    ASTAudioClassificationPipeline,
    stage_missing_files,
    verify_snapshot,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
LABELS = [f"label-{i}" for i in range(NUM_LABELS)]


def _fake_pipeline(logits=None, calls=None):
    def runner(waveform):
        if calls is not None:
            calls.append(waveform)
        out = np.full(NUM_LABELS, -5.0, dtype=np.float32) if logits is None else logits
        return out

    return ASTAudioClassificationPipeline(runner, LABELS, "cpu")


def _sine(seconds: float, rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_identity_constants_are_40_hex_and_match_manifest():
    assert re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION)
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert manifest["modelId"] == MODEL_ID
        assert manifest["revision"] == MODEL_REVISION


def _write_snapshot(tmp_path: Path, content: bytes, sha256: str, revision: str = MODEL_REVISION) -> Path:
    (tmp_path / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": revision,
        "files": [{"path": "config.json", "bytes": len(content), "sha256": sha256}],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_verify_snapshot_accepts_matching_digest(tmp_path):
    import hashlib

    content = b'{"a": 1}'
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    assert verify_snapshot(root)["revision"] == MODEL_REVISION


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    import hashlib

    content = b'{"a": 1}'
    good = hashlib.sha256(content).hexdigest()
    bad = ("0" if good[0] != "0" else "1") + good[1:]
    root = _write_snapshot(tmp_path, content, bad)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(root)


def test_verify_snapshot_rejects_wrong_revision_and_size(tmp_path):
    import hashlib

    content = b"xyz"
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest(), revision="f" * 40)
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(root)
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    (tmp_path / "config.json").write_bytes(b"xyzw")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(root)


def test_from_pretrained_refuses_without_snapshot(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError):
        ASTAudioClassificationPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_refuses_tampered_snapshot_before_loading(tmp_path, forbid_model_imports):
    _write_snapshot(tmp_path, b'{"a": 1}', "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        ASTAudioClassificationPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_from_pretrained_valid_snapshot_reaches_model_import(tmp_path, forbid_model_imports):
    import hashlib

    content = b'{"a": 1}'
    _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    with pytest.raises(AssertionError, match="model dependency imported before rejection: torch"):
        ASTAudioClassificationPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


@pytest.mark.parametrize(
    ("audio", "rate", "top_k", "exc"),
    [
        ([0.0] * 16_000, SAMPLE_RATE, 5, TypeError),  # not an ndarray
        (np.zeros((2, 16_000), dtype=np.float32), SAMPLE_RATE, 5, ValueError),  # stereo
        (np.zeros(16_000, dtype=np.int16), SAMPLE_RATE, 5, TypeError),  # int PCM
        (np.zeros(16_000, dtype=np.float32), "16000", 5, TypeError),  # rate not int
        (np.zeros(16_000, dtype=np.float32), 0, 5, TypeError),  # rate zero
        (np.zeros(16_000, dtype=np.float32), SAMPLE_RATE, 0, ValueError),  # top_k too small
        (np.zeros(16_000, dtype=np.float32), SAMPLE_RATE, NUM_LABELS + 1, ValueError),  # top_k too big
        (np.zeros(100, dtype=np.float32), SAMPLE_RATE, 5, ValueError),  # shorter than one frame
        (np.zeros(int(MAX_INPUT_SECONDS * SAMPLE_RATE) + 1, dtype=np.float32), SAMPLE_RATE, 5, ValueError),
        (np.array([0.0, np.nan] * 8_000, dtype=np.float32), SAMPLE_RATE, 5, ValueError),  # NaN
    ],
)
def test_predict_rejects_bad_input(audio, rate, top_k, exc):
    calls = []
    pipe = _fake_pipeline(calls=calls)
    with pytest.raises(exc):
        pipe.predict(audio, rate, top_k=top_k)
    assert calls == []  # the model never ran


def test_predict_output_fields_and_sigmoid_ranking():
    logits = np.full(NUM_LABELS, -5.0, dtype=np.float32)
    logits[137] = 3.0
    logits[0] = 1.0
    pipe = _fake_pipeline(logits)
    result = pipe.predict(_sine(3.0), SAMPLE_RATE, top_k=2)
    assert [p["index"] for p in result["predictions"]] == [137, 0]
    assert result["predictions"][0]["label"] == "label-137"
    assert abs(result["predictions"][0]["score"] - 1 / (1 + np.exp(-3.0))) < 1e-6
    assert result["activation"] == "sigmoid"
    assert result["truncated"] is False
    assert result["resampled"] is False
    assert abs(result["duration_seconds"] - 3.0) < 1e-9
    assert result["window_seconds"] == MAX_AUDIO_SECONDS
    assert result["model_id"] == MODEL_ID
    assert result["model_revision"] == MODEL_REVISION


def test_predict_flags_truncation_beyond_window():
    pipe = _fake_pipeline()
    result = pipe.predict(_sine(MAX_AUDIO_SECONDS + 1.0), SAMPLE_RATE)
    assert result["truncated"] is True


def test_predict_resamples_when_rate_differs():
    calls = []
    pipe = _fake_pipeline(calls=calls)
    result = pipe.predict(_sine(1.0, rate=8_000), 8_000)
    assert result["resampled"] is True
    assert result["input_sample_rate"] == 8_000
    assert len(calls) == 1
    assert abs(calls[0].shape[0] - SAMPLE_RATE) <= 1  # 1 s at 8 kHz became ~16000 samples
    assert calls[0].dtype == np.float32


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)
