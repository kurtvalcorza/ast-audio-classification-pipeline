"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import numpy as np
import pytest

from ast_audio_classification_pipeline import (
    DEFAULT_TOP_K,
    INPUT_SCHEMA,
    MAX_AUDIO_SECONDS,
    MAX_INPUT_SECONDS,
    MIN_AUDIO_SECONDS,
    MODEL_ID,
    MODEL_REVISION,
    NUM_LABELS,
    SAMPLE_RATE,
    evaluation_report,
    validate_inputs,
)


def _tone(seconds: float = 1.0, sample_rate: int = SAMPLE_RATE, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _result(n: int = 5, truncated: bool = False) -> dict:
    return {
        "predictions": [{"label": f"label-{i}", "index": i, "score": 0.9 - 0.1 * i} for i in range(n)],
        "activation": "sigmoid",
        "truncated": truncated,
        "duration_seconds": 3.0,
        "window_seconds": MAX_AUDIO_SECONDS,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs([_tone(1.0), _tone(0.5)], SAMPLE_RATE, names=["a", "b"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["duration_seconds"] == [MIN_AUDIO_SECONDS, MAX_INPUT_SECONDS]
    assert manifest["schema"]["top_k"] == [1, NUM_LABELS]
    assert manifest["schema"]["model_window_seconds"] == MAX_AUDIO_SECONDS
    assert manifest["sample_rate"] == SAMPLE_RATE
    assert manifest["top_k"] == DEFAULT_TOP_K
    assert manifest["n_clips"] == 2
    assert [entry["id"] for entry in manifest["inputs"]] == ["a", "b"]
    first = manifest["inputs"][0]
    assert first["samples"] == SAMPLE_RATE
    assert first["duration_seconds"] == 1.0
    assert first["peak_amplitude"] == pytest.approx(0.5, abs=1e-3)
    assert first["will_resample"] is False
    assert first["will_truncate"] is False
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_flags_resampling_and_truncation_without_rejecting() -> None:
    manifest = validate_inputs(_tone(MAX_AUDIO_SECONDS + 2.0, 8_000), 8_000)
    (entry,) = manifest["inputs"]
    assert entry["id"] == "clip-0"
    assert entry["will_resample"] is True
    assert entry["will_truncate"] is True
    assert manifest["verdict"] == "accepted"


def test_validate_inputs_rejects_like_predict() -> None:
    with pytest.raises(ValueError, match="minimum is"):
        validate_inputs(_tone(MIN_AUDIO_SECONDS / 2), SAMPLE_RATE)
    with pytest.raises(ValueError, match="ceiling is"):
        validate_inputs(np.zeros(int((MAX_INPUT_SECONDS + 1) * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(ValueError, match="1-D mono"):
        validate_inputs(np.zeros((2, SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(TypeError, match="float array"):
        validate_inputs(np.zeros(SAMPLE_RATE, dtype=np.int16), SAMPLE_RATE)
    with pytest.raises(TypeError, match="sample_rate"):
        validate_inputs(_tone(), 16_000.0)
    with pytest.raises(ValueError, match="top_k"):
        validate_inputs(_tone(), SAMPLE_RATE, top_k=NUM_LABELS + 1)
    with pytest.raises(ValueError, match="NaN or inf"):
        validate_inputs(np.full(SAMPLE_RATE, np.nan, dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(TypeError, match="numpy.ndarray"):
        validate_inputs("not audio", SAMPLE_RATE)
    with pytest.raises(ValueError, match="names must have one entry per waveform"):
        validate_inputs([_tone()], SAMPLE_RATE, names=["a", "b"])


def test_audioset_evaluation_report_is_not_measurable_without_aligned_ground_truth() -> None:
    report = evaluation_report(_result())
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_clips"] == 1
    assert report["n_scored_labels"] == 5
    assert "no AudioSet-labelled ground truth" in report["reason"]
    assert "mean average precision" in report["needs"]
    assert "0.4593" in report["needs"]
    assert "do not sum" in report["score_semantics"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_audioset_evaluation_report_rejects_unaligned_target_claims() -> None:
    report = evaluation_report(_result(), ["Speech"], sample_kind="BYOD")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["sample_kind"] == "BYOD"
    assert "not established as labels from the 527-class AudioSet ontology" in report["reason"]


def test_evaluation_report_carries_the_truncation_flag() -> None:
    assert evaluation_report(_result(truncated=True))["truncated"] is True
    assert evaluation_report(_result())["truncated"] is False


def test_evaluation_report_rejects_adapted_softmax_results() -> None:
    result = _result()
    result["activation"] = "softmax"
    with pytest.raises(ValueError, match="only supports unadapted AudioSet sigmoid results"):
        evaluation_report(result)
