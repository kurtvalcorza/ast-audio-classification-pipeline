"""Offline unit tests for the AST E2E adaptation profile."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import ASTConfig, ASTForAudioClassification

from ast_audio_classification_pipeline import (
    ADAPT_CLASSES,
    ARTIFACT_FORMAT,
    DATASET_REPRESENTATION,
    MODEL_ID,
    MODEL_REVISION,
    ASTAudioClassificationPipeline,
    confusion_matrix,
    evaluate_classification,
    freeze_backbone,
    macro_f1_score,
    majority_class_baseline,
    multiclass_accuracy,
    per_class_metrics,
    rehead_model,
    split_dataset,
    synthetic_audio_dataset,
    tutorial_tone,
    validate_dataset,
)


def _record(index: int, label: int, waveform: np.ndarray | None = None) -> dict:
    return {
        "id": f"record-{index}",
        "waveform": np.zeros(16_000, dtype=np.float32) if waveform is None else waveform,
        "sample_rate": 16_000,
        "label": label,
        "class_name": ADAPT_CLASSES[label] if 0 <= label < len(ADAPT_CLASSES) else "invalid",
    }


def test_tutorial_tone_generation():
    tone = tutorial_tone(duration=1.0, sample_rate=16_000, frequency=440.0)
    assert isinstance(tone, np.ndarray)
    assert tone.shape == (16_000,)
    assert tone.dtype == np.float32
    assert np.all(np.isfinite(tone))
    assert np.max(np.abs(tone)) == pytest.approx(0.5, abs=1e-2)
    with pytest.raises(ValueError, match="amplitude"):
        tutorial_tone(amplitude=1.1)


def test_synthetic_audio_dataset_contract_and_balance():
    records = synthetic_audio_dataset(n_samples=24, seed=42)
    assert len(records) == 24
    manifest = validate_dataset(records, ADAPT_CLASSES)
    assert manifest["verdict"] == "accepted"
    assert manifest["num_classes"] == 3
    assert manifest["class_counts"]["geophony"] == 8
    assert manifest["class_counts"]["biophony"] == 8
    assert manifest["class_counts"]["anthrophony"] == 8

    # Check reproducibility
    records_rep = synthetic_audio_dataset(n_samples=24, seed=42)
    np.testing.assert_array_equal(records[0]["waveform"], records_rep[0]["waveform"])
    records_other_seed = synthetic_audio_dataset(n_samples=24, seed=43)
    assert not np.array_equal(records[0]["waveform"], records_other_seed[0]["waveform"])
    assert manifest["representation"] == DATASET_REPRESENTATION
    with pytest.raises(ValueError, match="divisible"):
        synthetic_audio_dataset(n_samples=7)


def test_split_dataset_stratification():
    records = synthetic_audio_dataset(n_samples=24, seed=42)
    train, val = split_dataset(records, val_fraction=0.25, seed=42)
    assert len(train) == 18
    assert len(val) == 6

    # Verify no overlap between train and val
    train_ids = {r["id"] for r in train}
    val_ids = {r["id"] for r in val}
    assert train_ids.isdisjoint(val_ids)

    # Verify each split validates independently
    assert validate_dataset(train, ADAPT_CLASSES)["verdict"] == "accepted"
    assert validate_dataset(val, ADAPT_CLASSES)["verdict"] == "accepted"


def test_validate_dataset_rejects_malformed_inputs():
    with pytest.raises(TypeError, match="must be a sequence"):
        validate_dataset("not a sequence")

    with pytest.raises(ValueError, match="at least 2 records"):
        validate_dataset([_record(0, 0)])

    # Missing keys
    with pytest.raises(KeyError, match="missing required key 'waveform'"):
        validate_dataset([
            {"id": "a", "sample_rate": 16_000, "label": 0},
            _record(1, 1),
        ])
    with pytest.raises(KeyError, match="missing required key 'label'"):
        records = [_record(0, 0), _record(1, 1)]
        records[0].pop("label")
        validate_dataset(records)
    with pytest.raises(KeyError, match="missing required key 'class_name'"):
        records = [_record(0, 0), _record(1, 1), _record(2, 2)]
        records[0].pop("class_name")
        validate_dataset(records)

    # 2-D waveform rejected
    with pytest.raises(ValueError, match="1-D mono"):
        validate_dataset([
            _record(0, 0, np.zeros((2, 16_000), dtype=np.float32)),
            _record(1, 1, np.zeros((2, 16_000), dtype=np.float32)),
        ])

    # Non-finite values
    nan_wave = np.zeros(16000, dtype=np.float32)
    nan_wave[10] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        validate_dataset([
            _record(0, 0, nan_wave),
            _record(1, 1),
        ])

    # Out of range label
    with pytest.raises(ValueError, match="outside valid range"):
        validate_dataset([
            _record(0, 0),
            _record(1, 99),
        ])

    clipped = _record(0, 0)
    clipped["waveform"][0] = 1.1
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        validate_dataset([clipped, _record(1, 1), _record(2, 2)])

    with pytest.raises(TypeError, match="float32"):
        validate_dataset([
            _record(0, 0, np.zeros(16_000, dtype=np.float64)),
            _record(1, 1),
            _record(2, 2),
        ])

    duplicate = [_record(0, 0), _record(1, 1), _record(2, 2)]
    duplicate[1]["id"] = duplicate[0]["id"]
    with pytest.raises(ValueError, match="duplicates id"):
        validate_dataset(duplicate)


def test_split_dataset_rejects_invalid_boundaries():
    records = synthetic_audio_dataset(n_samples=6)
    with pytest.raises(ValueError, match="strictly between"):
        split_dataset(records, val_fraction=1.0)
    with pytest.raises(ValueError, match="at least 2 records"):
        split_dataset(records[:3])


def test_metrics_computation():
    predictions = [0, 1, 2, 0, 1, 2]
    targets = [0, 1, 1, 0, 1, 2]
    num_classes = 3
    class_names = ("c0", "c1", "c2")

    acc = multiclass_accuracy(predictions, targets)
    assert acc == pytest.approx(5 / 6)

    cm = confusion_matrix(predictions, targets, num_classes)
    assert cm == [
        [2, 0, 0],
        [0, 2, 1],
        [0, 0, 1],
    ]

    pcm = per_class_metrics(predictions, targets, num_classes)
    assert pcm[0]["precision"] == 1.0
    assert pcm[0]["recall"] == 1.0
    assert pcm[0]["f1"] == 1.0

    macro_f1 = macro_f1_score(predictions, targets, num_classes)
    assert 0.0 <= macro_f1 <= 1.0

    baseline = majority_class_baseline(targets, num_classes)
    assert baseline["majority_class_index"] == 1
    assert baseline["majority_class_count"] == 3
    assert baseline["majority_class_accuracy"] == 0.5

    eval_report = evaluate_classification(predictions, targets, class_names)
    assert eval_report["accuracy"] == pytest.approx(5 / 6, abs=1e-3)
    assert eval_report["baseline"]["majority_class_accuracy"] == 0.5
    assert eval_report["accuracy_delta_vs_baseline"] > 0

    with pytest.raises(ValueError, match="outside"):
        confusion_matrix([3], [0], num_classes)
    with pytest.raises(ValueError, match="unique"):
        evaluate_classification([0], [0], ("same", "same"))


def test_rehead_model_and_freeze_backbone():
    config = ASTConfig(num_labels=527)
    model = ASTForAudioClassification(config)

    # Initial state
    assert model.classifier.dense.out_features == 527
    initial_backbone_grad = all(p.requires_grad for p in model.audio_spectrogram_transformer.parameters())
    assert initial_backbone_grad is True

    # Re-head
    rehead_model(model, ADAPT_CLASSES, seed=123)
    assert model.classifier.dense.out_features == 3
    assert model.num_labels == 3
    assert model.config.num_labels == 3
    assert model.config.id2label[0] == "geophony"
    assert model.config.id2label[1] == "biophony"
    assert model.config.id2label[2] == "anthrophony"

    # Freeze backbone
    frozen_count = freeze_backbone(model)
    assert frozen_count > 80_000_000
    assert all(not p.requires_grad for p in model.audio_spectrogram_transformer.parameters())
    # Head parameters remain trainable
    assert model.classifier.dense.weight.requires_grad is True
    assert model.classifier.dense.bias.requires_grad is True


def test_pipeline_rehead_normalizes_class_names():
    model = ASTForAudioClassification(ASTConfig(num_labels=527))
    pipe = ASTAudioClassificationPipeline(
        _runner=lambda _: np.zeros(2, dtype=np.float32),
        labels=["old"],
        device="cpu",
        model=model,
        extractor=object(),
    )
    pipe.rehead((" first ", "second"))
    assert pipe.labels == ["first", "second"]
    assert pipe.adaptation_config["class_names"] == ["first", "second"]


def test_artifact_save_and_safe_reload(tmp_path):
    model = ASTForAudioClassification(ASTConfig(num_labels=527))
    rehead_model(model, ADAPT_CLASSES)

    # Inject dummy weights into dense layer
    with torch.no_grad():
        model.classifier.dense.weight.fill_(0.42)
        model.classifier.dense.bias.fill_(0.11)

    labels = list(ADAPT_CLASSES)
    pipe = ASTAudioClassificationPipeline(
        _runner=lambda w: np.array([0.1, 0.2, 0.3], dtype=np.float32),
        labels=labels,
        device="cpu",
        model=model,
        extractor=object(),  # stub
        activation="softmax",
    )

    artifact_file = tmp_path / "ast-audio-adapter-v1.pt"
    saved = pipe.save_artifact(artifact_file)
    assert saved.is_file()
    assert saved.stat().st_size < 1_000_000
    payload = torch.load(saved, weights_only=True)
    assert payload["format"] == ARTIFACT_FORMAT
    assert "classifier_state_dict" in payload
    assert "state_dict" not in payload

    # Create fresh model with random weights
    fresh_model = ASTForAudioClassification(ASTConfig(num_labels=527))
    fresh_pipe = ASTAudioClassificationPipeline(
        _runner=lambda w: np.array([0.0, 0.0, 0.0], dtype=np.float32),
        labels=["dummy"],
        device="cpu",
        model=fresh_model,
        extractor=pipe.extractor,
    )

    # Load artifact
    fresh_pipe.load_artifact(saved)
    assert fresh_pipe.labels == labels
    assert fresh_pipe.activation == "softmax"
    assert fresh_pipe.model.classifier.dense.out_features == 3
    assert torch.allclose(
        fresh_pipe.model.classifier.dense.weight,
        torch.full_like(fresh_pipe.model.classifier.dense.weight, 0.42),
    )
    assert torch.allclose(
        fresh_pipe.model.classifier.dense.bias,
        torch.full_like(fresh_pipe.model.classifier.dense.bias, 0.11),
    )


class _TinyExtractor:
    def __call__(self, waveforms, *, sampling_rate, return_tensors):
        assert sampling_rate == 16_000
        assert return_tensors == "pt"
        batch = len(waveforms) if isinstance(waveforms, list) else 1
        values = torch.linspace(-1.0, 1.0, 16 * 16).reshape(16, 16)
        return {"input_values": values.repeat(batch, 1, 1)}


def test_finetune_runs_bounded_gradient_adaptation():
    config = ASTConfig(
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
    model = ASTForAudioClassification(config)
    pipe = ASTAudioClassificationPipeline(
        _runner=lambda _: np.zeros(3, dtype=np.float32),
        labels=list(ADAPT_CLASSES),
        device="cpu",
        model=model,
        extractor=_TinyExtractor(),
        activation="softmax",
    )
    rehead_model(model, ADAPT_CLASSES)
    records = synthetic_audio_dataset(n_samples=12, duration=0.05)
    train, val = split_dataset(records, val_fraction=0.25)

    with pytest.raises(RuntimeError, match="call freeze_backbone"):
        pipe.finetune(train, epochs=1)

    pipe.freeze_backbone()
    history = pipe.finetune(train, val, epochs=2, batch_size=3, learning_rate=1e-3)
    assert len(history) == 2
    assert all(item["steps"] == 3 for item in history)
    assert pipe.adaptation_config["epochs"] == 2
    assert pipe.adaptation_config["train_records"] == 9


def test_artifact_safe_reload_rejects_corrupted_payload(tmp_path):
    bad_artifact = tmp_path / "bad.pt"
    torch.save({"format": "unknown.format"}, bad_artifact)

    config = ASTConfig(num_labels=527)
    model = ASTForAudioClassification(config)
    pipe = ASTAudioClassificationPipeline(
        _runner=lambda w: np.zeros(3),
        labels=list(ADAPT_CLASSES),
        device="cpu",
        model=model,
        extractor=object(),
    )

    with pytest.raises(ValueError, match="unrecognized artifact format"):
        pipe.load_artifact(bad_artifact)

    torch.save(
        {
            "format": ARTIFACT_FORMAT,
            "format_version": "1.0",
            "artifact_kind": "full-model",
        },
        bad_artifact,
    )
    with pytest.raises(ValueError, match="artifact_kind"):
        pipe.load_artifact(bad_artifact)


def test_artifact_rejection_does_not_mutate_live_classifier(tmp_path):
    config = ASTConfig(num_labels=527)
    model = ASTForAudioClassification(config)
    original_classifier = model.classifier
    pipe = ASTAudioClassificationPipeline(
        _runner=lambda w: np.zeros(527, dtype=np.float32),
        labels=[f"label-{i}" for i in range(527)],
        device="cpu",
        model=model,
        extractor=object(),
    )
    bad_artifact = tmp_path / "bad-state.pt"
    torch.save(
        {
            "format": ARTIFACT_FORMAT,
            "format_version": "1.0",
            "artifact_kind": "classifier-head-adapter",
            "base_model_id": MODEL_ID,
            "base_model_revision": MODEL_REVISION,
            "class_names": list(ADAPT_CLASSES),
            "num_classes": len(ADAPT_CLASSES),
            "activation": "softmax",
            "adaptation": {},
            "classifier_state_dict": {},
        },
        bad_artifact,
    )

    with pytest.raises(RuntimeError, match="Missing key"):
        pipe.load_artifact(bad_artifact)

    assert pipe.model.classifier is original_classifier
    assert pipe.model.config.num_labels == 527
    assert pipe.activation == "sigmoid"
    assert len(pipe.labels) == 527


def test_adapted_predict_default_uses_active_label_count():
    pipe = ASTAudioClassificationPipeline(
        _runner=lambda w: np.array([0.1, 0.2, 0.3], dtype=np.float32),
        labels=list(ADAPT_CLASSES),
        device="cpu",
        activation="softmax",
    )
    audio = np.zeros(16_000, dtype=np.float32)

    result = pipe.predict(audio, 16_000)
    assert len(result["predictions"]) == len(ADAPT_CLASSES)
    with pytest.raises(ValueError, match="exceeds the active label count"):
        pipe.predict(audio, 16_000, top_k=5)
