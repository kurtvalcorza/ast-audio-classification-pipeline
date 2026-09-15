from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
MODEL_LICENSE = "bsd-3-clause"
MODEL_KEY = "ast-audioset"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# The pinned checkpoint expects 16 kHz mono float32. Its feature extractor computes a 128-bin Kaldi
# fbank with a 10 ms hop and pads or crops to 1024 frames, so only the first 10.24 s of audio reach
# the model; anything longer is cropped and reported as `truncated`.
SAMPLE_RATE = 16_000
WINDOW_FRAMES = 1024
MAX_AUDIO_SECONDS = WINDOW_FRAMES * 0.010  # 10.24 s model window
MAX_INPUT_SECONDS = 120.0  # hard ceiling: longer input is rejected, the caller must chunk
MIN_AUDIO_SECONDS = 0.025  # one 25 ms fbank frame
NUM_LABELS = 527  # AudioSet ontology classes in the pinned config.json
DEFAULT_TOP_K = 5
ACTIVATION = "sigmoid"
ARTIFACT_FORMAT = "org.valcorza.ast-audio.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest.hexdigest()} != manifest {entry['sha256']}")
    return manifest


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _resample(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    import torch
    import torchaudio.functional as af

    resampled = af.resample(torch.from_numpy(waveform), orig_freq=sample_rate, new_freq=SAMPLE_RATE)
    return resampled.numpy().astype(np.float32)


INPUT_SCHEMA: dict[str, Any] = {
    "input": "1-D float numpy array of mono samples in [-1, 1], plus the sample_rate it was captured at",
    "sample_rate_hz": f"any positive int; resampled to SAMPLE_RATE={SAMPLE_RATE} when it differs",
    "duration_seconds": [MIN_AUDIO_SECONDS, MAX_INPUT_SECONDS],
    "model_window_seconds": MAX_AUDIO_SECONDS,
    "top_k": [1, NUM_LABELS],
    "preprocessing": (
        f"resample to {SAMPLE_RATE} Hz when needed, 128-bin Kaldi fbank with a 10 ms hop, "
        f"pad or crop to {WINDOW_FRAMES} frames ({MAX_AUDIO_SECONDS} s)"
    ),
}


def _check_inputs(audio: Any, sample_rate: Any, top_k: Any) -> float:
    """Raise TypeError/ValueError naming the first violated ceiling; return the clip duration in seconds."""
    if not isinstance(audio, np.ndarray):
        raise TypeError(f"audio must be a numpy.ndarray, got {type(audio).__name__}")
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D mono, got shape {audio.shape}")
    if not np.issubdtype(audio.dtype, np.floating):
        raise TypeError(f"audio must be a float array in [-1, 1], got dtype {audio.dtype}")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise TypeError("sample_rate must be a positive int (the rate the audio was captured at)")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= NUM_LABELS:
        raise ValueError(f"top_k must be an int in [1, {NUM_LABELS}]")
    duration = audio.shape[0] / sample_rate
    if duration < MIN_AUDIO_SECONDS:
        raise ValueError(f"audio is {duration:.4f} s; minimum is {MIN_AUDIO_SECONDS} s")
    if duration > MAX_INPUT_SECONDS:
        raise ValueError(f"audio is {duration:.2f} s; ceiling is {MAX_INPUT_SECONDS} s (chunk it first)")
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains NaN or inf samples")
    if np.max(np.abs(audio)) > 1.0:
        raise ValueError("audio samples must be in [-1, 1]")
    return duration


def validate_inputs(
    waveforms: Any,
    sample_rate: Any,
    *,
    top_k: int = DEFAULT_TOP_K,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-clip observations, verdict).

    The checks are the ones ``predict`` applies, through the same private ``_check_inputs``, so a
    rejection here raises exactly what ``predict`` would; a caller that wants the finding recorded
    catches the exception and stores ``str(exc)`` under ``findings``.
    """
    batch = [waveforms] if isinstance(waveforms, np.ndarray) else waveforms
    if not isinstance(batch, Sequence) or isinstance(batch, str | bytes):
        raise TypeError("waveforms must be a 1-D numpy.ndarray or a sequence of them")
    if len(batch) < 1:
        raise ValueError("at least one waveform is required")
    if names is not None and len(names) != len(batch):
        raise ValueError("names must have one entry per waveform")
    inputs = []
    for index, waveform in enumerate(batch):
        duration = _check_inputs(waveform, sample_rate, top_k)
        inputs.append(
            {
                "id": names[index] if names else f"clip-{index}",
                "samples": int(waveform.shape[0]),
                "dtype": str(waveform.dtype),
                "duration_seconds": round(duration, 4),
                "peak_amplitude": round(float(np.max(np.abs(waveform))), 6),
                "will_resample": sample_rate != SAMPLE_RATE,
                "will_truncate": duration > MAX_AUDIO_SECONDS,
            }
        )
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": inputs,
        "sample_rate": sample_rate,
        "top_k": top_k,
        "n_clips": len(inputs),
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    targets: Sequence[Any] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Describe the unadapted AudioSet inference result without inventing ground truth."""
    activation = result.get("activation", ACTIVATION)
    if activation != ACTIVATION:
        raise ValueError(
            "evaluation_report only supports unadapted AudioSet sigmoid results; "
            f"got activation={activation!r}"
        )
    predictions = result["predictions"]
    reason = "no AudioSet-labelled ground truth exists for the evaluated clip"
    if targets is not None:
        reason = (
            "targets were supplied, but they are not established as labels from the 527-class "
            "AudioSet ontology; "
            "score only ontology-aligned labels with an AudioSet-convention mAP implementation"
        )
    return {
        "task": "multi-label audio event classification over the 527 AudioSet labels",
        "score_semantics": (
            f"independent {activation} score per label: the scores do not sum "
            "to one, several labels can be high at once, none is a calibrated probability, and no "
            "threshold is shipped"
        ),
        "sample_kind": sample_kind,
        "n_clips": 1,
        "n_scored_labels": len(predictions),
        "metrics": [],
        "baselines": [],
        "verdict": "not-measurable",
        "reason": reason,
        "needs": (
            f"clips labelled against the same {NUM_LABELS}-label AudioSet ontology, scored over the full "
            "score vector (predict(..., top_k=527)) with mean average precision plus per-label precision "
            "and recall at a threshold chosen on your own labelled clips; the '0.4593' in the checkpoint "
            "name is the upstream-reported AudioSet mAP and is not measured here"
        ),
        "truncated": bool(result.get("truncated", False)),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def _build_reheaded_classifier(
    model: Any,
    class_names: Sequence[str],
    seed: int = 42,
) -> tuple[Any, list[str]]:
    """Build a re-headed classifier without mutating the live model."""
    import copy

    import torch
    import torch.nn as nn

    names = [str(name).strip() for name in class_names]
    if len(names) < 2 or any(not name for name in names):
        raise ValueError("class_names must contain at least 2 non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    classifier = copy.deepcopy(model.classifier)
    previous_dense = classifier.dense
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        new_dense = nn.Linear(previous_dense.in_features, len(names), bias=True)
        nn.init.kaiming_normal_(new_dense.weight, nonlinearity="linear")
        nn.init.zeros_(new_dense.bias)
    new_dense = new_dense.to(device=previous_dense.weight.device, dtype=previous_dense.weight.dtype)
    classifier.dense = new_dense
    return classifier, names


def _set_class_metadata(model: Any, names: Sequence[str]) -> None:
    """Update model label metadata after a classifier replacement succeeds."""
    model.num_labels = len(names)
    model.config.num_labels = len(names)
    model.config.id2label = dict(enumerate(names))
    model.config.label2id = {name: i for i, name in enumerate(names)}
    model.config.problem_type = "single_label_classification"


def rehead_model(
    model: Any,
    class_names: Sequence[str],
    seed: int = 42,
) -> None:
    """Dynamically replace ASTMLPHead dense layer with a new linear classifier sized to class_names."""
    classifier, names = _build_reheaded_classifier(model, class_names, seed=seed)
    model.classifier = classifier
    _set_class_metadata(model, names)


def freeze_backbone(model: Any) -> int:
    """Freeze all parameters in the audio spectrogram transformer backbone, returning frozen count."""
    frozen = 0
    for param in model.audio_spectrogram_transformer.parameters():
        param.requires_grad = False
        frozen += param.numel()
    return frozen


@dataclass
class ASTAudioClassificationPipeline:
    """Audio Spectrogram Transformer classifier for AudioSet and adapted audio tasks."""

    _runner: Callable[[np.ndarray], np.ndarray]
    labels: list[str]
    device: str
    model: Any | None = None
    extractor: Any | None = None
    activation: str = ACTIVATION
    adaptation_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ASTAudioClassificationPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), dict(local_files_only=True)
        elif allow_download:
            source, kwargs = MODEL_ID, dict(revision=MODEL_REVISION)
        else:
            raise FileNotFoundError(f"no verified snapshot at {root} and allow_download=False")
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        extractor = ASTFeatureExtractor.from_pretrained(source, trust_remote_code=False, **kwargs)
        model = ASTForAudioClassification.from_pretrained(
            source, trust_remote_code=False, dtype=torch.float32, **kwargs
        )
        model = model.to(resolved_device).eval()
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

        def runner(waveform: np.ndarray) -> np.ndarray:
            inputs = extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.inference_mode():
                logits = model(inputs["input_values"].to(resolved_device)).logits
            return logits[0].float().cpu().numpy()

        return cls(runner, labels, resolved_device, model=model, extractor=extractor)

    def rehead(self, class_names: Sequence[str], seed: int = 42) -> None:
        """Dynamically rehead the underlying model and update labels and runner."""
        if self.model is None or self.extractor is None:
            raise RuntimeError("cannot rehead a pipeline instance without an underlying torch model")
        rehead_model(self.model, class_names, seed=seed)
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.activation = "softmax"
        self.adaptation_config = {
            "mode": "classifier-head-gradient-adaptation",
            "class_names": list(self.labels),
            "rehead_seed": seed,
        }
        self.model.to(self.device)
        self._refresh_runner()

    def _refresh_runner(self) -> None:
        """Bind inference to the current model, extractor, and device."""
        if self.model is None or self.extractor is None:
            raise RuntimeError("cannot bind a runner without an underlying model and extractor")
        resolved_device = self.device
        model = self.model
        extractor = self.extractor

        def runner(waveform: np.ndarray) -> np.ndarray:
            import torch

            inputs = extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.inference_mode():
                logits = model(inputs["input_values"].to(resolved_device)).logits
            return logits[0].float().cpu().numpy()

        self._runner = runner

    def freeze_backbone(self) -> int:
        """Freeze the underlying backbone parameters."""
        if self.model is None:
            raise RuntimeError(
                "cannot freeze backbone on a pipeline instance without an underlying torch model"
            )
        frozen = freeze_backbone(self.model)
        self.adaptation_config["frozen_backbone_parameters"] = frozen
        return frozen

    def finetune(
        self,
        train_records: Sequence[dict[str, Any]],
        val_records: Sequence[dict[str, Any]] | None = None,
        *,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """In-process supervised fine-tuning using AdamW optimizer over trainable classifier parameters."""
        import random

        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW

        from .metrics import evaluate_classification
        from .samples import validate_dataset

        if self.model is None or self.extractor is None:
            raise RuntimeError("cannot fine-tune a pipeline without underlying model and extractor")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive int")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive int")
        valid_learning_rate = isinstance(learning_rate, int | float) and not isinstance(learning_rate, bool)
        if not valid_learning_rate or learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        validate_dataset(train_records, self.labels)
        if val_records is not None:
            validate_dataset(val_records, self.labels)
        if any(param.requires_grad for param in self.model.audio_spectrogram_transformer.parameters()):
            raise RuntimeError("backbone is trainable; call freeze_backbone() before finetune()")

        torch.manual_seed(seed)
        device = torch.device(self.device)
        self.model.to(device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("model has no trainable parameters")
        optimizer = AdamW(trainable_params, lr=learning_rate)

        def cached_features(
            records: Sequence[dict[str, Any]],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            feature_batches: list[torch.Tensor] = []
            self.model.audio_spectrogram_transformer.eval()
            with torch.inference_mode():
                for start in range(0, len(records), batch_size):
                    batch = records[start : start + batch_size]
                    inputs = self.extractor(
                        [record["waveform"] for record in batch],
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                    )
                    backbone_output = self.model.audio_spectrogram_transformer(
                        inputs["input_values"].to(device)
                    )
                    feature_batches.append(backbone_output.pooler_output.detach())
            targets = torch.tensor(
                [record["label"] for record in records],
                dtype=torch.long,
                device=device,
            )
            return torch.cat(feature_batches), targets

        train_features, train_targets = cached_features(train_records)
        val_features: torch.Tensor | None = None
        val_targets: torch.Tensor | None = None
        if val_records:
            val_features, val_targets = cached_features(val_records)

        history: list[dict[str, Any]] = []
        n_samples = len(train_records)

        for epoch in range(1, epochs + 1):
            self.model.classifier.train()
            indices = list(range(n_samples))
            rng = random.Random(seed + epoch * 13)
            rng.shuffle(indices)

            running_loss = 0.0
            steps = 0

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i : i + batch_size]
                index_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model.classifier(train_features.index_select(0, index_tensor))
                loss = F.cross_entropy(logits, train_targets.index_select(0, index_tensor))
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                steps += 1

            epoch_train_loss = running_loss / steps if steps > 0 else 0.0
            epoch_data: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(float(epoch_train_loss), 4),
                "steps": steps,
            }

            if val_features is not None and val_targets is not None:
                self.model.classifier.eval()
                with torch.inference_mode():
                    predictions = self.model.classifier(val_features).argmax(dim=-1)
                eval_metrics = evaluate_classification(
                    predictions.tolist(), val_targets.tolist(), self.labels
                )
                epoch_data["val_accuracy"] = eval_metrics["accuracy"]
                epoch_data["val_macro_f1"] = eval_metrics["macro_f1"]

            history.append(epoch_data)

        self.model.eval()
        self.adaptation_config.update(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": float(learning_rate),
                "training_seed": seed,
                "train_records": len(train_records),
                "validation_records": len(val_records or []),
                "feature_cache": "frozen-backbone-pooler-output",
                "history": history,
            }
        )
        return history

    def evaluate(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate the pipeline on a sequence of labelled audio records."""
        from .metrics import evaluate_classification
        from .samples import validate_dataset

        validate_dataset(records, self.labels)
        if self.model is not None:
            self.model.eval()
        predictions: list[int] = []
        targets: list[int] = []

        for record in records:
            waveform = record["waveform"]
            sample_rate = record.get("sample_rate", SAMPLE_RATE)
            result = self.predict(waveform, sample_rate=sample_rate, top_k=1)
            pred_idx = result["predictions"][0]["index"]
            predictions.append(pred_idx)
            targets.append(record["label"])

        return evaluate_classification(predictions, targets, self.labels)

    def save_artifact(self, output_path: str | Path) -> Path:
        """Export adapted weights, vocabulary, base model lineage, and artifact metadata."""
        import torch

        if self.model is None:
            raise RuntimeError("cannot save artifact without an underlying torch model")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if self.activation != "softmax" or len(self.labels) == NUM_LABELS:
            raise RuntimeError("artifact export requires an adapted single-label classifier")

        payload = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "artifact_kind": "classifier-head-adapter",
            "base_model_id": MODEL_ID,
            "base_model_revision": MODEL_REVISION,
            "class_names": list(self.labels),
            "num_classes": len(self.labels),
            "activation": self.activation,
            "adaptation": dict(self.adaptation_config),
            "classifier_state_dict": self.model.classifier.state_dict(),
        }
        torch.save(payload, out)
        return out

    def load_artifact(self, artifact_path: str | Path) -> None:
        """Load an adapted artifact into the existing pipeline safely using weights_only=True."""
        import torch

        if self.model is None or self.extractor is None:
            raise RuntimeError("cannot load artifact into a pipeline without an underlying torch model")

        payload = torch.load(artifact_path, map_location=self.device, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"expected artifact dict, got {type(payload).__name__}")
        if payload.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"unrecognized artifact format: {payload.get('format')}")
        if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(f"unsupported artifact format_version: {payload.get('format_version')}")
        if payload.get("artifact_kind") != "classifier-head-adapter":
            raise ValueError(f"unsupported artifact_kind: {payload.get('artifact_kind')}")
        if payload.get("base_model_id") != MODEL_ID:
            raise ValueError(f"base_model_id mismatch: {payload.get('base_model_id')} != {MODEL_ID}")
        if payload.get("base_model_revision") != MODEL_REVISION:
            raise ValueError(
                f"base_model_revision mismatch: {payload.get('base_model_revision')} != {MODEL_REVISION}"
            )
        if payload.get("activation") != "softmax":
            raise ValueError(f"unsupported artifact activation: {payload.get('activation')}")

        class_names = payload.get("class_names")
        if not isinstance(class_names, list) or payload.get("num_classes") != len(class_names):
            raise ValueError("artifact class_names and num_classes are inconsistent")
        if any(not isinstance(name, str) or not name.strip() for name in class_names):
            raise ValueError("artifact class_names must contain non-empty strings")
        if len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError("artifact class_names must contain at least 2 unique names")
        classifier_state = payload.get("classifier_state_dict")
        if not isinstance(classifier_state, dict):
            raise ValueError("artifact classifier_state_dict must be a dict")
        adaptation = payload.get("adaptation", {})
        if not isinstance(adaptation, dict):
            raise ValueError("artifact adaptation metadata must be a dict")

        # Build and validate off to the side so a rejected artifact cannot leave
        # the live pipeline with a partially replaced classifier.
        classifier, normalized_names = _build_reheaded_classifier(self.model, class_names)
        classifier.load_state_dict(classifier_state, strict=True)

        self.model.classifier = classifier
        _set_class_metadata(self.model, normalized_names)
        self.labels = normalized_names
        self.activation = "softmax"
        self.adaptation_config = dict(adaptation)
        self.model.to(self.device).eval()
        self._refresh_runner()

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        weights_dir: str | Path | None = None,
        device: str | None = None,
        allow_download: bool = False,
    ) -> ASTAudioClassificationPipeline:
        """Instantiate a base pipeline and load an adapted artifact."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_path)
        return pipe

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Classify one clip. `audio` is a 1-D float array; `sample_rate` is the rate it was captured at."""
        resolved_top_k = min(DEFAULT_TOP_K, len(self.labels)) if top_k is None else top_k
        duration = _check_inputs(audio, sample_rate, resolved_top_k)
        if resolved_top_k > len(self.labels):
            raise ValueError(
                f"top_k={resolved_top_k} exceeds the active label count ({len(self.labels)})"
            )
        waveform = audio.astype(np.float32, copy=False)
        resampled = sample_rate != SAMPLE_RATE
        if resampled:
            waveform = _resample(waveform, sample_rate)
        logits = np.asarray(self._runner(waveform), dtype=np.float32)
        if logits.shape != (len(self.labels),):
            raise RuntimeError(f"backend returned logits {logits.shape}, expected ({len(self.labels)},)")
        if self.activation == "sigmoid":
            scores = 1.0 / (1.0 + np.exp(-logits))
        elif self.activation == "softmax":
            shifted = logits - np.max(logits)
            exp_scores = np.exp(shifted)
            scores = exp_scores / np.sum(exp_scores)
        else:
            raise RuntimeError(f"unsupported activation: {self.activation}")
        order = np.argsort(-scores)[:resolved_top_k]
        return {
            "predictions": [
                {"label": self.labels[int(i)], "index": int(i), "score": float(scores[i])} for i in order
            ],
            "activation": self.activation,
            "truncated": duration > MAX_AUDIO_SECONDS,
            "duration_seconds": duration,
            "window_seconds": MAX_AUDIO_SECONDS,
            "resampled": resampled,
            "input_sample_rate": sample_rate,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
