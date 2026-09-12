from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass
class ASTAudioClassificationPipeline:
    """Multi-label AudioSet classifier. `_runner` maps a 16 kHz float32 waveform to raw logits (527,)."""

    _runner: Callable[[np.ndarray], np.ndarray]
    labels: list[str]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ASTAudioClassificationPipeline:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), dict(local_files_only=True)
        elif allow_download:
            source, kwargs = MODEL_ID, dict(revision=MODEL_REVISION)
        else:
            raise FileNotFoundError(f"no verified snapshot at {root} and allow_download=False")
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

        return cls(runner, labels, resolved_device)

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        """Classify one clip. `audio` is a 1-D float array; `sample_rate` is the rate it was captured at."""
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
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio contains NaN or inf samples")
        duration = audio.shape[0] / sample_rate
        if duration < MIN_AUDIO_SECONDS:
            raise ValueError(f"audio is {duration:.4f} s; minimum is {MIN_AUDIO_SECONDS} s")
        if duration > MAX_INPUT_SECONDS:
            raise ValueError(f"audio is {duration:.2f} s; ceiling is {MAX_INPUT_SECONDS} s (chunk it first)")

        waveform = audio.astype(np.float32, copy=False)
        resampled = sample_rate != SAMPLE_RATE
        if resampled:
            waveform = _resample(waveform, sample_rate)
        logits = np.asarray(self._runner(waveform), dtype=np.float32)
        if logits.shape != (len(self.labels),):
            raise RuntimeError(f"backend returned logits {logits.shape}, expected ({len(self.labels)},)")
        scores = 1.0 / (1.0 + np.exp(-logits))
        order = np.argsort(-scores)[:top_k]
        return {
            "predictions": [
                {"label": self.labels[int(i)], "index": int(i), "score": float(scores[i])} for i in order
            ],
            "activation": ACTIVATION,
            "truncated": duration > MAX_AUDIO_SECONDS,
            "duration_seconds": duration,
            "window_seconds": MAX_AUDIO_SECONDS,
            "resampled": resampled,
            "input_sample_rate": sample_rate,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
