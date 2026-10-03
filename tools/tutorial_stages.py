"""Stage runner for the standalone AST acoustic-ecology tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --weights WEIGHTS_DIR --stage dataset [--zip PATH]

Nothing is installed into the notebook kernel, so a hosted runtime's preloaded packages are never replaced and no
restart is needed. Each stage is a separate process and starts from files only: the verified snapshot under
``--weights``, the dataset written by ``dataset``, the adapter artifact written by ``finetune`` and the JSON records of
earlier stages. Learner-facing exports go to ``RUN_DIR/outputs``; hand-off state goes to ``RUN_DIR/state``. On failure
a stage writes ``RUN_DIR/state/<stage>.error.json`` with the exception type and message, which the notebook re-raises
in the kernel.

Stages: weights → runtime → pretrained → dataset → baseline → finetune → evaluate → reload → bundle, plus the optional
``activity``. The package API does the work; this runner only sequences it and adds the tutorial's own data hygiene
(per-record variation of the generated clips, duplicate refusal, archive reading for BYOD).
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import sys
import time
import traceback
import wave
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

STEM = "ast_audio_classification"
SEED = 42
N_RECORDS = 24
VAL_FRACTION = 0.25
TOP_K_BASE = 5
UNSEEN_INDEX = 99
UNSEEN_CLASS = 1  # biophony
ADAPTER = "ast-audio-adapter-v1.pt"
LOCK = "requirements.txt"
SNAPSHOT_KEY = "ast-audioset"
PARITY_RTOL = 1e-5
PARITY_ATOL = 1e-6
ZIP_MAX_BYTES = 64 * 1024**2
ZIP_MAX_EXPANDED = 256 * 1024**2
ZIP_MAX_FILES = 100
SCORE_SEMANTICS = "uncalibrated softmax scores over the target classes; the predicted label is the argmax"
DECISION_RULE = "argmax: the class with the highest score is the prediction; no threshold is applied and no abstention is possible"


# --------------------------------------------------------------------------------------------------
# run context and small helpers
# --------------------------------------------------------------------------------------------------


class Run:
    """Paths of one run: carried sources and state under ``root``, the snapshot under ``weights``."""

    def __init__(self, root: Path, weights: Path, options: argparse.Namespace) -> None:
        self.root = root
        self.weights = weights
        self.options = options
        self.out = root / "outputs"
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    @property
    def snapshot(self) -> Path:
        return self.weights / SNAPSHOT_KEY

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def sha256_array(waveform: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(waveform, dtype=np.float32).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_versions(text: str) -> dict[str, str]:
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", text, re.M)}


# --------------------------------------------------------------------------------------------------
# data: per-record variation, PCM WAV decoding, the BYOD archive reader, duplicate refusal, baselines
# --------------------------------------------------------------------------------------------------


def vary_waveform(waveform: Any, rng: Any) -> Any:
    """Seeded per-record variation of a generated clip, so no two records are the same waveform (AST-M2).

    Speed perturbation (every frequency scaled by 0.92–1.08), a circular time shift, a gain of 0.55–0.95 of full scale and
    a white-noise floor 30–45 dB below the signal. These are standard audio augmentations; the class character of the
    clip (broadband noise, chirps, low drone) is unchanged.
    """
    import numpy as np

    x = np.asarray(waveform, dtype=np.float64)
    n = x.size
    factor = rng.uniform(0.92, 1.08)
    offset = rng.uniform(0, n)
    positions = (np.arange(n) * factor + offset) % n
    stretched = np.interp(positions, np.arange(n + 1), np.append(x, x[0]))
    peak = float(np.max(np.abs(stretched))) or 1.0
    signal = stretched / peak * rng.uniform(0.55, 0.95)
    rms = float(np.sqrt(np.mean(signal**2))) or 1e-3
    noise = rng.standard_normal(n) * rms * 10 ** (-rng.uniform(30.0, 45.0) / 20.0)
    return np.clip(signal + noise, -1.0, 1.0).astype(np.float32)


def default_dataset(seed: int = SEED, n_records: int = N_RECORDS) -> list[dict[str, Any]]:
    """The package's balanced generated dataset, each clip given its own seeded variation."""
    import numpy as np

    from ast_audio_classification_pipeline import synthetic_audio_dataset

    records = synthetic_audio_dataset(n_samples=n_records, seed=seed)
    for index, record in enumerate(records):
        record["waveform"] = vary_waveform(record["waveform"], np.random.default_rng([seed, index]))
    return records


def unseen_clip(seed: int = SEED) -> Any:
    """A newly generated biophony clip that is not in the dataset, varied the same way."""
    import numpy as np

    from ast_audio_classification_pipeline import SAMPLE_RATE, generate_audio_clip

    clip = generate_audio_clip(index=UNSEEN_INDEX, class_idx=UNSEEN_CLASS, duration=3.0, sample_rate=SAMPLE_RATE, seed=seed)
    return vary_waveform(clip, np.random.default_rng([seed, UNSEEN_INDEX]))


def decode_pcm_wav(payload: bytes, name: str = "WAV") -> tuple[Any, int]:
    """Decode an uncompressed PCM WAV to mono float32 in [-1, 1] and its sample rate."""
    import numpy as np

    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
            compression = handle.getcomptype()
            raw = handle.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{name}: not a readable PCM WAV ({exc}); export it as 16-bit PCM WAV") from exc
    if compression != "NONE":
        raise ValueError(f"{name}: compressed WAV ({compression}) is unsupported; export it as 16-bit PCM WAV")
    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{name}: {8 * sample_width}-bit PCM WAV is unsupported; convert it to 16-bit PCM")
    if channels < 1 or samples.size % channels:
        raise ValueError(f"{name}: invalid WAV channel layout")
    mono = samples.reshape(-1, channels).mean(axis=1) if channels > 1 else samples
    return mono.astype(np.float32), rate


def _archive_layout(names: list[str], classes: tuple[str, ...]) -> tuple[str | None, list[str]]:
    """The single parent folder to strip (``mydata/geophony/a.wav``), or None; and the skipped metadata entries."""
    skipped = [n for n in names if n.startswith("__MACOSX/") or any(part.startswith(".") for part in PurePosixPath(n).parts)]
    kept = [n for n in names if n not in skipped]
    firsts = {PurePosixPath(n).parts[0] for n in kept if PurePosixPath(n).parts}
    if len(firsts) == 1 and next(iter(firsts)) not in classes:
        parent = next(iter(firsts))
        if all(len(PurePosixPath(n).parts) >= 2 and PurePosixPath(n).parts[1] in classes for n in kept if n.rstrip("/") != parent):
            return parent, skipped
    return None, skipped


def load_zip_dataset(path: Path, classes: tuple[str, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a labelled BYOD ZIP in memory (never extracted): ``<class>/<clip>.wav`` for every class.

    Accepts one enclosing folder (what "compress folder" produces) and skips ``__MACOSX/`` and dot-file metadata,
    reporting both. Every refusal names the member and the corrective action (AST-m3).
    """
    from ast_audio_classification_pipeline import (
        MAX_DATASET_AUDIO_SECONDS,
        MIN_DATASET_AUDIO_SECONDS,
        SAMPLE_RATE,
    )

    expected = "zip the three class folders " + ", ".join(f"{c}/" for c in classes) + " so they sit at the top level of the archive (or inside one folder)"
    if path.suffix.lower() != ".zip":
        raise ValueError(f"{path.name}: the dataset must be one .zip file")
    if path.stat().st_size > ZIP_MAX_BYTES:
        raise ValueError(f"{path.name}: the ZIP is larger than 64 MiB compressed; use fewer or shorter clips")
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        parent, skipped = _archive_layout([i.filename for i in infos], classes)
        members = [i for i in infos if i.filename not in skipped and not i.is_dir()]
        if not members or len(members) > ZIP_MAX_FILES:
            raise ValueError(f"{path.name}: the ZIP must contain 1–{ZIP_MAX_FILES} WAV files, found {len(members)}")
        if sum(i.file_size for i in members) > ZIP_MAX_EXPANDED:
            raise ValueError(f"{path.name}: the ZIP expands beyond 256 MiB; use fewer or shorter clips")
        for info in members:
            name = info.filename
            parts = PurePosixPath(name).parts
            if PurePosixPath(name).is_absolute() or ".." in parts:
                raise ValueError(f"{name}: unsafe member path (absolute or '..'); {expected}")
            if parent is not None:
                parts = parts[1:]
            if len(parts) != 2:
                raise ValueError(f"{name}: expected <class>/<clip>.wav; {expected}")
            class_name = parts[0]
            if class_name not in classes:
                raise ValueError(f"{name}: unknown class folder {class_name!r}; the folders must be exactly {list(classes)}")
            if info.flag_bits & 0x1 or not name.lower().endswith(".wav"):
                raise ValueError(f"{name}: every member must be an unencrypted PCM .wav file")
            waveform, rate = decode_pcm_wav(archive.read(info), name)
            if rate != SAMPLE_RATE:
                raise ValueError(f"{name}: sample rate {rate} Hz; resample the clip to {SAMPLE_RATE} Hz (16 kHz) PCM WAV")
            seconds = waveform.size / rate
            if seconds > MAX_DATASET_AUDIO_SECONDS:
                raise ValueError(f"{name}: duration {seconds:.2f} s is longer than {MAX_DATASET_AUDIO_SECONDS} s; trim the clip to at most {MAX_DATASET_AUDIO_SECONDS} s")
            if seconds < MIN_DATASET_AUDIO_SECONDS:
                raise ValueError(f"{name}: duration {seconds:.3f} s is shorter than {MIN_DATASET_AUDIO_SECONDS} s; use a longer clip")
            records.append({"id": name, "waveform": waveform, "sample_rate": rate, "label": classes.index(class_name), "class_name": class_name})
    present = {r["class_name"] for r in records}
    missing = [c for c in classes if c not in present]
    if missing:
        raise ValueError(f"{path.name}: no clips for {missing}; {expected}")
    return records, {"archive": path.name, "archive_sha256": sha256_file(path), "stripped_parent_folder": parent, "skipped_metadata_entries": skipped}


def refuse_duplicates(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Refuse any waveform that occurs more than once: a copy could land in training and evaluation (AST-M2)."""
    by_digest: dict[str, list[str]] = {}
    for record in records:
        by_digest.setdefault(sha256_array(record["waveform"]), []).append(record["id"])
    duplicates = {d: ids for d, ids in by_digest.items() if len(ids) > 1}
    if duplicates:
        groups = "; ".join(" = ".join(ids) for ids in duplicates.values())
        raise ValueError(f"identical waveforms found ({groups}); remove the copies so no clip can be both trained on and evaluated")
    return by_digest


def training_majority_baseline(train_labels: list[int], eval_labels: list[int], class_names: list[str]) -> dict[str, Any]:
    """A trivial predictor fitted on the TRAINING labels only, scored on the evaluation labels (AST-m2)."""
    from ast_audio_classification_pipeline import majority_class_baseline

    fitted = majority_class_baseline(train_labels, len(class_names))
    predicted = fitted["majority_class_index"]
    correct = sum(1 for label in eval_labels if label == predicted)
    return {
        "rule": "always predict the most frequent class in the training split",
        "fitted_on": "train",
        "predicted_class": class_names[predicted],
        "predicted_class_index": predicted,
        "accuracy": round(correct / len(eval_labels), 4),
        "n_evaluation_clips": len(eval_labels),
    }


def add_noise_at_snr(waveform: Any, snr_db: float, rng: Any) -> Any:
    import numpy as np

    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))) or 1e-6
    noisy = waveform + rng.standard_normal(waveform.size) * rms * 10 ** (-snr_db / 20.0)
    return np.clip(noisy, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------------------------------
# model factories (the CPU pre-flight test replaces these with a small randomly initialised stand-in)
# --------------------------------------------------------------------------------------------------


def load_pipeline(run: Run) -> Any:
    from ast_audio_classification_pipeline import ASTAudioClassificationPipeline

    return ASTAudioClassificationPipeline.from_pretrained(weights_dir=run.snapshot)


def load_from_artifact(run: Run, artifact: Path) -> Any:
    from ast_audio_classification_pipeline import ASTAudioClassificationPipeline

    return ASTAudioClassificationPipeline.from_artifact(artifact, weights_dir=run.snapshot)


def load_data(run: Run, stage: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """The train and evaluation records the `dataset` stage wrote, checked against their recorded digests."""
    import numpy as np

    split = run.read_state("split.json", stage)
    arrays = np.load(run.state / "dataset.npz")
    by_id = {}
    for i, rid in enumerate(split["ids"]):
        waveform = arrays[f"w{i}"].astype(np.float32)
        if sha256_array(waveform) != split["digests"][i]:
            raise RuntimeError(f"the dataset changed after the dataset stage ({rid}); run the notebook from Section 6")
        by_id[rid] = {"id": rid, "waveform": waveform, "sample_rate": split["sample_rate"], "label": split["labels"][i], "class_name": split["class_names"][split["labels"][i]]}
    train = [by_id[rid] for rid in split["train_ids"]]
    evaluation = [by_id[rid] for rid in split["eval_ids"]]
    return train, evaluation, split


def score_rows(pipe: Any, records: list[dict[str, Any]], clip: Any) -> dict[str, Any]:
    """Full score vectors for every evaluation clip and the unseen clip (the reload-parity reference)."""
    from ast_audio_classification_pipeline import SAMPLE_RATE

    k = len(pipe.labels)
    rows = {r["id"]: pipe.predict(r["waveform"], sample_rate=r["sample_rate"], top_k=k)["predictions"] for r in records}
    rows["unseen"] = pipe.predict(clip, sample_rate=SAMPLE_RATE, top_k=k)["predictions"]
    return {rid: {p["label"]: p["score"] for p in preds} for rid, preds in rows.items()}


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    """Section 3: install the carried manifest, fetch the absent files at the pinned revision, verify every file."""
    from ast_audio_classification_pipeline import (
        MODEL_ID,
        MODEL_LICENSE,
        MODEL_REVISION,
        stage_missing_files,
        verify_snapshot,
    )
    from ast_audio_classification_pipeline.pipeline import MANIFEST_NAME

    carried = run.root / "weights" / SNAPSHOT_KEY / MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity in the carried pipeline module; regenerate the notebook")
    run.snapshot.mkdir(parents=True, exist_ok=True)
    (run.snapshot / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({"model_id": MODEL_ID, "revision": MODEL_REVISION, "license": MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]}, flush=True)
    fetched = stage_missing_files(run.snapshot, allow_download=True)
    verified = verify_snapshot(run.snapshot)
    files = verified.get("files", []) if isinstance(verified, dict) else []
    record = {"weights_dir": str(run.snapshot), "fetched": fetched, "verified_files": len(files) if isinstance(files, list) else files, "revision": MODEL_REVISION}
    run.write_output("weights.json", record)
    print(record)


def stage_runtime(run: Run) -> None:
    """Section 4: the isolated environment's versions, checked against the carried lock; device and input ceilings."""
    import torch
    import torchaudio
    import transformers

    from ast_audio_classification_pipeline import (
        MAX_AUDIO_SECONDS,
        MAX_INPUT_SECONDS,
        MIN_AUDIO_SECONDS,
        NUM_LABELS,
        SAMPLE_RATE,
    )

    locked = lock_versions((run.root / LOCK).read_text(encoding="utf-8"))
    installed = {"torch": torch.__version__, "torchaudio": torchaudio.__version__, "transformers": transformers.__version__}
    mismatched = {name: {"locked": locked.get(name), "installed": version} for name, version in installed.items() if version.split("+", 1)[0] != locked.get(name)}
    if mismatched:
        raise RuntimeError(f"installed versions differ from the carried lock: {mismatched}; rebuild the isolated environment from Section 2")
    record = {
        "python": platform.python_version(),
        **installed,
        "versions_match_lock": True,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "ceilings": {"SAMPLE_RATE": SAMPLE_RATE, "MIN_AUDIO_SECONDS": MIN_AUDIO_SECONDS, "MAX_AUDIO_SECONDS": MAX_AUDIO_SECONDS, "MAX_INPUT_SECONDS": MAX_INPUT_SECONDS, "NUM_LABELS": NUM_LABELS},
    }
    run.write_output("runtime.json", record)
    print(record)


def stage_pretrained(run: Run) -> None:
    """Section 5: the unchanged 527-label AudioSet head on the tutorial tone (or one BYOD WAV)."""
    from ast_audio_classification_pipeline import SAMPLE_RATE, tutorial_tone, validate_inputs
    from ast_audio_classification_pipeline.samples import TONE_AMPLITUDE, TONE_HZ

    if run.options.wav:
        path = Path(run.options.wav)
        audio, sample_rate = decode_pcm_wav(path.read_bytes(), path.name)
        clip_name, sample_kind = path.name, "BYOD"
    else:
        sample_rate = SAMPLE_RATE
        audio = tutorial_tone(duration=3.0, sample_rate=sample_rate, frequency=TONE_HZ, amplitude=TONE_AMPLITUDE)
        clip_name, sample_kind = f"synthetic_sine_{int(TONE_HZ)}hz_3s.wav", "synthetic"
    manifest = validate_inputs(audio, sample_rate, top_k=TOP_K_BASE, names=[clip_name])
    run.write_state("input_manifest.json", manifest)
    pipe = load_pipeline(run)
    result = pipe.predict(audio, sample_rate=sample_rate, top_k=TOP_K_BASE)
    record = {"sample_kind": sample_kind, "name": clip_name, "samples": int(audio.shape[0]), "sample_rate": sample_rate, "sha256": sha256_array(audio), "device": pipe.device, "result": result}
    run.write_output(f"{STEM}_pretrained.json", record)
    print({"sample_kind": sample_kind, "name": clip_name, "samples": int(audio.shape[0]), "sample_rate": sample_rate, "sha256": record["sha256"][:16] + "...", "activation": result["activation"], "truncated": result["truncated"]})
    for rank, item in enumerate(result["predictions"], start=1):
        print(f"  {rank:>2}. index {item['index']:>3}  score {item['score']:.4f}  {item['label']}")


def stage_dataset(run: Run) -> None:
    """Sections 6–7: build or read the dataset, validate it, refuse duplicates, split it, record the over-long probe."""
    import numpy as np

    from ast_audio_classification_pipeline import (
        ADAPT_CLASSES,
        MAX_INPUT_SECONDS,
        SAMPLE_RATE,
        split_dataset,
        validate_dataset,
        validate_inputs,
    )

    source: dict[str, Any] = {}
    if run.options.zip:
        records, source = load_zip_dataset(Path(run.options.zip), ADAPT_CLASSES)
        kind = "BYOD"
    else:
        records, kind = default_dataset(), "synthetic"
        source = {"generator": "synthetic_audio_dataset(n_samples=24, seed=42) + per-record variation (vary_waveform, seed [42, index])"}
    summary = validate_dataset(records, ADAPT_CLASSES)
    thin = [name for name, count in summary["class_counts"].items() if count < 2]
    if thin:
        raise ValueError(f"every class needs at least two clips for a disjoint split; add clips for {thin}")
    digests = refuse_duplicates(records)
    train, evaluation = split_dataset(records, val_fraction=VAL_FRACTION, seed=SEED)
    train_digests = {sha256_array(r["waveform"]) for r in train}
    overlap = sorted(r["id"] for r in evaluation if sha256_array(r["waveform"]) in train_digests)
    if overlap:
        raise RuntimeError(f"evaluation clips identical to training clips: {overlap}")
    np.savez(run.state / "dataset.npz", **{f"w{i}": r["waveform"] for i, r in enumerate(records)})
    run.write_state("split.json", {
        "dataset_kind": kind, "sample_rate": SAMPLE_RATE, "class_names": list(ADAPT_CLASSES),
        "ids": [r["id"] for r in records], "labels": [r["label"] for r in records], "digests": [sha256_array(r["waveform"]) for r in records],
        "train_ids": [r["id"] for r in train], "eval_ids": [r["id"] for r in evaluation],
    })
    manifest = run.read_state("input_manifest.json", "dataset")
    manifest.update({"dataset_kind": kind, "dataset_source": source, "dataset_representation": summary["representation"], "dataset_records": summary["n_records"], "class_counts": summary["class_counts"], "distinct_waveforms": len(digests)})
    try:
        validate_inputs(np.zeros(int((MAX_INPUT_SECONDS + 1) * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)
    except ValueError as exc:
        manifest["findings"].append({"input": "over-long-probe", "verdict": "rejected", "message": str(exc)})
    run.write_output(f"{STEM}_input_manifest.json", manifest)
    counts = {
        "train": {c: sum(r["class_name"] == c for r in train) for c in ADAPT_CLASSES},
        "evaluation": {c: sum(r["class_name"] == c for r in evaluation) for c in ADAPT_CLASSES},
    }
    split_record = {
        "dataset_kind": kind, "records": len(records), "distinct_waveforms": len(digests), "class_counts": summary["class_counts"],
        "split": "stratified, seeded (seed 42), 75% train / 25% held-out evaluation", "train_clips": len(train), "evaluation_clips": len(evaluation),
        "counts": counts, "train_evaluation_duplicates": len(overlap), "eval_ids": [r["id"] for r in evaluation],
    }
    run.write_output(f"{STEM}_split.json", split_record)
    print({"dataset_kind": kind, "records": len(records), "distinct_waveforms": len(digests), "class_counts": summary["class_counts"], "verdict": summary["verdict"], **({"archive": source} if kind == "BYOD" else {})})
    print({"refusal_probe": manifest["findings"][-1] if manifest["findings"] else None})


def stage_baseline(run: Run) -> None:
    """Section 8: re-head, freeze, and score the untrained head and the training-majority rule on the evaluation clips."""
    train, evaluation, split = load_data(run, "baseline")
    pipe = load_pipeline(run)
    pipe.rehead(split["class_names"], seed=SEED)
    frozen = pipe.freeze_backbone()
    trainable = sum(p.numel() for p in pipe.model.parameters() if p.requires_grad)
    pre = pipe.evaluate(evaluation)
    pre.pop("baseline", None)
    pre.pop("accuracy_delta_vs_baseline", None)
    baseline = training_majority_baseline([r["label"] for r in train], [r["label"] for r in evaluation], split["class_names"])
    record = {"stage": "reheaded_pre_adaptation", "classes": pipe.labels, "frozen_backbone_parameters": frozen, "trainable_parameters": trainable, "untrained_head": pre, "majority_baseline": baseline}
    run.write_output(f"{STEM}_pre_adaptation.json", record)
    print({"classes": pipe.labels, "frozen_backbone_parameters": frozen, "trainable_parameters": trainable})
    print({"untrained_head_accuracy": pre["accuracy"], "untrained_head_macro_f1": pre["macro_f1"], "majority_baseline": f"always '{baseline['predicted_class']}' (fitted on train) -> accuracy {baseline['accuracy']}"})


def stage_finetune(run: Run) -> None:
    """Section 9: bounded head fine-tune on cached frozen features; record the in-memory scores; export the adapter."""
    train, evaluation, split = load_data(run, "finetune")
    opts = run.options
    pipe = load_pipeline(run)
    pipe.rehead(split["class_names"], seed=SEED)
    pipe.freeze_backbone()
    history = pipe.finetune(train_records=train, val_records=evaluation, epochs=opts.epochs, batch_size=opts.batch_size, learning_rate=opts.learning_rate, seed=SEED)
    for row in history:
        row["evaluation_accuracy"] = row.pop("val_accuracy", None)
        row["evaluation_macro_f1"] = row.pop("val_macro_f1", None)
    run.write_state("in_memory_scores.json", score_rows(pipe, evaluation, unseen_clip()))
    artifact = pipe.save_artifact(run.out / ADAPTER)
    run.write_output(f"{STEM}_training_history.json", {"epochs": opts.epochs, "batch_size": opts.batch_size, "learning_rate": opts.learning_rate, "seed": SEED, "history": history})
    run.write_state("adapter.json", {"path": str(artifact), "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)})
    for row in history:
        print(f"Epoch {row['epoch']}/{opts.epochs}: train_loss={row['train_loss']:.4f}  evaluation_accuracy={row['evaluation_accuracy']:.4f}  steps={row['steps']}")
    print({"adapter": artifact.name, "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)[:16] + "...", "contains": "classifier head only (the frozen backbone is rebuilt from the pinned base revision)"})


def stage_evaluate(run: Run) -> None:
    """Sections 10–11: a fresh process loads the exported adapter, scores the evaluation clips and the unseen clip."""
    from ast_audio_classification_pipeline import ARTIFACT_FORMAT, MODEL_ID, MODEL_REVISION, SAMPLE_RATE

    train, evaluation, split = load_data(run, "evaluate")
    adapter = run.read_state("adapter.json", "evaluate")
    pipe = load_from_artifact(run, Path(adapter["path"]))
    report = pipe.evaluate(evaluation)
    report.pop("baseline", None)
    report.pop("accuracy_delta_vs_baseline", None)
    baseline = training_majority_baseline([r["label"] for r in train], [r["label"] for r in evaluation], split["class_names"])
    payload = {
        "task": "multiclass acoustic ecology classification",
        "score_semantics": SCORE_SEMANTICS,
        "decision_rule": DECISION_RULE,
        "calibration": "none: the scores are not calibrated probabilities; calibration and any threshold are the responsibility of a downstream user, on their own data",
        "sample_kind": f"{split['dataset_kind']}_heldout_evaluation",
        "n_clips": len(evaluation),
        "classes": split["class_names"],
        "metrics": report,
        "baseline": baseline,
        "accuracy_delta_vs_baseline": round(report["accuracy"] - baseline["accuracy"], 4),
        "split_assumption": "random stratified split; assumes clips are independent (no shared recording or source across clips)",
        "verdict": "sample-sanity",
        "reason": f"{len(evaluation)} held-out clip(s) evaluated against a training-majority baseline; not a benchmark claim",
        "adapter_format": ARTIFACT_FORMAT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    run.write_output(f"{STEM}_evaluation_report.json", payload)
    clip = unseen_clip()
    result = pipe.predict(clip, sample_rate=SAMPLE_RATE, top_k=len(pipe.labels))
    unseen = {"clip": "unseen_synthetic_biophony_99", "true_class": split["class_names"][UNSEEN_CLASS], "sha256": sha256_array(clip), "in_dataset": sha256_array(clip) in set(split["digests"]), "score_semantics": SCORE_SEMANTICS, "decision_rule": DECISION_RULE, "result": result}
    run.write_output(f"{STEM}_unseen_prediction.json", unseen)
    with (run.out / f"{STEM}_top_k.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip", "rank", "index", "label", "score"])
        for rank, item in enumerate(result["predictions"], start=1):
            writer.writerow(["unseen_test_clip_99", rank, item["index"], item["label"], f"{item['score']:.6f}"])
    print({"evaluation_clips": len(evaluation), "accuracy": report["accuracy"], "macro_f1": report["macro_f1"], "majority_baseline_accuracy": baseline["accuracy"], "delta_vs_baseline": payload["accuracy_delta_vs_baseline"], "verdict": "sample-sanity"})
    print("Per class (precision / recall / F1 / support):")
    for name, values in report["per_class"].items():
        print(f"  {name:<12} {values['precision']:.4f} / {values['recall']:.4f} / {values['f1']:.4f} / {values['support']}")
    print("Confusion matrix (rows = true class, columns = predicted class):", split["class_names"])
    for name, row in zip(split["class_names"], report["confusion_matrix"], strict=True):
        print(f"  {name:<12} {row}")


def stage_reload(run: Run) -> None:
    """Section 12: another fresh process rebuilds the pipeline from the adapter and compares every score with training."""
    import numpy as np

    _train, evaluation, _split = load_data(run, "reload")
    reference = run.read_state("in_memory_scores.json", "reload")
    adapter = run.read_state("adapter.json", "reload")
    if sha256_file(Path(adapter["path"])) != adapter["sha256"]:
        raise RuntimeError("the adapter file changed after it was exported; rerun from Section 9")
    reloaded = load_from_artifact(run, Path(adapter["path"]))
    scores = score_rows(reloaded, evaluation, unseen_clip())
    labels_equal = all(max(scores[k], key=scores[k].get) == max(reference[k], key=reference[k].get) for k in reference)
    deltas = [abs(scores[k][c] - reference[k][c]) for k in reference for c in reference[k]]
    close = all(np.isclose(scores[k][c], reference[k][c], rtol=PARITY_RTOL, atol=PARITY_ATOL) for k in reference for c in reference[k])
    record = {"clips_compared": len(reference), "scores_compared": len(deltas), "max_abs_score_difference": float(max(deltas)), "predicted_labels_identical": labels_equal, "rtol": PARITY_RTOL, "atol": PARITY_ATOL, "labels": reloaded.labels, "reload_verification": "PASSED" if (close and labels_equal) else "FAILED"}
    run.write_output(f"{STEM}_reload_parity.json", record)
    print(record)
    if not (close and labels_equal):
        raise RuntimeError("the reloaded adapter does not reproduce the trained model's scores; do not use the export")


def stage_bundle(run: Run) -> None:
    """Section 13: one provenance bundle that links every record of this run."""
    from ast_audio_classification_pipeline import MODEL_ID, MODEL_LICENSE, MODEL_REVISION

    def load(name: str) -> Any:
        path = run.out / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the notebook from the top before the bundle")
        return json.loads(path.read_text(encoding="utf-8"))

    source = json.loads((run.root / "source.json").read_text(encoding="utf-8"))
    payload = {
        "evaluation_report": load(f"{STEM}_evaluation_report.json"),
        "pre_adaptation": load(f"{STEM}_pre_adaptation.json"),
        "input_manifest": load(f"{STEM}_input_manifest.json"),
        "split": load(f"{STEM}_split.json"),
        "test_prediction": load(f"{STEM}_unseen_prediction.json"),
        "training_history": load(f"{STEM}_training_history.json"),
        "reload_parity": load(f"{STEM}_reload_parity.json"),
        "notebook_source": source,
        "repository_revision": source["revision"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "runtime": load("runtime.json"),
        "environment": "isolated hash-locked uv environment; nothing installed into the notebook kernel",
    }
    payload["files"] = {p.name: sha256_file(p) for p in sorted(run.out.iterdir()) if p.is_file() and p.name != f"{STEM}_result.json"}
    run.write_output(f"{STEM}_result.json", payload)
    print({"outputs_directory": str(run.out), "files": sorted(p.name for p in run.out.iterdir() if p.is_file())})


def stage_activity(run: Run) -> None:
    """Section 14 (optional): add seeded noise at one SNR to the evaluation clips and score them with the same adapter."""
    import numpy as np

    _train, evaluation, split = load_data(run, "activity")
    adapter = run.read_state("adapter.json", "activity")
    canonical = {p.name: sha256_file(p) for p in sorted(run.out.iterdir()) if p.is_file()}
    pipe = load_from_artifact(run, Path(adapter["path"]))
    snr = float(run.options.snr_db)
    clean = pipe.evaluate(evaluation)
    noisy_records = [{**r, "waveform": add_noise_at_snr(r["waveform"], snr, np.random.default_rng([SEED, 7, i]))} for i, r in enumerate(evaluation)]
    noisy = pipe.evaluate(noisy_records)
    changed = sum(1 for a, b in zip(clean["confusion_matrix"], noisy["confusion_matrix"], strict=True) if a != b)
    record = {"changed_variable": "evaluation-clip noise level", "snr_db": snr, "clean": {"accuracy": clean["accuracy"], "macro_f1": clean["macro_f1"], "confusion_matrix": clean["confusion_matrix"]}, "noisy": {"accuracy": noisy["accuracy"], "macro_f1": noisy["macro_f1"], "confusion_matrix": noisy["confusion_matrix"]}, "confusion_rows_changed": changed, "classes": split["class_names"]}
    out = run.out / "activity"
    out.mkdir(exist_ok=True)
    tag = f"{snr:g}".replace("-", "minus").replace(".", "p")
    (out / f"{STEM}_activity_snr_{tag}db.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    after = {p.name: sha256_file(p) for p in sorted(run.out.iterdir()) if p.is_file()}
    if after != canonical:
        raise RuntimeError("the activity changed a canonical output; it must write only under outputs/activity/")
    print(record)


STAGES = {
    "weights": stage_weights,
    "runtime": stage_runtime,
    "pretrained": stage_pretrained,
    "dataset": stage_dataset,
    "baseline": stage_baseline,
    "finetune": stage_finetune,
    "evaluate": stage_evaluate,
    "reload": stage_reload,
    "bundle": stage_bundle,
    "activity": stage_activity,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="run directory holding the carried sources")
    parser.add_argument("--weights", type=Path, required=True, help="directory holding the pinned snapshot")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--wav", default="", help="pretrained: one BYOD PCM WAV instead of the tutorial tone")
    parser.add_argument("--zip", default="", help="dataset: a labelled BYOD ZIP instead of the generated dataset")
    parser.add_argument("--epochs", type=int, default=5, help="finetune: training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="finetune: clips per optimiser step")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="finetune: AdamW learning rate")
    parser.add_argument("--snr-db", type=float, default=0.0, help="activity: signal-to-noise ratio of the added noise, in dB")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    root = options.root.resolve()
    carried_src = root / "src"
    if carried_src.is_dir() and str(carried_src) not in sys.path:
        sys.path.insert(0, str(carried_src))
    run = Run(root, options.weights.resolve(), options)
    error_file = run.state / f"{options.stage}.error.json"
    error_file.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        STAGES[options.stage](run)
    except Exception as exc:  # the notebook re-raises this message in the kernel
        traceback.print_exc()
        message = str(exc) or repr(exc)
        error_file.write_text(json.dumps({"stage": options.stage, "type": type(exc).__name__, "message": message}), encoding="utf-8")
        print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)
        return 2
    print({"stage": options.stage, "status": "ok", "seconds": round(time.perf_counter() - started, 1)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
