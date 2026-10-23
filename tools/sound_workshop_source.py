"""Standalone sound-event workshop worker; heavy libraries load only in model stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path

import numpy as np

SEED = 42
ROLES = ("train", "validation", "test")
STAGES = ("prepare", "train", "reload", "activity", "report")
# Exit code for a refused learner input (a BYOD manifest or recording, or the unlabelled recording). The
# worker prints the reason once and the notebook cell returns without raising; other failures still raise.
REFUSED_EXIT = 65
RECIPE = {
    "seed": 42,
    "epochs": 10,
    "batch_size": 4,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "precision": "float32",
    "shuffle": "random.Random(42 + epoch * 13)",
}


class InputRefused(ValueError):
    """A learner's input (BYOD manifest or recording, unlabelled recording) breaks a documented rule."""


def is_input_refusal(error):
    return isinstance(error, InputRefused) or isinstance(error.__cause__, InputRefused)


def refuse(reason, action):
    print(f"Refused: {reason}. {action} and run this cell again.", flush=True)
    raise SystemExit(REFUSED_EXIT)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(a):
    return hashlib.sha256(np.asarray(a, dtype="<f4").tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def contained(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise InputRefused("audio must be a contained relative path")
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise InputRefused("audio path traversal")
    path = (root / p).resolve()
    if not path.is_relative_to(root.resolve()):
        raise InputRefused("audio path escapes dataset")
    return path


def audio_header(path):
    """Bound RIFF chunks before scipy allocates decoded samples."""
    path = Path(path)
    if not 44 <= path.stat().st_size <= 8 * 1024**2:
        raise InputRefused("WAV must be 44 bytes to 8 MiB")
    with path.open("rb") as f:
        header = f.read(12)
        if header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise InputRefused("expected RIFF WAV")
        if struct.unpack("<I", header[4:8])[0] + 8 != path.stat().st_size:
            raise InputRefused("truncated or trailing WAV bytes")
        fmt = None
        size = None
        while f.tell() < path.stat().st_size:
            chunk = f.read(8)
            if len(chunk) != 8:
                raise InputRefused("truncated chunk")
            kind, length = struct.unpack("<4sI", chunk)
            if f.tell() + length > path.stat().st_size:
                raise InputRefused("truncated WAV data")
            if kind == b"fmt ":
                if length < 16 or length > 128 or fmt is not None:
                    raise InputRefused("invalid WAV format chunk")
                fmt = struct.unpack("<HHIIHH", f.read(16))
                f.seek(length - 16, 1)
            elif kind == b"data":
                if size is not None:
                    raise InputRefused("multiple data chunks")
                size = length
                f.seek(length, 1)
            else:
                f.seek(length, 1)
            if length % 2:
                f.seek(1, 1)
    if fmt is None or size is None:
        raise InputRefused("missing WAV format/data")
    encoding, channels, rate, byte_rate, align, bits = fmt
    if (encoding, bits) not in ((1, 16), (3, 32)) or channels not in (1, 2):
        raise InputRefused("only mono/stereo PCM16 or IEEE float32 WAV accepted")
    if not 8000 <= rate <= 48000 or align != channels * bits // 8 or byte_rate != rate * align:
        raise InputRefused("invalid rate/block alignment")
    if size % align or not 0.5 <= size / align / rate <= 10:
        raise InputRefused("WAV duration must be 0.5-10 seconds")
    return {
        "sample_rate": rate,
        "channels": channels,
        "frames": size // align,
        "seconds": size / align / rate,
        "encoding": encoding,
        "bits": bits,
    }


def read_audio(path):
    info = audio_header(path)
    from scipy.io import wavfile

    rate, raw = wavfile.read(path)
    if raw.shape[0] != info["frames"] or rate != info["sample_rate"]:
        raise InputRefused("decoded WAV disagrees with header")
    x = raw.astype(np.float32) / (32768.0 if raw.dtype == np.int16 else 1.0)
    if not np.isfinite(x).all() or np.max(np.abs(x)) > 1:
        raise InputRefused("nonfinite or out-of-range WAV samples")
    info["decoded_sha256"] = array_hash(x)
    if x.ndim == 2:
        x = x.mean(axis=1, dtype=np.float32)
    info["mono_sha256"] = array_hash(x)
    if np.sqrt(np.mean(x.astype(np.float64) ** 2)) <= 1e-8:
        raise InputRefused("silent audio is not a labelled sound example")
    info["original_mono_peak"] = float(np.max(np.abs(x)))
    info["fixed_input_gain"] = 0.5
    x = x * np.float32(0.5)
    if rate != 16000:
        import torch
        import torchaudio.functional as af

        x = af.resample(torch.from_numpy(x), rate, 16000).numpy().astype(np.float32)
    if not np.isfinite(x).all() or np.max(np.abs(x)) > 1:
        raise InputRefused("resampling exceeded normalized amplitude; no silent clipping")
    info.update(
        {
            "waveform_sha256": array_hash(x),
            "resampled_frames": len(x),
            "resampler": "torchaudio.functional.resample default sinc_interp_hann",
            "resampled_peak": float(np.max(np.abs(x))),
            "channel_policy": "arithmetic mean" if info["channels"] == 2 else "mono unchanged",
        }
    )
    return x, info


def validate_records(records, class_names, byod=False):
    if not 2 <= len(class_names) <= 10 or len(set(class_names)) != len(class_names):
        raise InputRefused("vocabulary needs 2-10 unique classes")
    if any(not isinstance(n, str) or not n.strip() or n != n.strip() for n in class_names):
        raise InputRefused("invalid class names")
    if not records or len(records) > 400:
        raise InputRefused("dataset needs 1-400 records")
    ids, hashes, mono, groups = set(), set(), set(), {}
    total = 0.0
    for index, r in enumerate(records):
        missing = [key for key in ("id", "role", "label", "group_id") if key not in r]
        if missing:
            raise InputRefused(f"record {index} is missing required field(s) {missing}")
        if not isinstance(r["id"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", r["id"]):
            raise InputRefused(f"record {index}: id {r['id']!r} must be 1-160 characters from A-Z a-z 0-9 _ . -")
        if r["id"] in ids:
            raise InputRefused(f"record id {r['id']!r} is duplicated; ids must be unique")
        ids.add(r["id"])
        if r["role"] not in ROLES:
            raise InputRefused(f"record {r['id']!r}: role {r['role']!r} must be one of {list(ROLES)}")
        if type(r["label"]) is not int or not 0 <= r["label"] < len(class_names):
            raise InputRefused(f"record {r['id']!r}: label {r['label']!r} is not in class_names {class_names}")
        g = r["group_id"]
        if not isinstance(g, str) or not g:
            raise InputRefused(f"record {r['id']!r}: group_id must be a nonempty string")
        if g in groups and groups[g] != r["role"]:
            raise InputRefused(
                f"recording group {g!r} appears in both {groups[g]} and {r['role']} (record {r['id']!r}); "
                "keep every excerpt of one original recording in a single role"
            )
        groups[g] = r["role"]
        for key, seen in (("waveform_sha256", hashes), ("mono_sha256", mono)):
            if r[key] in seen:
                raise InputRefused(f"record {r['id']!r} duplicates another record's decoded/resampled waveform")
            seen.add(r[key])
        total += r["seconds"]
    if total > 4000:
        raise InputRefused("dataset exceeds 4000 seconds")
    for role in ROLES:
        for label in range(len(class_names)):
            subset = [r for r in records if r["role"] == role and r["label"] == label]
            minimum = (4 if role == "train" else 2) if byod else (24 if role == "train" else 8)
            if len(subset) < minimum or (not byod and len(subset) != minimum):
                rule = "at least" if byod else "exactly (frozen default split)"
                raise InputRefused(
                    f"class {class_names[label]!r} has {len(subset)} {role} clip(s); {rule} {minimum} required"
                )
            independent = len({r["group_id"] for r in subset})
            if byod and independent < 2:
                raise InputRefused(
                    f"class {class_names[label]!r} has {independent} recording group(s) in {role}; "
                    "BYOD needs at least 2 independent groups per class per role"
                )
    return {
        "roles": {role: sum(r["role"] == role for r in records) for role in ROLES},
        "groups": {role: len({r["group_id"] for r in records if r["role"] == role}) for role in ROLES},
        "seconds": total,
        "class_names": class_names,
        "private_byod": byod,
    }


def metrics(targets, predictions, class_names):
    n = len(class_names)
    t, p = np.asarray(targets), np.asarray(predictions)
    if len(t) == 0 or t.shape != p.shape or t.ndim != 1:
        raise ValueError("metric inputs must be nonempty aligned vectors")
    if (
        t.dtype.kind not in "iu"
        or p.dtype.kind not in "iu"
        or min(t.min(), p.min()) < 0
        or max(t.max(), p.max()) >= n
    ):
        raise ValueError("metric indices outside vocabulary")
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (t, p), 1)
    if np.any(cm.sum(1) == 0):
        raise ValueError("every declared class requires evaluation support")
    rows = {}
    for i, name in enumerate(class_names):
        tp, support, predicted = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        rows[name] = {
            "precision": tp / predicted if predicted else None,
            "precision_reason": None if predicted else "no predicted positives",
            "recall": tp / support,
            "f1": 2 * tp / (support + predicted),
            "support": support,
        }
    return {
        "accuracy": float(np.trace(cm) / len(t)),
        "macro_f1": sum(r["f1"] for r in rows.values()) / n,
        "per_class": rows,
        "confusion_matrix": cm.tolist(),
        "row_normalized_confusion": (cm / cm.sum(1, keepdims=True)).tolist(),
        "clips": len(t),
    }


def noise_conditions(waveform, clip_id):
    x = np.asarray(waveform, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if not np.isfinite(x).all() or rms <= 1e-8:
        raise ValueError("SNR undefined for silent/nonfinite signal")
    seed = SEED + int(hashlib.sha256(clip_id.encode()).hexdigest()[:16], 16)
    z = np.random.default_rng(seed).normal(size=len(x))
    z -= z.mean()
    z /= np.sqrt(np.mean(z**2))
    signals = {"clean": x.astype(np.float64)}
    for snr in (20, 10, 0):
        signals[str(snr)] = x + z * rms * 10 ** (-snr / 20)
    gain = min(1.0, 0.99 / max(float(np.max(np.abs(y))) for y in signals.values()))
    result = []
    for condition, y in signals.items():
        audio = (gain * y).astype(np.float32)
        noise = audio.astype(np.float64) - gain * x.astype(np.float64)
        measured = None if condition == "clean" else 20 * math.log10(gain * rms / np.sqrt(np.mean(noise**2)))
        result.append(
            {
                "condition": condition,
                "audio": audio,
                "gain": gain,
                "actual_snr_db": measured,
                "peak": float(np.max(np.abs(audio))),
                "clipped_samples": int(np.sum(np.abs(audio) >= 1)),
                "noise_sha256": array_hash(z),
                "waveform_sha256": array_hash(audio),
                "seed": seed,
            }
        )
    return result


def activity_ids(records, names):
    chosen = []
    for label in range(len(names)):
        pool = sorted(
            (r for r in records if r["role"] == "validation" and r["label"] == label), key=lambda r: r["id"]
        )
        first = pool[0]
        second = next((r for r in pool[1:] if r["group_id"] != first["group_id"]), pool[1])
        chosen.extend([first["id"], second["id"]])
    return chosen


# Fixed display scales shared by every waveform/spectrogram figure, so conditions of one clip compare
# directly: amplitude is full scale; log filterbank energy runs from the float32 floor (about -15.9)
# to just above the largest value a full-scale test tone reaches (about 10.7).
WAVEFORM_LIMITS = (-1.0, 1.0)
FBANK_LIMITS = (-16.0, 11.0)


def plot_audio(x, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torchaudio.compliance.kaldi as kaldi
    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path.with_suffix(".wav"), 16000, x.astype(np.float32))
    mel = kaldi.fbank(
        torch.from_numpy(x).unsqueeze(0),
        sample_frequency=16000,
        num_mel_bins=128,
        dither=0.0,
        window_type="hanning",
        use_energy=False,
        htk_compat=True,
    ).numpy()
    fig, ax = plt.subplots(2, 1, figsize=(9, 5))
    ax[0].plot(np.arange(len(x)) / 16000, x, linewidth=0.5)
    ax[0].set(xlabel="Seconds", ylabel="Amplitude (full scale)", title=title, ylim=WAVEFORM_LIMITS)
    image = ax[1].imshow(
        mel.T, aspect="auto", origin="lower", extent=(0, len(mel) * 0.01, 0, 128),
        vmin=FBANK_LIMITS[0], vmax=FBANK_LIMITS[1],
    )
    fig.colorbar(image, ax=ax[1], extend="both", label="Log filterbank energy (fixed scale)")
    ax[1].set(
        xlabel="Seconds",
        ylabel="Mel-frequency bin (low to high)",
        title="Log-mel filterbank before AST normalization/padding",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def check_byod_manifest(byod):
    """Structure of a BYOD directory and manifest, checked before prepare() reads it (refused cleanly)."""
    path = byod / "manifest.json"
    if not byod.is_dir():
        raise InputRefused(f"BYOD_PATH {str(byod)!r} is not a directory")
    if not path.is_file():
        raise InputRefused("manifest.json is missing from BYOD_PATH")
    if path.stat().st_size > 1024**2:
        raise InputRefused("BYOD manifest exceeds 1 MiB")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InputRefused(f"manifest.json is not valid UTF-8 JSON ({error})") from error
    if not isinstance(manifest, dict):
        raise InputRefused("manifest.json must be a JSON object with 'class_names' and 'records'")
    missing = [key for key in ("class_names", "records") if key not in manifest]
    if missing:
        raise InputRefused(f"manifest.json is missing required key(s) {missing}")
    names = manifest["class_names"]
    if (
        not isinstance(names, list)
        or not 2 <= len(names) <= 10
        or any(not isinstance(n, str) or not n.strip() or n != n.strip() for n in names)
        or len(set(names)) != len(names)
    ):
        raise InputRefused("class_names must be a list of 2-10 unique, nonempty names without surrounding spaces")
    records = manifest["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= 400:
        raise InputRefused(f"records must be a list of 1-400 record objects, found {type(records).__name__}")
    for index, r in enumerate(records):
        if not isinstance(r, dict):
            raise InputRefused(f"manifest record {index} must be an object, found {type(r).__name__}")
        missing = [key for key in ("id", "audio", "label", "group_id", "role") if key not in r]
        if missing:
            raise InputRefused(f"manifest record {index} is missing required field(s) {missing}")
        if not isinstance(r["id"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", r["id"]):
            raise InputRefused(
                f"manifest record {index}: id {r['id']!r} must be 1-160 characters from A-Z a-z 0-9 _ . -"
            )
        label = r["label"]
        if not ((isinstance(label, str) and label in names) or (type(label) is int and 0 <= label < len(names))):
            raise InputRefused(f"manifest record {r['id']!r}: label {label!r} is not in class_names {names}")
        if not isinstance(r["group_id"], str) or not r["group_id"]:
            raise InputRefused(f"manifest record {r['id']!r}: group_id must be a nonempty string")
        if not isinstance(r["role"], str) or r["role"] not in ROLES:
            raise InputRefused(f"manifest record {r['id']!r}: role {r['role']!r} must be one of {list(ROLES)}")
        if not contained(byod, r["audio"]).is_file():
            raise InputRefused(f"manifest record {r['id']!r}: audio file {r['audio']!r} not found under BYOD_PATH")


def prepare(root, byod=None):
    if byod and (byod / "manifest.json").stat().st_size > 1024**2:
        raise ValueError("BYOD manifest exceeds 1 MiB")
    manifest = read_json((byod / "manifest.json") if byod else root / "sample.json")
    names = manifest["class_names"]
    if (
        not isinstance(names, list)
        or not 2 <= len(names) <= 10
        or any(not isinstance(n, str) or not n.strip() or n != n.strip() for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("invalid class vocabulary")
    source_records = manifest["records"]
    if not 1 <= len(source_records) <= 400:
        raise ValueError("maximum 400 clips")
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    records = []
    for index, entry in enumerate(source_records):
        r = dict(entry)
        if byod:
            missing = [key for key in ("id", "audio", "label", "group_id", "role") if key not in r]
            if missing:
                raise ValueError(f"manifest record {index} is missing required field(s) {missing}")
            if isinstance(r["label"], str) and r["label"] not in names:
                raise ValueError(f"manifest record {r['id']!r}: label {r['label']!r} is not in class_names {names}")
        if not isinstance(r.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", r["id"]):
            raise ValueError(
                f"manifest record {index}: id {r.get('id')!r} must be 1-160 characters from A-Z a-z 0-9 _ . -"
            )
        if byod:
            path = contained(byod, r["audio"])
        else:
            path = cache / (r["id"] + ".wav")
            if not path.exists():
                with urllib.request.urlopen(r["url"], timeout=60) as response:
                    payload = response.read(r["bytes"] + 1)
                if len(payload) != r["bytes"] or hashlib.sha256(payload).hexdigest() != r["sha256"]:
                    raise ValueError("download integrity mismatch")
                path.write_bytes(payload)
            if path.stat().st_size != r["bytes"] or digest(path) != r["sha256"]:
                raise ValueError("cached data integrity mismatch")
        try:
            x, info = read_audio(path)
        except ValueError as error:
            raise ValueError(f"record {r['id']!r} ({path.name}): {error}") from error
        if not byod and (info["channels"] != 1 or info["sample_rate"] != 44100 or info["frames"] != 220500):
            raise ValueError("default ESC10 format changed")
        label = r.get("label", r.get("category"))
        label = names.index(label) if isinstance(label, str) else label
        if type(label) is not int or not 0 <= label < len(names):
            raise ValueError("label outside vocabulary")
        group = r.get("group_id", r.get("src_file", r.get("source_id", "")))
        if not isinstance(group, str) or not group:
            raise ValueError("group_id must be a nonempty string")
        if "decoded_sha256" in r:
            r["source_pcm_decoded_sha256"] = r["decoded_sha256"]
        r.update(info)
        r.update(
            {
                "label": label,
                "category": names[label],
                "group_id": group,
                "audio": f"data/{r['id']}.npy",
                "original_sample_rate": info["sample_rate"],
                "original_frames": info["frames"],
                "sample_rate": 16000,
                "frames": len(x),
                "channels": 1,
                "original_channels": info["channels"],
            }
        )
        np.save(root / r["audio"], x, allow_pickle=False)
        records.append(r)
    summary = validate_records(records, names, byod is not None)
    write_json(root / "outputs/data_manifest.json", manifest)
    write_json(root / "outputs/records.json", records)
    write_json(root / "outputs/prepare.json", summary)
    write_json(
        root / "outputs/configuration.json",
        {"recipe": RECIPE, "class_names": names, "activity_ids": activity_ids(records, names)},
    )
    train_labels = [r["label"] for r in records if r["role"] == "train"]
    majority = int(np.bincount(train_labels, minlength=len(names)).argmax())
    baseline = {
        role: metrics(
            [r["label"] for r in records if r["role"] == role],
            [majority] * sum(r["role"] == role for r in records),
            names,
        )
        for role in ("validation", "test")
    }
    write_json(
        root / "outputs/baseline.json",
        {"majority_label": majority, "tie_rule": "lowest vocabulary index", "metrics": baseline},
    )
    for label, name in enumerate(names):
        r = min((r for r in records if r["role"] == "train" and r["label"] == label), key=lambda r: r["id"])
        plot_audio(np.load(root / r["audio"]), root / f"outputs/examples/{r['id']}.png", name)
    return summary


def cache_identity(root, records, extractor_config):
    identity = {
        "inputs": [(r["id"], r["waveform_sha256"]) for r in records],
        "extractor": extractor_config,
        "dtype": "float32",
        "source": read_json(root / "source.json"),
        "worker_sha256": digest(__file__),
        "weights": read_json(root / "weights/ast/dimer-base-manifest.json"),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def require_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Model stages require a hosted CUDA GPU; local validation is CPU-only")
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    return "cuda:0"


def load_pipe(root):
    from ast_reference.pipeline import ASTAudioClassificationPipeline

    return ASTAudioClassificationPipeline.from_pretrained(
        device=require_gpu(), weights_dir=root / "weights/ast", allow_download=True
    )


def backbone_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.audio_spectrogram_transformer.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def extract_features(pipe, waveforms):
    import torch

    result = []
    pipe.model.audio_spectrogram_transformer.eval()
    for start in range(0, len(waveforms), 4):
        batch = pipe.extractor(waveforms[start : start + 4], sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            feature = pipe.model.audio_spectrogram_transformer(
                batch["input_values"].to(pipe.device)
            ).pooler_output
            array = feature.float().cpu().numpy().copy()
        # Ordinary tensors outside inference_mode can be saved for classifier backward.
        result.append(torch.from_numpy(array).to(pipe.device))
    return torch.cat(result)


def head_logits(pipe, features):
    import torch

    pipe.model.classifier.eval()
    with torch.inference_mode():
        return pipe.model.classifier(features).float().cpu().numpy()


def softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def train_head(classifier, train_features, targets, val_features, val_targets, names):
    """One optimizer, exact macro-F1, epoch-zero snapshot and earliest winning tie."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(SEED)
    initial = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
    best = initial
    best_score = -1.0
    selected = 0
    history = []
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=0.01)
    steps = 0
    maximum_delta = 0.0
    for epoch in range(11):
        loss_total = 0.0
        count = 0
        if epoch:
            classifier.train()
            order = list(range(len(targets)))
            random.Random(SEED + epoch * 13).shuffle(order)
            for start in range(0, len(order), 4):
                indices = torch.tensor(order[start : start + 4], device=train_features.device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(classifier(train_features[indices]), targets[indices])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                optimizer.step()
                steps += 1
                loss_total += float(loss.detach()) * len(indices)
                count += len(indices)
            maximum_delta = max(
                maximum_delta,
                max(
                    float((v.detach().cpu() - initial[k]).abs().max())
                    for k, v in classifier.state_dict().items()
                ),
            )
        classifier.eval()
        with torch.inference_mode():
            pred = classifier(val_features).argmax(-1).cpu().tolist()
        report = metrics(val_targets.cpu().tolist(), pred, names)
        if report["macro_f1"] > best_score:
            best_score = report["macro_f1"]
            selected = epoch
            best = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / count if count else None,
                "validation_macro_f1": report["macro_f1"],
                "validation_accuracy": report["accuracy"],
                "optimizer_steps": steps,
            }
        )
        print(f"Epoch {epoch}: validation macro-F1={report['macro_f1']:.6f}, updates={steps}", flush=True)
    if steps != 10 * math.ceil(len(targets) / 4) or maximum_delta <= 0:
        raise RuntimeError("actual optimizer updates or parameter changes missing")
    classifier.load_state_dict(best)
    return (
        initial,
        history,
        {
            "selected_epoch": selected,
            "validation_macro_f1": best_score,
            "optimizer_steps": steps,
            "max_head_delta_during_training": maximum_delta,
            "selected_max_head_delta": max(float((v - initial[k]).abs().max()) for k, v in best.items()),
            "tie_rule": "strict improvement, earliest epoch including zero",
        },
    )


def score_rows(records, logits, model, names):
    probabilities = softmax(logits)
    return [
        {
            "id": r["id"],
            "role": r["role"],
            "group_id": r["group_id"],
            "fold": r.get("fold"),
            "label": r["label"],
            "category": names[r["label"]],
            "model": model,
            "predicted": int(np.argmax(row)),
            "logits": row.tolist(),
            "probabilities": prob.tolist(),
        }
        for r, row, prob in zip(records, logits, probabilities, strict=True)
    ]


def train(root):
    import torch

    records = read_json(root / "outputs/records.json")
    names = read_json(root / "outputs/configuration.json")["class_names"]
    setup_start = time.perf_counter()
    pipe = load_pipe(root)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_start
    extractor = pipe.extractor.to_dict()
    write_json(root / "outputs/extractor.json", extractor)
    example = next(r for r in records if r["role"] == "train")
    x = np.load(root / example["audio"])
    inputs = pipe.extractor(x, sampling_rate=16000, return_tensors="pt")["input_values"][0].numpy()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(inputs.T, aspect="auto", origin="lower")
    valid = max(0, (len(x) - 400) // 160 + 1)
    ax.axvline(valid - 0.5, color="red", label=f"Observed frames end ({valid}); padding follows")
    ax.set(xlabel="10 ms frame index", ylabel="Mel bin", title="Actual normalized AST input")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "outputs/model_input.png")
    plt.close(fig)
    pipe.rehead(names, seed=SEED)
    pipe.freeze_backbone()
    before = backbone_hash(pipe.model)
    trainable = {name: p.numel() for name, p in pipe.model.named_parameters() if p.requires_grad}
    if not trainable or any(not n.startswith("classifier.") for n in trainable):
        raise RuntimeError("unexpected trainable backbone parameters")
    subsets = {role: [r for r in records if r["role"] == role] for role in ROLES}
    features = {}
    cache_start = time.perf_counter()
    for role in ("train", "validation"):
        subset = subsets[role]
        features[role] = extract_features(pipe, [np.load(root / r["audio"]) for r in subset])
        np.save(root / f"data/features_{role}.npy", features[role].cpu().numpy())
    torch.cuda.synchronize()
    cache_seconds = time.perf_counter() - cache_start
    identity = cache_identity(root, records, extractor)
    write_json(
        root / "outputs/cache_identity.json",
        {
            "identity": identity,
            "feature_cache_seconds": cache_seconds,
            "files": {role: digest(root / f"data/features_{role}.npy") for role in features},
        },
    )
    targets = {
        role: torch.tensor([r["label"] for r in subset], device=pipe.device)
        for role, subset in subsets.items()
        if role in ("train", "validation")
    }
    torch.cuda.synchronize()
    training_start = time.perf_counter()
    initial, history, selection = train_head(
        pipe.model.classifier,
        features["train"],
        targets["train"],
        features["validation"],
        targets["validation"],
        names,
    )
    torch.cuda.synchronize()
    selection["training_seconds"] = time.perf_counter() - training_start
    selection["setup_including_model_download_seconds"] = setup_seconds
    selection["feature_cache_seconds"] = cache_seconds
    selected = {k: v.detach().cpu().clone() for k, v in pipe.model.classifier.state_dict().items()}
    if before != backbone_hash(pipe.model):
        raise RuntimeError("frozen backbone changed")
    selection.update(
        {
            "backbone_sha256_before": before,
            "backbone_sha256_after": before,
            "trainable_parameters": trainable,
            "pid": os.getpid(),
            "precision": "float32",
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        }
    )
    write_json(root / "outputs/training_history.json", history)
    write_json(root / "outputs/selection.json", selection)
    # Test extraction and label evaluation occur only after selection is irrevocably recorded.
    features["test"] = extract_features(pipe, [np.load(root / r["audio"]) for r in subsets["test"]])
    rows = []
    for model, state in (("initial", initial), ("selected", selected)):
        pipe.model.classifier.load_state_dict(state)
        for role in ("validation", "test"):
            rows.extend(score_rows(subsets[role], head_logits(pipe, features[role]), model, names))
    write_json(root / "outputs/scores.json", rows)
    reports = {"majority": read_json(root / "outputs/baseline.json")["metrics"]}
    for model in ("initial", "selected"):
        reports[model] = {}
        for role in ("validation", "test"):
            subset = [r for r in rows if r["model"] == model and r["role"] == role]
            reports[model][role] = metrics(
                [r["label"] for r in subset], [r["predicted"] for r in subset], names
            )
    write_json(root / "outputs/metrics.json", reports)
    pipe.adaptation_config.update({"recipe": RECIPE, "selection": selection})
    # Safetensors state dict (no pickle) plus the repository's plain-JSON sidecar selected_head.json beside it.
    path = pipe.save_artifact(root / "outputs/selected_head.safetensors")
    from ast_reference.pipeline import MODEL_ID, MODEL_REVISION, artifact_sidecar_path

    write_json(
        root / "outputs/adapter_manifest.json",
        {
            "format": "safetensors state dict + JSON sidecar",
            "sha256": digest(path),
            "sidecar_sha256": digest(artifact_sidecar_path(path)),
            "base_model_id": MODEL_ID,
            "base_model_revision": MODEL_REVISION,
            "class_names": names,
            "activation": "softmax",
            "extractor_sha256": digest(root / "outputs/extractor.json"),
            "recipe": RECIPE,
            "selection": selection,
            "tensor_schema": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in selected.items()
            },
        },
    )


def verify_parity(live, reloaded):
    a, b = np.asarray(live), np.asarray(reloaded)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("reload shape/finite mismatch")
    pa, pb = softmax(a), softmax(b)
    if (
        not np.allclose(a, b, atol=1e-5, rtol=1e-4)
        or not np.allclose(pa, pb, atol=1e-6, rtol=1e-4)
        or not np.array_equal(a.argmax(-1), b.argmax(-1))
    ):
        raise ValueError("reload parity failed")
    return {
        "passed": True,
        "max_logit_delta": float(np.max(np.abs(a - b))),
        "max_probability_delta": float(np.max(np.abs(pa - pb))),
    }


def strict_load(root, pipe):
    import torch
    from ast_reference.pipeline import MODEL_ID, MODEL_REVISION, artifact_sidecar_path
    from safetensors.torch import load_file

    manifest = read_json(root / "outputs/adapter_manifest.json")
    names = read_json(root / "outputs/configuration.json")["class_names"]
    path = root / "outputs/selected_head.safetensors"
    sidecar = artifact_sidecar_path(path)
    if (
        digest(path) != manifest["sha256"]
        or digest(sidecar) != manifest["sidecar_sha256"]
        or manifest["class_names"] != names
        or manifest["activation"] != "softmax"
        or manifest["base_model_id"] != MODEL_ID
        or manifest["base_model_revision"] != MODEL_REVISION
    ):
        raise ValueError("adapter digest, vocabulary or lineage mismatch")
    if digest(root / "outputs/extractor.json") != manifest[
        "extractor_sha256"
    ] or pipe.extractor.to_dict() != read_json(root / "outputs/extractor.json"):
        raise ValueError("extractor identity changed")
    # The sidecar is plain JSON and both digests matched above, so the weights are read only now, as tensors.
    payload = read_json(sidecar)
    if payload["class_names"] != names or payload["activation"] != "softmax":
        raise ValueError("payload vocabulary or activation mismatch")
    pipe.rehead(names, seed=SEED)
    expected = pipe.model.classifier.state_dict()
    state = load_file(str(path), device="cpu")
    if set(state) != set(expected):
        raise ValueError("adapter tensor names mismatch")
    for key, tensor in state.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != expected[key].shape
            or tensor.dtype != expected[key].dtype
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("adapter tensor schema or finiteness mismatch")
    # Real repository loader used after stricter checks; it re-checks the sidecar digest. No pickle is read.
    pipe.load_artifact(path)
    return pipe


def reload(root):
    selection = read_json(root / "outputs/selection.json")
    if selection["pid"] == os.getpid():
        raise RuntimeError("reload must run in a fresh process")
    pipe = strict_load(root, load_pipe(root))
    names = pipe.labels
    records = read_json(root / "outputs/records.json")
    live = read_json(root / "outputs/scores.json")
    probes = sorted((r for r in records if r["role"] == "validation"), key=lambda r: r["id"])[:4]
    tested = probes + [r for r in records if r["role"] == "test"]
    logits = head_logits(pipe, extract_features(pipe, [np.load(root / r["audio"]) for r in tested]))
    live_by_id = {r["id"]: r for r in live if r["model"] == "selected"}
    parity = verify_parity([live_by_id[r["id"]]["logits"] for r in tested], logits)
    rows = score_rows(tested, logits, "reloaded", names)
    test = [r for r in rows if r["role"] == "test"]
    report = metrics([r["label"] for r in test], [r["predicted"] for r in test], names)
    previous = read_json(root / "outputs/metrics.json")["selected"]["test"]
    if report["confusion_matrix"] != previous["confusion_matrix"] or any(
        abs(report[k] - previous[k]) > 1e-12 for k in ("accuracy", "macro_f1")
    ):
        raise ValueError("reload metric parity failed")
    parity.update({"pid": os.getpid(), "probes": len(probes), "test_clips": len(test), "metric_parity": True})
    write_json(root / "outputs/reload_scores.json", rows)
    write_json(root / "outputs/reload_parity.json", parity)


def activity(root):
    pipe = strict_load(root, load_pipe(root))
    names = pipe.labels
    records = {r["id"]: r for r in read_json(root / "outputs/records.json")}
    ids = read_json(root / "outputs/configuration.json")["activity_ids"]
    rows = []
    for number, clip_id in enumerate(ids):
        record = records[clip_id]
        conditions = noise_conditions(np.load(root / record["audio"]), clip_id)
        logits = head_logits(pipe, extract_features(pipe, [c["audio"] for c in conditions]))
        for condition, row in zip(
            conditions, score_rows([record] * 4, logits, "selected", names), strict=True
        ):
            if number == 0:
                plot_audio(
                    condition["audio"],
                    root / f"outputs/noise_examples/{clip_id}_{condition['condition']}.png",
                    f"Gain-matched {condition['condition']} dB",
                )
            row.update({k: v for k, v in condition.items() if k != "audio"})
            rows.append(row)
    summary = []
    clean = {r["id"]: r["predicted"] for r in rows if r["condition"] == "clean"}
    for condition in ("clean", "20", "10", "0"):
        subset = [r for r in rows if r["condition"] == condition]
        report = metrics([r["label"] for r in subset], [r["predicted"] for r in subset], names)
        summary.append(
            {
                "condition": condition,
                **report,
                "changed_predictions": sum(r["predicted"] != clean[r["id"]] for r in subset),
            }
        )
    write_json(root / "outputs/noise_records.json", rows)
    write_json(root / "outputs/noise_summary.json", summary)


def report(root):
    import csv
    import importlib.metadata

    import matplotlib.pyplot as plt

    output = root / "outputs"
    reports = read_json(output / "metrics.json")
    records = read_json(output / "records.json")
    rows = read_json(output / "scores.json")
    initial = {r["id"]: r for r in rows if r["model"] == "initial" and r["role"] == "test"}
    selected = [r for r in rows if r["model"] == "selected" and r["role"] == "test"]
    transitions = {"corrected": 0, "new_errors": 0, "unchanged_correct": 0, "unchanged_incorrect": 0}
    for r in selected:
        old = initial[r["id"]]["predicted"] == r["label"]
        new = r["predicted"] == r["label"]
        transitions[
            ("unchanged_correct" if old else "corrected")
            if new
            else ("new_errors" if old else "unchanged_incorrect")
        ] += 1
    mistakes = sorted(
        (r for r in selected if r["predicted"] != r["label"]),
        key=lambda r: (-max(r["probabilities"]), r["id"]),
    )[:5]
    cm = np.array(reports["selected"]["test"]["confusion_matrix"])
    offdiag = cm.copy()
    np.fill_diagonal(offdiag, 0)
    pair = list(map(int, np.unravel_index(offdiag.argmax(), offdiag.shape))) if offdiag.max() else None
    write_json(
        output / "paired_errors.json",
        {
            **transitions,
            "highest_confidence_mistakes": mistakes,
            "representative_confusion_pair": pair,
            "fixed_example": min(selected, key=lambda r: r["id"]),
            "selection_rule": (
                "test filename minimum; largest off-diagonal count row-major tie; "
                "error softmax descending then filename"
            ),
        },
    )
    names = read_json(output / "configuration.json")["class_names"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, matrix, title in zip(
        axes, (cm, cm / cm.sum(1, keepdims=True)), ("Counts", "Row-normalized"), strict=True
    ):
        im = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=range(len(names)),
            yticks=range(len(names)),
            xticklabels=names,
            yticklabels=names,
            xlabel="Predicted",
            ylabel="Reference",
            title=title,
        )
        ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output / "confusion.png")
    plt.close(fig)
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "scores.csv").open(encoding="utf-8", newline="") as f:
        loaded = list(csv.DictReader(f))
    if len(loaded) != len(rows) or any(
        a["id"] != b["id"] or json.loads(a["logits"]) != b["logits"]
        for a, b in zip(loaded, rows, strict=True)
    ):
        raise ValueError("exported score table round trip failed")
    history = read_json(output / "training_history.json")
    selection = read_json(output / "selection.json")
    best = selection["validation_macro_f1"]
    tied = [row["epoch"] for row in history if row["validation_macro_f1"] == best]
    chosen = next(row for row in history if row["epoch"] == selection["selected_epoch"])
    selection_line = (
        f"Selected epoch: {selection['selected_epoch']} (validation macro-F1 {best:.6f}). "
        f"Updates contained in the selected head: {chosen['optimizer_steps']} of "
        f"{selection['optimizer_steps']} performed in total. "
        f"Epochs tied at the best validation macro-F1: {tied}."
    )
    if len(tied) > 1:
        selection_line += (
            f" {len(tied)} epochs tie, so validation cannot tell them apart; the earliest-tie rule kept "
            f"epoch {selection['selected_epoch']}. This does not show that later epochs were worse."
        )
    test_records = [r for r in records if r["role"] == "test"]
    test_groups = len({r["group_id"] for r in test_records})
    base = read_json(root / "weights/ast/dimer-base-manifest.json")
    backbone = f"Backbone: frozen {base['modelId']} at revision {base['revision'][:12]}"
    head = (
        "Training adapted only the new classifier head (its LayerNorm and linear layer); 'initial' is that "
        "head at its seeded random initialization, and 'majority' always predicts the most frequent "
        "training class."
    )
    original = (
        "The original AudioSet classifier is not evaluated here, so beating the random head is not an "
        "improvement over pretrained AST."
    )
    if read_json(output / "prepare.json")["private_byod"]:
        context_lines = [
            f"{backbone}, fine-tuned on AudioSet's 527 labels; whether your classes are covered by those "
            f"labels is not checked here. {head}",
            f"Scope: {len(test_records)} test clips from {test_groups} recording groups in your manifest's "
            f"split, with one training seed ({RECIPE['seed']}). These are point estimates, and clips from one "
            f"recording group are correlated. {original}",
        ]
    else:
        context_lines = [
            f"{backbone}, already fine-tuned on AudioSet, whose 527 labels include all ten ESC-10 categories. "
            f"{head}",
            f"Scope: {len(test_records)} test clips from {test_groups} source recordings on this fixed split "
            f"(ESC-50 folds 1-3 train, 4 validation, 5 test) with one training seed ({RECIPE['seed']}). These "
            "are point estimates, not the five-fold ESC benchmark, and clips from one source recording are "
            "correlated.",
            "AudioSet category coverage is known; whether any of these recordings were also in the "
            f"backbone's training data is unresolved. {original}",
        ]
    lines = [
        "# Sound event classification results",
        "",
        "| System | Test accuracy | Test macro-F1 |",
        "|---|---:|---:|",
    ]
    for system in ("majority", "initial", "selected"):
        score = reports[system]["test"]
        lines.append(f"| {system} | {score['accuracy']:.6f} | {score['macro_f1']:.6f} |")
    lines += [
        "",
        selection_line,
        "",
        *context_lines,
        "Softmax is uncalibrated; synthetic additive noise is not evidence of real-world noise robustness.",
        "Candidate: this run is evidence for maintainer review, not automatic release approval.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output / "run_summary.json",
        {
            "status": "completed_candidate",
            "records": len(records),
            "class_names": names,
            "test_metrics": reports["selected"]["test"],
            "reload": read_json(output / "reload_parity.json"),
            "private_byod": read_json(output / "prepare.json")["private_byod"],
            "release_approved": False,
        },
    )
    write_json(
        output / "provenance.json",
        {
            "python": sys.version,
            "packages": {
                p: importlib.metadata.version(p)
                for p in ("torch", "torchaudio", "transformers", "numpy", "scipy")
            },
            "source": read_json(root / "source.json"),
            "requirements_sha256": digest(root / "requirements.txt"),
            "stage_receipts_before_report_completion": read_json(root / "stage_receipts.json"),
            "base_manifest": read_json(root / "weights/ast/dimer-base-manifest.json"),
        },
    )
    allow = [
        "data_manifest.json",
        "records.json",
        "prepare.json",
        "configuration.json",
        "baseline.json",
        "extractor.json",
        "cache_identity.json",
        "training_history.json",
        "selection.json",
        "scores.json",
        "scores.csv",
        "metrics.json",
        "selected_head.safetensors",
        "selected_head.json",
        "adapter_manifest.json",
        "reload_scores.json",
        "reload_parity.json",
        "noise_records.json",
        "noise_summary.json",
        "paired_errors.json",
        "summary.md",
        "run_summary.json",
        "provenance.json",
        "confusion.png",
    ]
    checksums = {name: digest(output / name) for name in allow}
    write_json(output / "checksums.json", checksums)
    with zipfile.ZipFile(output / "sound_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in allow + ["checksums.json"]:
            z.write(output / name, name)
    with zipfile.ZipFile(output / "sound_results.zip") as z:
        if any(hashlib.sha256(z.read(name)).hexdigest() != value for name, value in checksums.items()):
            raise ValueError("archive hash mismatch")


def input_identity(root):
    paths = [
        root / "source.json",
        root / "requirements.txt",
        root / "sample.json",
        root / "weights/ast/dimer-base-manifest.json",
        Path(__file__),
    ]
    paths += [root / "outputs/records.json", root / "outputs/configuration.json"]
    result = {
        str(p.relative_to(root)) if p.is_relative_to(root) else p.name: digest(p) for p in paths if p.exists()
    }
    if (root / "outputs/records.json").exists():
        for r in read_json(root / "outputs/records.json"):
            result[r["audio"]] = digest(root / r["audio"])
    return result


STAGE_SECTIONS = {
    "prepare": "Section 2",
    "train": "Section 5",
    "reload": "Section 6",
    "activity": "Section 7",
    "report": "Section 8",
}


def require_complete(root, receipts, stage, previous_stages):
    current = input_identity(root)
    for previous in previous_stages:
        receipt = receipts["stages"].get(previous, {})
        remedy = (
            f"Re-run the '{previous}' cell ({STAGE_SECTIONS[previous]}) and then each later stage cell in order."
        )
        if receipt.get("status") != "complete" or receipt["identity"] != current:
            if receipt.get("status") == "complete":
                state = "stale: its inputs changed after it completed"
            elif receipt.get("status"):
                state = receipt["status"]
            else:
                state = "not run, or reset because an earlier stage was re-run"
            raise RuntimeError(f"stage '{stage}' needs stage '{previous}' to be complete and current ({state}). {remedy}")
        if any(
            not (root / p).exists() or digest(root / p) != sha
            for p, sha in receipt.get("outputs", {}).items()
        ):
            raise RuntimeError(f"an output of stage '{previous}' changed after it completed. {remedy}")


def begin_stage(root, stage):
    path = root / "stage_receipts.json"
    receipts = read_json(path) if path.exists() else {"run_id": uuid.uuid4().hex, "stages": {}}
    idx = STAGES.index(stage)
    if stage == "prepare":
        receipts = {"run_id": uuid.uuid4().hex, "stages": {}}
    else:
        require_complete(root, receipts, stage, STAGES[:idx])
    for descendant in STAGES[idx:]:
        receipts["stages"].pop(descendant, None)
    receipts["stages"][stage] = {"status": "running", "pid": os.getpid()}
    write_json(path, receipts)
    return time.perf_counter()


STAGE_FILES = {
    "prepare": ["data_manifest.json", "records.json", "prepare.json", "configuration.json", "baseline.json"],
    "train": [
        "training_history.json",
        "selection.json",
        "scores.json",
        "metrics.json",
        "selected_head.safetensors",
        "selected_head.json",
        "adapter_manifest.json",
        "extractor.json",
    ],
    "reload": ["reload_scores.json", "reload_parity.json"],
    "activity": ["noise_records.json", "noise_summary.json"],
    "report": ["sound_results.zip", "checksums.json", "run_summary.json"],
}


def finish_stage(root, stage, start):
    receipts = read_json(root / "stage_receipts.json")
    rss = None
    if sys.platform != "win32":
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gpu_peak = None
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.cuda.is_initialized():
        torch_module.cuda.synchronize()
        gpu_peak = torch_module.cuda.max_memory_allocated()
    receipts["stages"][stage] = {
        "status": "complete",
        "pid": os.getpid(),
        "seconds": time.perf_counter() - start,
        "peak_host_rss_bytes": rss,
        "peak_gpu_bytes": gpu_peak,
        "identity": input_identity(root),
        "outputs": {f"outputs/{name}": digest(root / "outputs" / name) for name in STAGE_FILES[stage]},
    }
    write_json(root / "stage_receipts.json", receipts)


def byod_run(root, directory):
    child = root / "byod_runs" / uuid.uuid4().hex
    child.mkdir(parents=True)
    import shutil

    for name in ("source.json", "requirements.txt", "sample.json"):
        shutil.copyfile(root / name, child / name)
    (child / "weights").symlink_to((root / "weights").resolve(), target_is_directory=True)
    for stage in STAGES:
        command = [sys.executable, __file__, "--root", str(child), "--stage", stage]
        if stage == "prepare":
            command += ["--byod", str(directory)]
        completed = subprocess.run(command)
        if completed.returncode == REFUSED_EXIT:
            raise SystemExit(REFUSED_EXIT)  # the child printed the reason once
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)
    write_json(
        root / "outputs/byod_output.json",
        {"root": str(child), "archive": str(child / "outputs/sound_results.zip")},
    )


def infer(root, audio_path):
    if not Path(audio_path).is_file():
        refuse(f"{Path(audio_path).name}: file not found", "Set AUDIO_PATH to an uploaded WAV")
    try:
        x, info = read_audio(audio_path)
    except InputRefused as error:
        refuse(f"{Path(audio_path).name}: {error}", "Use a WAV within the documented limits")
    # The head is the one this root's train stage selected (the ESC-10 head of Section 5 in the canonical
    # run). Like the later stage cells, refuse a missing, failed or stale prepare/train stage.
    path = root / "stage_receipts.json"
    receipts = read_json(path) if path.exists() else {"stages": {}}
    require_complete(root, receipts, "infer", STAGES[: STAGES.index("train") + 1])
    pipe = strict_load(root, load_pipe(root))
    logits = head_logits(pipe, extract_features(pipe, [x]))[0]
    write_json(
        root / "outputs/inference.json",
        {
            "class_names": pipe.labels,
            "logits": logits.tolist(),
            "probabilities": softmax(logits).tolist(),
            "predicted": pipe.labels[int(logits.argmax())],
            "accuracy": None,
            "macro_f1": None,
            "reason": "unlabelled audio; no unknown-class rejection",
            "audio": info,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=(*STAGES, "byod", "infer"), required=True)
    parser.add_argument("--byod", type=Path)
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(exist_ok=True)
    if args.stage == "byod":
        if args.byod is None:
            parser.error("--byod directory required")
        byod_run(root, args.byod.resolve())
        return
    if args.stage == "infer":
        if args.audio is None:
            parser.error("--audio WAV required")
        infer(root, args.audio)
        return
    start = begin_stage(root, args.stage)
    try:
        if args.stage == "prepare":
            if args.byod:
                check_byod_manifest(args.byod.resolve())
            prepare(root, args.byod.resolve() if args.byod else None)
        else:
            globals()[args.stage](root)
        finish_stage(root, args.stage, start)
    except Exception as error:
        # A labelled BYOD manifest or recording that breaks a documented rule is refused, not a crash.
        refused = args.byod is not None and is_input_refusal(error)
        receipts = read_json(root / "stage_receipts.json")
        receipts["stages"][args.stage].update(
            {"status": "refused" if refused else "failed", "error_type": type(error).__name__}
        )
        write_json(root / "stage_receipts.json", receipts)
        if refused:
            refuse(error, "Correct the manifest or recordings in BYOD_PATH")
        raise


if __name__ == "__main__":
    main()
