"""Deterministic in-code sample audio data: pretrained demonstration tone and acoustic ecology dataset.

Nothing here is downloaded and nothing requires external dependencies beyond numpy,
ensuring zero network dataset dependencies and deterministic execution across clean-room environments.

Two separate label contexts live here and must not be conflated:
1. ``tutorial_tone``: 440 Hz sinusoidal waveform demonstrating pretrained 527-class AudioSet inference.
2. ``synthetic_audio_dataset``: 24 audio clips labelled with ``ADAPT_CLASSES``, the canonical
   Krause/Pijanowski tripartite acoustic ecology vocabulary:
   - ``geophony``: non-biological natural sounds (rain, wind, flowing water).
   - ``biophony``: non-human biological vocalizations (birdsong, insect chirps).
   - ``anthrophony``: human-generated mechanical sounds (engines, drones, machinery).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np

SAMPLE_RATE = 16_000
SAMPLE_DURATION = 3.0  # seconds; comfortably inside AST's 10.24 s window
MIN_DATASET_AUDIO_SECONDS = 0.025
MAX_DATASET_AUDIO_SECONDS = 10.24
TONE_HZ = 440.0
TONE_AMPLITUDE = 0.5

# Canonical Krause/Pijanowski acoustic ecology classification vocabulary
ADAPT_CLASSES: tuple[str, ...] = ("geophony", "biophony", "anthrophony")
DATASET_REPRESENTATION = "io.github.kurtvalcorza.dataset.audio.waveform-classification.v1"


def tutorial_tone(
    duration: float = SAMPLE_DURATION,
    sample_rate: int = SAMPLE_RATE,
    frequency: float = TONE_HZ,
    amplitude: float = TONE_AMPLITUDE,
) -> np.ndarray:
    """Deterministic single-tone audio waveform demonstrating pretrained AudioSet inference."""
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive int")
    if not isinstance(duration, int | float) or isinstance(duration, bool) or duration <= 0:
        raise ValueError("duration must be positive")
    if not isinstance(frequency, int | float) or isinstance(frequency, bool) or frequency <= 0:
        raise ValueError("frequency must be positive")
    if not isinstance(amplitude, int | float) or isinstance(amplitude, bool) or not 0 <= amplitude <= 1:
        raise ValueError("amplitude must be in [0, 1]")
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


def generate_audio_clip(
    index: int,
    class_idx: int,
    duration: float = SAMPLE_DURATION,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 42,
) -> np.ndarray:
    """Generate a deterministic synthetic 1-D audio waveform characteristic of its acoustic class."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative int")
    valid_class_idx = (
        isinstance(class_idx, int)
        and not isinstance(class_idx, bool)
        and 0 <= class_idx < len(ADAPT_CLASSES)
    )
    if not valid_class_idx:
        raise ValueError(f"class_idx must be an int in [0, {len(ADAPT_CLASSES) - 1}]")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"sample_rate must be {SAMPLE_RATE}")
    if not isinstance(duration, int | float) or isinstance(duration, bool):
        raise ValueError("duration must be numeric")
    if not MIN_DATASET_AUDIO_SECONDS <= duration <= MAX_DATASET_AUDIO_SECONDS:
        raise ValueError(
            f"duration must be in [{MIN_DATASET_AUDIO_SECONDS}, {MAX_DATASET_AUDIO_SECONDS}]"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    n_samples = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(seed=seed + index * 101 + class_idx * 17)

    if class_idx == 0:
        # Geophony: broadband turbulent noise with slow gentle envelope (simulating rain/wind)
        raw_noise = rng.standard_normal(n_samples).astype(np.float32)
        # 1-pole low-pass smooth filter
        alpha = 0.15
        filtered = np.zeros_like(raw_noise)
        acc = 0.0
        for i in range(n_samples):
            acc = alpha * raw_noise[i] + (1.0 - alpha) * acc
            filtered[i] = acc
        # Gentle swell envelope
        envelope = 0.7 + 0.3 * np.sin(2.0 * np.pi * 0.5 * t)
        signal = filtered * envelope
    elif class_idx == 1:
        # Biophony: rapid frequency-modulated chirp whistles (simulating bird song / biophonic calls)
        chirp_rate = 3.0 + (index % 3) * 1.5  # chirps per second
        f_center = 2800.0 + (index % 4) * 400.0
        f_dev = 1200.0
        # Phase modulation
        instantaneous_freq = f_center + f_dev * np.sin(2.0 * np.pi * chirp_rate * t)
        phase = 2.0 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
        fundamental = np.sin(phase)
        harmonic = 0.35 * np.sin(2.0 * phase)
        # Pulsed amplitude gate
        gate = np.clip(np.sin(2.0 * np.pi * chirp_rate * t), 0.0, 1.0) ** 2
        signal = (fundamental + harmonic) * gate
    elif class_idx == 2:
        # Anthrophony: low-frequency harmonic drone with motor buzz (simulating engine/machinery)
        f0 = 120.0 + (index % 3) * 20.0  # 120 Hz fundamental
        harmonics = [
            (1.0, 0.45),
            (2.0, 0.30),
            (3.0, 0.20),
            (4.0, 0.15),
            (5.0, 0.10),
        ]
        drone = np.zeros(n_samples, dtype=np.float32)
        for mult, weight in harmonics:
            drone += weight * np.sin(2.0 * np.pi * (f0 * mult) * t)
        # Add motor stroke amplitude pulsation (15 Hz)
        engine_pulse = 0.75 + 0.25 * np.sin(2.0 * np.pi * 15.0 * t)
        signal = drone * engine_pulse
    # Normalize peak amplitude safely to [-0.85, 0.85]
    peak = np.max(np.abs(signal))
    if peak > 1e-6:
        signal = (signal / peak) * 0.85
    return signal.astype(np.float32)


def synthetic_audio_dataset(
    n_samples: int = 24,
    seed: int = 42,
    duration: float = SAMPLE_DURATION,
    sample_rate: int = SAMPLE_RATE,
) -> list[dict[str, Any]]:
    """Generate a balanced deterministic dataset across ADAPT_CLASSES."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < len(ADAPT_CLASSES) * 2:
        raise ValueError(f"n_samples must be an int >= {len(ADAPT_CLASSES) * 2}")
    if n_samples % len(ADAPT_CLASSES):
        raise ValueError(f"n_samples must be divisible by {len(ADAPT_CLASSES)} for class balance")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    num_classes = len(ADAPT_CLASSES)
    records: list[dict[str, Any]] = []
    for idx in range(n_samples):
        class_idx = idx % num_classes
        waveform = generate_audio_clip(
            index=idx,
            class_idx=class_idx,
            duration=duration,
            sample_rate=sample_rate,
            seed=seed,
        )
        records.append(
            {
                "id": f"clip-{idx:03d}",
                "waveform": waveform,
                "sample_rate": sample_rate,
                "label": class_idx,
                "class_name": ADAPT_CLASSES[class_idx],
                "duration": round(float(duration), 4),
            }
        )
    return records


def split_dataset(
    records: Sequence[dict[str, Any]],
    val_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified train/validation split preserving balanced class distribution."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    by_class: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        by_class.setdefault(r["label"], []).append(r)

    rng = random.Random(seed)
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []

    for label in sorted(by_class.keys()):
        class_list = list(by_class[label])
        if len(class_list) < 2:
            raise ValueError(f"class {label} requires at least 2 records for a disjoint split")
        rng.shuffle(class_list)
        n_val = min(len(class_list) - 1, max(1, round(len(class_list) * val_fraction)))
        val_records.extend(class_list[:n_val])
        train_records.extend(class_list[n_val:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def validate_dataset(
    records: Sequence[dict[str, Any]],
    class_names: Sequence[str] = ADAPT_CLASSES,
) -> dict[str, Any]:
    """Validate that audio records strictly conform to the classification contract."""
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise TypeError(f"records must be a sequence, got {type(records).__name__}")
    if len(records) < 2:
        raise ValueError(f"dataset requires at least 2 records, got {len(records)}")

    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("class_names must contain at least 2 classes")
    if any(not isinstance(name, str) or not name.strip() for name in class_names):
        raise ValueError("class_names must contain non-empty strings")
    if len(set(class_names)) != num_classes:
        raise ValueError("class_names must be unique")
    class_counts = [0] * num_classes
    seen_ids: set[str] = set()

    for i, r in enumerate(records):
        if not isinstance(r, dict):
            raise TypeError(f"record {i} must be a dict, got {type(r).__name__}")
        if "waveform" not in r:
            raise KeyError(f"record {i} missing required key 'waveform'")
        if "label" not in r:
            raise KeyError(f"record {i} missing required key 'label'")
        if "id" not in r:
            raise KeyError(f"record {i} missing required key 'id'")
        if "sample_rate" not in r:
            raise KeyError(f"record {i} missing required key 'sample_rate'")
        if "class_name" not in r:
            raise KeyError(f"record {i} missing required key 'class_name'")

        record_id = r["id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"record {i} id must be a non-empty string")
        if record_id in seen_ids:
            raise ValueError(f"record {i} duplicates id {record_id!r}")
        seen_ids.add(record_id)

        sample_rate = r["sample_rate"]
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"record {i} sample_rate must be {SAMPLE_RATE}, got {sample_rate!r}")

        waveform = r["waveform"]
        if not isinstance(waveform, np.ndarray):
            raise TypeError(f"record {i} waveform must be a np.ndarray, got {type(waveform).__name__}")
        if waveform.ndim != 1:
            raise ValueError(f"record {i} waveform must be 1-D mono, got shape {waveform.shape}")
        if waveform.dtype != np.float32:
            raise TypeError(f"record {i} waveform must be float32, got {waveform.dtype}")
        if not np.all(np.isfinite(waveform)):
            raise ValueError(f"record {i} waveform contains NaN or inf values")
        if waveform.size == 0:
            raise ValueError(f"record {i} waveform must not be empty")
        if np.max(np.abs(waveform)) > 1.0:
            raise ValueError(f"record {i} waveform samples must be in [-1, 1]")
        duration = waveform.size / sample_rate
        if not MIN_DATASET_AUDIO_SECONDS <= duration <= MAX_DATASET_AUDIO_SECONDS:
            raise ValueError(
                f"record {i} duration {duration:.4f} s outside "
                f"[{MIN_DATASET_AUDIO_SECONDS}, {MAX_DATASET_AUDIO_SECONDS}]"
            )

        label = r["label"]
        if isinstance(label, bool) or not isinstance(label, int):
            raise TypeError(f"record {i} label must be an int, got {type(label).__name__}")
        if not 0 <= label < num_classes:
            raise ValueError(f"record {i} label {label} outside valid range [0, {num_classes - 1}]")
        if r["class_name"] != class_names[label]:
            raise ValueError(
                f"record {i} class_name {r['class_name']!r} does not match "
                f"label {label} ({class_names[label]!r})"
            )

        class_counts[label] += 1

    missing_classes = [class_names[i] for i, count in enumerate(class_counts) if count == 0]
    if missing_classes:
        raise ValueError(f"dataset has no records for classes: {missing_classes}")

    return {
        "representation": DATASET_REPRESENTATION,
        "n_records": len(records),
        "num_classes": num_classes,
        "class_counts": {class_names[c]: class_counts[c] for c in range(num_classes)},
        "verdict": "accepted",
    }
