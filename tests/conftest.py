# ruff: noqa: E501
import builtins
import importlib.util
import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in {"torch", "transformers"}:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def load_tool(name: str):
    """Import tools/<name>.py as a module (the tools directory is not a package)."""
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pcm_wav(seconds: float = 1.0, rate: int = 16_000, freq: float = 440.0, channels: int = 1, seed: int = 0) -> bytes:
    """A 16-bit PCM WAV payload; `seed` adds a little noise so payloads differ."""
    t = np.arange(int(seconds * rate)) / rate
    signal = 0.4 * np.sin(2 * np.pi * freq * t) + 0.01 * np.random.default_rng(seed).standard_normal(t.size)
    frames = np.repeat((signal * 32767).astype("<i2")[:, None], channels, axis=1)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    return buffer.getvalue()


def _tiny_pipeline():
    """A small randomly initialised AST with the real architecture and feature extractor (a CPU stand-in, NOT the
    pretrained model): same 527-label head shape, 128 mel bins and 1024-frame window, one 32-wide layer."""
    import torch
    from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

    from ast_audio_classification_pipeline import ASTAudioClassificationPipeline

    torch.manual_seed(0)
    config = ASTConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        frequency_stride=16, time_stride=16, num_labels=527,
        id2label={i: f"label_{i}" for i in range(527)}, label2id={f"label_{i}": i for i in range(527)},
    )
    model = ASTForAudioClassification(config).eval()
    pipe = ASTAudioClassificationPipeline(None, [f"label_{i}" for i in range(527)], "cpu", model=model, extractor=ASTFeatureExtractor())
    pipe._refresh_runner()
    return pipe


@pytest.fixture
def ast_stages(monkeypatch, tmp_path):
    """The carried stage runner with the model factories replaced by the tiny stand-in, and a run directory holding
    the carried files the stages read (package via sys.path, lock, manifest, source.json)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    stages = load_tool("tutorial_stages")

    def from_artifact(run, artifact):
        pipe = _tiny_pipeline()
        pipe.load_artifact(artifact)
        return pipe

    monkeypatch.setattr(stages, "load_pipeline", lambda run: _tiny_pipeline())
    monkeypatch.setattr(stages, "load_from_artifact", from_artifact)
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "requirements.txt").write_text((ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (run_root / "source.json").write_text('{"repository": "test", "revision": "test-revision"}', encoding="utf-8")
    if str(ROOT / "src") not in sys.path:
        monkeypatch.syspath_prepend(str(ROOT / "src"))
    return stages, run_root, tmp_path / "weights"


def run_stage_inproc(stages, run_root: Path, weights: Path, stage: str, *options: str, expect_ok: bool = True) -> dict | None:
    """Run one stage in this process; return the parsed error record on failure."""
    import json

    code = stages.main(["--root", str(run_root), "--weights", str(weights), "--stage", stage, *map(str, options)])
    error = run_root / "state" / f"{stage}.error.json"
    if expect_ok:
        assert code == 0, error.read_text(encoding="utf-8") if error.exists() else f"{stage} exited {code}"
        return None
    assert code == 2, f"{stage} was expected to fail"
    return json.loads(error.read_text(encoding="utf-8"))
