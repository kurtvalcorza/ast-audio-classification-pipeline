"""The isolated-environment tutorial path (NOTEBOOK_SPEC 2.2 §25.13): the kernel's `run_stage` helper and the carried
stage runner.

* The kernel-side tests execute the generated notebook's own carrier and `run_stage` code (no model library needed in
  the kernel): the carried files are written and hash-verified into a run directory, and a failing stage stops the
  kernel with a RuntimeError that repeats the stage's own message.
* The CPU pre-flight runs every stage in order, in this process, against a small randomly initialised AST stand-in (the
  real architecture and feature extractor, not the pretrained weights). Each stage builds its own pipeline, so
  everything a later stage uses crosses over through files in the run directory. It proves the stage plumbing and the
  hand-offs, not the model's accuracy.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from pathlib import Path

import pytest

from conftest import ROOT, load_tool, pcm_wav, run_stage_inproc

build = load_tool("build_notebook")
TEMPLATE = load_tool("notebook_template").TEMPLATE


def _infrastructure_sources() -> tuple[str, str]:
    notebook = build.render(ROOT, TEMPLATE, "test-revision")
    code = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    carrier = next(s for s in code if s.startswith("# @title Infrastructure: write and verify the carried"))
    install = next(s for s in code if s.startswith("# @title Infrastructure: install the locked runtime"))
    return carrier, install


def kernel(tmp_path: Path) -> dict:
    """The kernel namespace after the carrier cell and the helper definitions, with the current interpreter standing
    in for the isolated environment's Python."""
    carrier, install = _infrastructure_sources()
    run_root = tmp_path / "run"
    run_root.mkdir()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    namespace = {"ROOT": run_root, "WEIGHTS": tmp_path / "weights", "PYTHON": Path(sys.executable), "ENV": env, "Path": Path}
    exec("import hashlib\nimport json\nimport subprocess\n" + carrier, namespace)  # noqa: S102 - the notebook's own cell
    exec(install[install.index("def run_stage(") :], namespace)  # noqa: S102
    return namespace


def test_carrier_writes_and_verifies_every_carried_file(tmp_path: Path) -> None:
    ns = kernel(tmp_path)
    for dest, source in TEMPLATE["carried"].items():
        assert (ns["ROOT"] / dest).read_bytes() == (ROOT / source).read_text(encoding="utf-8").encode("utf-8"), dest
    assert ns["NOTEBOOK_SOURCE"]["revision"] == "test-revision"


def test_failed_stage_raises_in_the_kernel_with_the_stage_message(tmp_path: Path) -> None:
    """A labelled ZIP with one class missing is refused by the `dataset` stage (no model needed) and the kernel cell
    raises with the runner's own message (run as a real subprocess through `run_stage`)."""
    ns = kernel(tmp_path)
    archive = tmp_path / "two_classes.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for i, name in enumerate(("geophony/a.wav", "geophony/b.wav", "biophony/c.wav", "biophony/d.wav")):
            handle.writestr(name, pcm_wav(seed=i))
    (ns["ROOT"] / "state").mkdir()
    (ns["ROOT"] / "state" / "input_manifest.json").write_text(json.dumps({"findings": []}), encoding="utf-8")
    try:
        ns["run_stage"]("dataset", "--zip", archive)
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - the stage must fail
        raise AssertionError("the dataset stage accepted a ZIP without anthrophony clips")
    assert "Stage 'dataset' failed (exit 2)" in message
    assert "no clips for ['anthrophony']" in message


def test_kernel_cells_install_nothing_into_the_kernel() -> None:
    notebook = build.render(ROOT, TEMPLATE, "test-revision")
    kernel_code = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources"))
    assert not re.search(r"\bpip\b[^\n]*\binstall\b", kernel_code.replace("'pip', 'install', '--python', str(PYTHON)", ""))
    assert "sys.executable" not in kernel_code
    assert "'--require-hashes'" in kernel_code and "'--only-binary', ':all:'" in kernel_code
    for library in ("import torch", "import transformers", "import torchaudio", "import numpy"):
        assert library not in kernel_code


def test_cpu_preflight_runs_every_stage_through_files(ast_stages, capsys) -> None:
    pytest.importorskip("torchaudio")  # the runtime stage checks it against the lock
    stages, run_root, weights = ast_stages
    for stage, options in (
        ("runtime", ()),
        ("pretrained", ()),
        ("dataset", ()),
        ("baseline", ()),
        ("finetune", ("--epochs", "2")),
        ("evaluate", ()),
        ("reload", ()),
        ("bundle", ()),
    ):
        run_stage_inproc(stages, run_root, weights, stage, *options)
    out = run_root / "outputs"
    stem = stages.STEM
    for name in (
        f"{stem}_pretrained.json",
        f"{stem}_input_manifest.json",
        f"{stem}_split.json",
        f"{stem}_pre_adaptation.json",
        f"{stem}_training_history.json",
        f"{stem}_evaluation_report.json",
        f"{stem}_unseen_prediction.json",
        f"{stem}_top_k.csv",
        f"{stem}_reload_parity.json",
        f"{stem}_result.json",
        stages.ADAPTER,
    ):
        assert (out / name).is_file(), name
    result = json.loads((out / f"{stem}_result.json").read_text(encoding="utf-8"))
    assert result["reload_parity"]["reload_verification"] == "PASSED"
    assert result["reload_parity"]["clips_compared"] == 7 and result["reload_parity"]["scores_compared"] == 21
    assert set(result["files"]) >= {f"{stem}_evaluation_report.json", stages.ADAPTER, f"{stem}_top_k.csv"}
    history = json.loads((out / f"{stem}_training_history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history["history"]] == [1, 2]
    printed = capsys.readouterr().out
    assert "'rejected'" in printed and "reload_verification" in printed


def test_stage_run_out_of_order_is_refused(ast_stages) -> None:
    stages, run_root, weights = ast_stages
    error = run_stage_inproc(stages, run_root, weights, "evaluate", expect_ok=False)
    assert "split.json is missing" in error["message"]
