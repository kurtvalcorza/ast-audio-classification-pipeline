"""Refused user input is shown as one clean line (NOTEBOOK_SPEC 3.0 DAT19/DAT20).

Each refusal path runs the generated notebook's own kernel code (the carrier cell, the ``run_stage`` and
``obtain_upload`` helpers, and the learner's BYOD cell) against the real stage runner in a child process, as the
notebook does, with only ``google.colab``'s upload dialog stood in. For every path the cell must raise ``Refused`` (an
exception, so Run all stops at that cell), render as exactly one ``Refused: <file>: <reason>`` line through IPython's
``_render_traceback_`` hook, name the file the learner supplied, and print no ``Traceback``. The earlier run's results
are removed, so none can be read as this input's result. A genuine failure still raises ``RuntimeError`` with the
stage's traceback. The Section 5 content case is the hosted refusal run's input: a text file (``/etc/hostname`` in a
fresh Colab runtime) given as the WAV.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

from conftest import ROOT, load_tool, pcm_wav
from test_tutorial_stages import kernel

build = load_tool("build_notebook")
TEMPLATE = load_tool("notebook_template").TEMPLATE
STEM = "ast_audio_classification"
REFUSAL_CELL_ID = "ast_audio_classification-20"  # the hosted refusal run's halting cell: Section 5's BYOD cell


def _cell_record(needle: str) -> dict:
    found = [c for c in build.render(ROOT, TEMPLATE, "t")["cells"] if c["cell_type"] == "code" and needle in c["source"]]
    assert len(found) == 1, needle
    return found[0]


def _cell(needle: str) -> str:
    return _cell_record(needle)["source"]


def _byod_cell(section: int, path: str) -> str:
    use, field = ("USE_BYOD", "BYOD_WAV_PATH") if section == 5 else ("USE_BYOD_DATASET", "BYOD_ZIP_PATH")
    cell = _cell(f"{use} = False  # @param")
    lines = []
    for line in cell.split("\n"):
        if line.startswith(f"{use} = "):
            line = f"{use} = True" + line[line.index("  #"):]
        elif line.startswith(f"{field} = "):
            line = f"{field} = {path!r}" + line[line.index("  #"):]
        lines.append(line)
    return "\n".join(lines)


@contextlib.contextmanager
def _colab(*uploads: dict):
    queue = list(uploads)
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: queue.pop(0))
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _run_all(ns: dict, cells: list[str]) -> tuple[BaseException | None, str, int]:
    """Run cells in order as Run all does: the first cell that raises stops the run. Returns the exception, the cell
    output (stdout plus the exception as IPython renders it) and how many cells ran."""
    out, ran, raised = io.StringIO(), 0, None
    with contextlib.redirect_stdout(out):
        for source in cells:
            ran += 1
            try:
                exec(compile(source, f"<cell {ran}>", "exec"), ns)  # noqa: S102 - the notebook's own cells
            except BaseException as exc:  # noqa: BLE001 -- what IPython would show is asserted below
                raised = exc
                break
    shown = out.getvalue()
    if raised is not None and hasattr(raised, "_render_traceback_"):
        shown += "\n".join(raised._render_traceback_()) + "\n"
    return raised, shown, ran


def _assert_refused(raised, shown: str, ran: int, file: str) -> str:
    assert type(raised).__name__ == "Refused", (raised, shown)
    lines = raised._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith(f"Refused: {file}: "), lines
    assert [line for line in shown.splitlines() if "Refused:" in line] == lines, shown
    assert "Traceback" not in shown and "Error" not in shown.replace("Refused", ""), shown
    assert ran == 1, "Run all must stop at the refusing cell"
    return lines[0]


def _with_earlier_results(ns: dict) -> Path:
    """An earlier default run's results on disk, as a learner who ran the notebook once would have them."""
    out, state = ns["ROOT"] / "outputs", ns["ROOT"] / "state"
    out.mkdir(exist_ok=True)
    state.mkdir(exist_ok=True)
    for name in (f"{STEM}_pretrained.json", f"{STEM}_split.json", f"{STEM}_evaluation_report.json", "ast-audio-adapter-v2.safetensors", "runtime.json"):
        (out / name).write_text("{}", encoding="utf-8")
    for name in ("input_manifest.json", "split.json", "adapter.json"):
        (state / name).write_text(json.dumps({"findings": []}), encoding="utf-8")
    return out


def test_section5_text_file_as_wav_is_one_refused_line_and_clears_earlier_results(tmp_path: Path) -> None:
    """The hosted refusal run's input: a file that exists but is not a WAV (``/etc/hostname`` in Colab), set through
    ``USE_BYOD=True;BYOD_WAV_PATH=/etc/hostname``. The refusal fires in the cell with id ``REFUSAL_CELL_ID``."""
    assert _cell_record("USE_BYOD = False  # @param")["id"] == REFUSAL_CELL_ID
    ns = kernel(tmp_path)
    out = _with_earlier_results(ns)
    hostname = tmp_path / "hostname"
    hostname.write_text("colab-runtime\n", encoding="utf-8")
    raised, shown, ran = _run_all(ns, [_byod_cell(5, str(hostname)), "print('AFTER THE REFUSAL')"])
    line = _assert_refused(raised, shown, ran, "hostname")
    assert "not a readable PCM WAV" in line and "run the Section 5 cell again" in line
    assert "AFTER THE REFUSAL" not in shown
    assert sorted(p.name for p in out.iterdir()) == ["runtime.json"]
    assert not (ns["ROOT"] / "state" / "input_manifest.json").exists()
    error = json.loads((ns["ROOT"] / "state" / "pretrained.error.json").read_text(encoding="utf-8"))
    assert error["refused"] is True and error["file"] == "hostname"


def test_section5_missing_path_is_one_refused_line(tmp_path: Path) -> None:
    ns = kernel(tmp_path)
    missing = str(tmp_path / "absent.wav")
    raised, shown, ran = _run_all(ns, [_byod_cell(5, missing)])
    line = _assert_refused(raised, shown, ran, missing)
    assert "BYOD_WAV_PATH is not a file" in line


def test_section6_zip_missing_a_class_is_one_refused_line(tmp_path: Path) -> None:
    ns = kernel(tmp_path)
    out = _with_earlier_results(ns)
    archive = tmp_path / "my_sounds.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for i, name in enumerate(("geophony/a.wav", "geophony/b.wav", "biophony/c.wav", "biophony/d.wav")):
            handle.writestr(name, pcm_wav(seed=i))
    raised, shown, ran = _run_all(ns, [_byod_cell(6, str(archive))])
    line = _assert_refused(raised, shown, ran, "my_sounds.zip")
    assert "no clips for ['anthrophony']" in line and "run the Section 6 cell again" in line
    assert sorted(p.name for p in out.iterdir()) == ["runtime.json"]
    assert (ns["ROOT"] / "state" / "input_manifest.json").exists()  # Section 5's record stays: only Section 6 reruns
    assert not (ns["ROOT"] / "state" / "split.json").exists()


def test_section6_file_that_is_not_a_zip_is_one_refused_line(tmp_path: Path) -> None:
    ns = kernel(tmp_path)
    _with_earlier_results(ns)
    fake = tmp_path / "notes.zip"
    fake.write_text("not an archive\n", encoding="utf-8")
    raised, shown, ran = _run_all(ns, [_byod_cell(6, str(fake))])
    _assert_refused(raised, shown, ran, "notes.zip")


@pytest.mark.parametrize(
    ("uploads", "file"),
    [
        ({}, "(no file)"),
        ({"a.wav": pcm_wav(), "b.wav": pcm_wav(seed=1)}, "a.wav, b.wav"),
        ({"clip.mp3": b"ID3"}, "clip.mp3"),
    ],
)
def test_upload_dialog_refusals_are_one_refused_line(tmp_path: Path, uploads: dict, file: str) -> None:
    ns = kernel(tmp_path)
    with _colab(uploads):
        raised, shown, ran = _run_all(ns, [_byod_cell(5, "")])
    _assert_refused(raised, shown, ran, file)


def test_empty_path_outside_colab_is_one_refused_line(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # importing it raises ImportError
    ns = kernel(tmp_path)
    raised, shown, ran = _run_all(ns, [_byod_cell(6, "")])
    _assert_refused(raised, shown, ran, "BYOD_ZIP_PATH")


def test_a_genuine_failure_is_still_an_error_with_its_traceback(tmp_path: Path) -> None:
    """A stage run out of order is a notebook failure, not a refusal: RuntimeError, and the stage's traceback shows."""
    ns = kernel(tmp_path)
    raised, shown, ran = _run_all(ns, ["run_stage('evaluate')"])
    assert type(raised) is RuntimeError and "split.json is missing" in str(raised)
    assert "Traceback" in shown and "Refused:" not in shown
