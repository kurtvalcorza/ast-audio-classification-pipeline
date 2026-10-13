"""The sound workshop generator (SEC2-m8, option A): tools/build_sound_workshop.py is the source of truth.

Static checks with the standard library only; they are not execution evidence.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "build_sound_workshop.py"
NOTEBOOK = ROOT / "tutorials" / "DIMER_Sound_Event_Classification_Workshop.ipynb"


def _generator():
    spec = importlib.util.spec_from_file_location("build_sound_workshop", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _carried() -> tuple[dict[str, str], dict[str, str]]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if "".join(c["source"]).startswith("CARRIED_FILES = "))
    body = ast.parse("".join(cell["source"])).body
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in body if isinstance(n, ast.Assign)}
    return values["CARRIED_FILES"], values["CARRIED_HASHES"]


def test_generator_reproduces_the_committed_notebook_byte_for_byte() -> None:
    assert _generator().build().encode("utf-8") == NOTEBOOK.read_bytes()


def test_generated_from_records_the_generator_identity() -> None:
    generated = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]["generated_from"]
    recorded = _generator().RECORDED_GENERATOR
    assert (generated["generator"], generated["generator_sha256"]) == recorded
    assert generated["generator"].split("/")[0] == GENERATOR.name
    files, _hashes = _carried()
    source = json.loads(files["source.json"])
    assert (source["generator"], source["generator_sha256"]) == recorded


def test_carried_files_are_the_generator_inputs() -> None:
    files, hashes = _carried()
    module = _generator()
    for carried, source in module.CARRIED_SOURCES.items():
        assert files[carried] == (ROOT / source).read_bytes().decode("utf-8").replace("\r\n", "\n"), carried
    for name, text in files.items():
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == hashes[name], name


def test_a_changed_cell_text_breaks_parity() -> None:
    module = _generator()
    kind, text, metadata = module.CELLS[0]
    module.CELLS[0] = (kind, text + " ", metadata)
    assert module.build().encode("utf-8") != NOTEBOOK.read_bytes()


def test_ci_runs_the_generator_check() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python tools/build_sound_workshop.py --check" in workflow
