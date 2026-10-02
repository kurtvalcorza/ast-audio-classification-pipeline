"""WORKSHOP notebook (NOTEBOOK_SPEC 2.2): carried-source integrity, package parity and negative controls."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_release_assets.py"
WORKSHOP = "DIMER_Sound_Event_Classification_Workshop.ipynb"


def _load_validator(root: Path):
    spec = importlib.util.spec_from_file_location("validate_release_assets_workshop", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    return module


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for name in ("docs", "tutorials", "src"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "weights" / "ast-audioset").mkdir(parents=True)
    manifest = "weights/ast-audioset/dimer-base-manifest.json"
    shutil.copy2(ROOT / manifest, tmp_path / manifest)
    shutil.copy2(ROOT / "LICENSE", tmp_path / "LICENSE")
    return tmp_path


def _carried_cell(notebook: dict) -> dict:
    return next(c for c in notebook["cells"] if "".join(c["source"]).startswith("CARRIED_FILES = "))


def _edit(root: Path, mutate) -> None:
    path = root / "tutorials" / WORKSHOP
    notebook = json.loads(path.read_text(encoding="utf-8"))
    mutate(notebook)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _rewrite_carried(root: Path, name: str, old: str, new: str) -> None:
    def mutate(notebook: dict) -> None:
        cell = _carried_cell(notebook)
        text = "".join(cell["source"])
        body = ast.parse(text).body
        files = ast.literal_eval(body[0].value)
        assert old in files[name]
        files[name] = files[name].replace(old, new, 1)
        segment = ast.get_source_segment(text, body[0].value)
        cell["source"] = [text.replace(segment, repr(files), 1)]

    _edit(root, mutate)


def test_committed_workshop_passes() -> None:
    _load_validator(ROOT).validate_workshop_notebooks()


def test_carried_files_verify_and_pin_esc50() -> None:
    notebook = json.loads((ROOT / "tutorials" / WORKSHOP).read_text(encoding="utf-8"))
    body = ast.parse("".join(_carried_cell(notebook)["source"])).body
    files, hashes = (ast.literal_eval(node.value) for node in body[:2])
    assert {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()} == hashes
    sample = json.loads(files["sample.json"])
    assert sample["metadata_pins"]["csv"]["sha256"] == hashes["licenses/esc-metadata.csv"]
    assert sample["metadata_pins"]["license"]["sha256"] == hashes["licenses/esc-attribution.txt"]
    # The frozen split: official folds 1-3 / 4 / 5, with no source recording in more than one role.
    assert sample["counts"] == {"train": 240, "validation": 80, "test": 80}
    roles = {}
    for record in sample["records"]:
        roles.setdefault(record["group_id"], set()).add(record["role"])
    assert all(len(r) == 1 for r in roles.values())


def test_control_tampered_carried_file_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _rewrite_carried(tree, "workshop.py", "import", "import  ")
    with pytest.raises(module.ValidationError, match="CARRIED_HASHES digest"):
        module.validate_workshop_notebooks()


def test_control_package_drift_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    metrics = tree / "src" / "ast_audio_classification_pipeline" / "metrics.py"
    metrics.write_text(metrics.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    drift = "differs from src/ast_audio_classification_pipeline/metrics.py"
    with pytest.raises(module.ValidationError, match=drift):
        module.validate_workshop_notebooks()


def test_control_unpinned_esc_metadata_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)

    def mutate(nb: dict) -> None:
        cell = _carried_cell(nb)
        text = "".join(cell["source"])
        body = ast.parse(text).body
        carried, hashes = (ast.literal_eval(node.value) for node in body[:2])
        csv_name = "licenses/esc-metadata.csv"
        previous = hashes[csv_name]
        carried[csv_name] += "extra,row\n"
        hashes[csv_name] = hashlib.sha256(carried[csv_name].encode()).hexdigest()
        # Keep source.json and the metadata consistent so only the ESC-50 pin is wrong.
        carried["source.json"] = carried["source.json"].replace(previous, hashes[csv_name])
        hashes["source.json"] = hashlib.sha256(carried["source.json"].encode()).hexdigest()
        nb["metadata"]["dimer"]["generated_from"]["files"][csv_name] = hashes[csv_name]
        text = text.replace(ast.get_source_segment(text, body[0].value), repr(carried), 1)
        text = text.replace(ast.get_source_segment(text, ast.parse(text).body[1].value), repr(hashes), 1)
        cell["source"] = [text]

    _edit(tree, mutate)
    with pytest.raises(module.ValidationError, match="does not match the pinned ESC-50 csv"):
        module.validate_workshop_notebooks()


def test_control_persisted_output_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _edit(tree, lambda notebook: _carried_cell(notebook).__setitem__("execution_count", 3))
    with pytest.raises(module.ValidationError, match="persists outputs"):
        module.validate_workshop_notebooks()


@pytest.mark.parametrize("key", ["notebook_profile", "notebook_mode"])
def test_control_metadata_key_is_required(tree: Path, key: str) -> None:
    module = _load_validator(tree)
    _edit(tree, lambda notebook: notebook["metadata"]["dimer"].pop(key))
    with pytest.raises(module.ValidationError, match=key):
        module.validate_workshop_notebooks()


def test_worker_environment_forces_a_file_backend_for_matplotlib() -> None:
    # Colab exports MPLBACKEND=module://matplotlib_inline.backend_inline, which the isolated worker cannot
    # import; the 2026-09-27 Colab run failed in `prepare` until the runner environment set Agg.
    notebook = json.loads((ROOT / "tutorials" / WORKSHOP).read_text(encoding="utf-8"))
    runner = next(
        "".join(c["source"]) for c in notebook["cells"] if "ENV = dict(os.environ" in "".join(c["source"])
    )
    assert "ENV['MPLBACKEND'] = 'Agg'" in runner
    assert runner.index("ENV['MPLBACKEND'] = 'Agg'") < runner.index("subprocess.run([str(UV), 'venv'")
