"""Regression tests for the 2026-10-02 sound workshop review fixes (SEC-M1..M3, SEC-m1..m6).

They execute the notebook's own carried worker functions and display helpers with NumPy and inert
stand-ins, so they need nothing beyond CI's install line (no scipy, matplotlib, IPython or GPU). Source and
helper checks only; they are not hosted execution evidence.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_Sound_Event_Classification_Workshop.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell(cell_id: str) -> str:
    return "".join(next(c for c in _notebook()["cells"] if c["id"] == cell_id)["source"])


def _carried() -> dict[str, str]:
    body = ast.parse(_cell("code-03")).body
    return ast.literal_eval(body[0].value)


@pytest.fixture
def worker(tmp_path: Path):
    for name, text in _carried().items():
        path = tmp_path / "carried" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(
        "sound_workshop_worker", tmp_path / "carried" / "workshop.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Display:
    def __init__(self) -> None:
        self.items: list[object] = []

    def __call__(self, item: object) -> None:
        self.items.append(item)

    def markdown(self) -> str:
        return "\n".join(str(item[1]) for item in self.items if isinstance(item, tuple) and item[0] == "md")


def _helpers(root: Path, display: _Display) -> dict:
    """Exec the function definitions of the notebook's runner cell with inert IPython stand-ins."""
    namespace = {
        "json": json,
        "Path": Path,
        "subprocess": __import__("subprocess"),
        "ROOT": root,
        "ENV": None,
        "PYTHON": Path(sys.executable),
        "display": display,
        "Markdown": lambda text: ("md", text),
        "FileLink": lambda path: ("link", path),
        "Image": lambda filename: ("image", filename),
        "Audio": lambda filename, normalize, autoplay: ("audio", filename),
    }
    tree = ast.parse(_cell("code-04"))
    defs = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    exec(compile(defs, "code-04", "exec"), namespace)
    return namespace


def _byod_records():
    records = []
    index = 0
    for label in range(2):
        for role, count in (("train", 4), ("validation", 2), ("test", 2)):
            for k in range(count):
                records.append(
                    {
                        "id": f"c{index}",
                        "role": role,
                        "label": label,
                        "group_id": f"{role}-{label}-{k}",
                        "waveform_sha256": f"w{index}",
                        "mono_sha256": f"m{index}",
                        "seconds": 1.0,
                    }
                )
                index += 1
    return records


# ---------------------------------------------------------------- SEC-m2: actionable validation messages


def test_byod_class_support_message_names_class_role_and_counts(worker) -> None:
    records = _byod_records()[:-1]  # one frog test clip missing
    with pytest.raises(ValueError, match=r"class 'frog' has 1 test clip\(s\); at least 2 required"):
        worker.validate_records(records, ["owl", "frog"], byod=True)


def test_group_crossing_message_names_the_group(worker) -> None:
    records = _byod_records()
    records[-1]["group_id"] = records[0]["group_id"]
    with pytest.raises(ValueError, match="recording group 'train-0-0' appears in both train and test"):
        worker.validate_records(records, ["owl", "frog"], byod=True)


def test_independent_group_message_names_class_and_role(worker) -> None:
    records = _byod_records()
    for record in records:
        if record["role"] == "validation" and record["label"] == 0:
            record["group_id"] = "same"
    with pytest.raises(ValueError, match="class 'owl' has 1 recording group\\(s\\) in validation"):
        worker.validate_records(records, ["owl", "frog"], byod=True)


def test_missing_field_is_named_instead_of_raising_key_error(worker) -> None:
    records = _byod_records()
    records[3].pop("role")
    with pytest.raises(ValueError, match=r"record 3 is missing required field\(s\) \['role'\]"):
        worker.validate_records(records, ["owl", "frog"], byod=True)


def test_default_frozen_split_still_validates(worker) -> None:
    sample = json.loads(_carried()["sample.json"])
    records = [
        dict(r, waveform_sha256=f"w{i}", mono_sha256=f"m{i}", seconds=5.0)
        for i, r in enumerate(sample["records"])
    ]
    summary = worker.validate_records(records, sample["class_names"], byod=False)
    assert summary["roles"] == {"train": 240, "validation": 80, "test": 80}
    records[0]["role"] = "validation"
    with pytest.raises(ValueError, match="exactly \\(frozen default split\\)"):
        worker.validate_records(records, sample["class_names"], byod=False)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            {"id": "a", "label": "owl", "group_id": "g", "role": "train"},
            r"missing required field\(s\) \['audio'\]",
        ),
        (
            {"id": "a", "audio": "a.wav", "label": "bat", "group_id": "g", "role": "train"},
            "label 'bat' is not in class_names",
        ),
        (
            {"id": "a b", "audio": "a.wav", "label": "owl", "group_id": "g", "role": "train"},
            "id 'a b' must be",
        ),
    ],
)
def test_byod_manifest_errors_are_raised_before_any_audio_is_decoded(
    worker, tmp_path, record, message
) -> None:
    byod = tmp_path / "byod"
    byod.mkdir()
    (byod / "manifest.json").write_text(json.dumps({"class_names": ["owl", "frog"], "records": [record]}))
    root = tmp_path / "run"
    (root / "outputs").mkdir(parents=True)
    with pytest.raises(ValueError, match=message):
        worker.prepare(root, byod)


# ---------------------------------------------------------------- SEC-m5: stale-stage message


def test_stale_stage_message_names_the_stage_and_section(worker, tmp_path) -> None:
    carried = _carried()
    root = tmp_path / "stages"
    (root / "outputs").mkdir(parents=True)
    (root / "weights/ast").mkdir(parents=True)
    for name in ("source.json", "requirements.txt", "sample.json"):
        (root / name).write_text(carried[name], encoding="utf-8")
    (root / "weights/ast/dimer-base-manifest.json").write_text(
        carried["weights/ast/dimer-base-manifest.json"]
    )
    for outputs in worker.STAGE_FILES.values():
        for name in outputs:
            (root / "outputs" / name).write_text("x", encoding="utf-8")
    (root / "outputs/records.json").write_text("[]", encoding="utf-8")
    for stage in worker.STAGES:
        worker.finish_stage(root, stage, worker.begin_stage(root, stage))
    worker.finish_stage(root, "reload", worker.begin_stage(root, "reload"))
    with pytest.raises(
        RuntimeError, match=r"needs stage 'activity'.*Re-run the 'activity' cell \(Section 7\)"
    ):
        worker.begin_stage(root, "report")


# ---------------------------------------------------------------- SEC-m1: log printed once


@pytest.mark.parametrize("code", [0, 3])
def test_run_stage_streams_each_line_once_and_repeats_the_tail_only_on_failure(
    tmp_path, capsys, code
) -> None:
    (tmp_path / "workshop.py").write_text(f"import sys\nprint('marker-line')\nsys.exit({code})\n")
    helpers = _helpers(tmp_path, _Display())
    if code:
        with pytest.raises(RuntimeError, match="failed with exit 3"):
            helpers["run_stage"]("prepare")
    else:
        helpers["run_stage"]("prepare")
    assert capsys.readouterr().out.count("marker-line") == (2 if code else 1)


# ---------------------------------------------------------------- SEC-M2: saturated selection is explained


def _write(root: Path, name: str, data) -> None:
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / name).write_text(json.dumps(data), encoding="utf-8")


def test_selection_summary_reports_ties_and_updates_in_the_selected_head(tmp_path, capsys) -> None:
    history = [{"epoch": 0, "validation_macro_f1": 0.05, "optimizer_steps": 0}]
    history += [{"epoch": e, "validation_macro_f1": 1.0, "optimizer_steps": 60 * e} for e in range(1, 11)]
    _write(tmp_path, "training_history.json", history)
    _write(
        tmp_path, "selection.json", {"selected_epoch": 1, "validation_macro_f1": 1.0, "optimizer_steps": 600}
    )
    _helpers(tmp_path, _Display())["show_selection"]()
    out = capsys.readouterr().out
    assert "Updates contained in the selected head: 60 of 600" in out
    assert "Epochs tied at the best validation macro-F1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]" in out
    assert "10 epochs tie, so validation cannot tell them apart" in out


def test_selection_summary_has_no_tie_sentence_when_one_epoch_wins(tmp_path, capsys) -> None:
    history = [
        {"epoch": e, "validation_macro_f1": f, "optimizer_steps": 60 * e}
        for e, f in enumerate((0.1, 0.8, 0.9))
    ]
    _write(tmp_path, "training_history.json", history)
    _write(
        tmp_path, "selection.json", {"selected_epoch": 2, "validation_macro_f1": 0.9, "optimizer_steps": 120}
    )
    _helpers(tmp_path, _Display())["show_selection"]()
    out = capsys.readouterr().out
    assert "Epochs tied at the best validation macro-F1: [2]" in out and "cannot tell them apart" not in out


# ---------------------------------------------------------------- SEC-M1: paired errors are displayed


def _paired(root: Path, mistakes, pair) -> None:
    _write(root, "configuration.json", {"class_names": ["dog", "rooster", "rain"]})
    _write(
        root,
        "paired_errors.json",
        {
            "corrected": 70,
            "new_errors": 1,
            "unchanged_correct": 7,
            "unchanged_incorrect": 2,
            "highest_confidence_mistakes": mistakes,
            "representative_confusion_pair": pair,
        },
    )


def test_paired_errors_show_transitions_mistakes_and_confusion_pair(tmp_path, capsys) -> None:
    mistake = {"id": "5-1-A-0", "label": 0, "predicted": 1, "probabilities": [0.2, 0.7, 0.1]}
    _paired(tmp_path, [mistake], [0, 1])
    shown = _Display()
    _helpers(tmp_path, shown)["show_paired_errors"]()
    text = shown.markdown()
    for needle in (
        "| corrected | 70 |",
        "| new errors | 1 |",
        "| unchanged incorrect | 2 |",
        "| 5-1-A-0 | dog | rooster | 0.7000 |",
    ):
        assert needle in text
    assert "Most frequent confusion (reference -> predicted): dog -> rooster" in capsys.readouterr().out


def test_paired_errors_say_so_when_there_is_no_mistake(tmp_path, capsys) -> None:
    _paired(tmp_path, [], None)
    _helpers(tmp_path, _Display())["show_paired_errors"]()
    out = capsys.readouterr().out
    assert "no test errors" in out and "Most frequent confusion (reference -> predicted): none" in out


def test_default_cells_call_the_new_displays_and_explain_the_confusion_matrix_beside_it() -> None:
    assert "show_selection()" in _cell("code-10")
    assert "show_paired_errors()" in _cell("code-16")
    report = _cell("code-16")
    assert report.index("show_figures('confusion.png'") < report.index("show_paired_errors()")
    assert "Confusion rows are true labels" not in _cell("md-11")
    assert "rows are reference labels and columns are predictions" in _cell("md-15")
    assert "introduced [count] new errors" in _cell("md-17")  # every slot now has a displayed value


# ---------------------------------------------------------------- SEC-M3, SEC-m3: BYOD results and prose


def test_byod_cell_displays_the_byod_run_not_the_canonical_run(tmp_path, capsys) -> None:
    canonical, byod = tmp_path / "canonical", tmp_path / "byod_run"
    _write(canonical, "byod_output.json", {"root": str(byod)})
    _write(byod, "metrics.json", {"selected": {"test": {"accuracy": 0.625, "macro_f1": 0.6}}})
    history = [
        {"epoch": 0, "validation_macro_f1": 0.5, "optimizer_steps": 0},
        {"epoch": 1, "validation_macro_f1": 0.75, "optimizer_steps": 4},
    ]
    _write(byod, "training_history.json", history)
    _write(byod, "selection.json", {"selected_epoch": 1, "validation_macro_f1": 0.75, "optimizer_steps": 40})
    _write(byod, "reload_parity.json", {"passed": True, "max_logit_delta": 0.0, "probes": 4})
    _write(
        byod,
        "noise_summary.json",
        [{"condition": "clean", "accuracy": 0.75, "macro_f1": 0.7, "changed_predictions": 0}],
    )
    _paired(byod, [], None)
    shown = _Display()
    namespace = _helpers(canonical, shown)
    namespace["run_stage"] = lambda *args, **kwargs: None
    source = (
        _cell("code-18")
        .replace("RUN_BYOD = False", "RUN_BYOD = True")
        .replace("BYOD_PATH = ''", "BYOD_PATH = 'x'")
    )
    exec(compile(source, "code-18", "exec"), namespace)
    text = shown.markdown()
    out = capsys.readouterr().out
    assert "| selected | test | 0.6250 | 0.6000 |" in text
    assert "| clean | 0.7500 | 0.7000 | 0 |" in text
    assert (
        "Updates contained in the selected head: 4 of 40" in out and "Reload parity: {'passed': True" in out
    )
    assert ("link", str(byod / "outputs/sound_results.zip")) in shown.items


def test_byod_cell_is_unchanged_when_disabled(capsys) -> None:
    namespace = {"Path": Path}
    exec(compile(_cell("code-18"), "code-18", "exec"), namespace)
    assert capsys.readouterr().out.strip() == "Optional labelled BYOD disabled."


def test_byod_prose_states_data_movement_warning_and_how_to_place_files() -> None:
    text = _cell("md-17")
    assert "does not send them to any external service" in text  # DAT17
    assert "Do not use confidential, restricted, personal" in text  # DAT18
    assert "Files pane" in text and "BYOD_PATH = 'my_sounds'" in text


# ---------------------------------------------------------------- SEC-m4, SEC-m6, provenance


def test_setup_check_prints_principal_library_versions() -> None:
    tree = ast.parse(_cell("code-04"))
    check = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "VERSION_CHECK"
    )
    compile(check, "VERSION_CHECK", "exec")
    for library in ("torch", "torchaudio", "transformers"):
        assert f"{library}.__version__" in check
    assert "get_device_name" in check


def test_registry_row_records_the_hosted_run() -> None:
    readme = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    row = next(
        line for line in readme.splitlines() if "`DIMER_Sound_Event_Classification_Workshop.ipynb`" in line
    )
    assert "not yet recorded" not in row and "384d9736" in row


def test_post_generation_revision_is_recorded() -> None:
    revisions = _notebook()["metadata"]["dimer"]["generated_from"]["post_generation_revisions"]
    assert any(r["date"] == "2026-10-02" and "workshop.py" in r["carried_files_changed"] for r in revisions)
