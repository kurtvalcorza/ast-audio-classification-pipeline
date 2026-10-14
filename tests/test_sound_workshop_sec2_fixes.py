"""Regression tests for the 2026-10-12 sound workshop review fixes (SEC2-m1..m7, except m8).

They run the notebook's own carried worker and runner helpers with NumPy and inert stand-ins, so they
need nothing beyond CI's install line. A test that decodes valid WAVs needs SciPy and skips without it;
the byod-stage test needs directory symlinks and skips where they are not permitted. Source and helper
checks only; they are not hosted execution evidence.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from test_sound_workshop_review_fixes import _carried, _cell, _Display, _helpers, _notebook, _write

ROOT = Path(__file__).resolve().parents[1]
REFUSED_EXIT = 65


def _wav(path: Path, seconds: float, seed: int, rate: int = 16000) -> None:
    rng = np.random.default_rng(seed)
    samples = (rng.uniform(-0.3, 0.3, int(seconds * rate)) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


@pytest.fixture
def carried_root(tmp_path: Path) -> Path:
    root = tmp_path / "carried"
    for name, text in _carried().items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return root


def _run_stage(root: Path, *args: str):
    """The notebook's own run_stage helper against the carried worker, as code-18 and code-20 call it."""
    return _helpers(root, _Display())["run_stage"](*args)


def _manifest(directory: Path, class_names: list[str], records: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps({"class_names": class_names, "records": records}))
    return directory


# ---------------------------------------------------------------- SEC2-m1: versions and device are shown


def test_setup_cell_announces_the_install_and_prints_the_captured_version_check() -> None:
    source = _cell("code-04")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and any(
            isinstance(arg, ast.List) and any(getattr(e, "id", None) == "VERSION_CHECK" for e in arg.elts)
            for arg in node.args
        )
    ]
    assert len(calls) == 1
    keywords = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert keywords.get("capture_output") == "True" and keywords.get("text") == "True"
    assert "check" not in keywords  # a failure prints stdout and stderr before raising
    assert "print(versions.stdout.strip())" in source
    assert source.index("print('Creating the isolated Python") < source.index("[str(UV), 'venv'")


# ---------------------------------------------------------------- SEC2-m2: refusals are one clean message


def _assert_one_clean_refusal(out: str, rule: str) -> None:
    assert "Traceback" not in out
    assert out.count(rule) == 1
    assert len(re.findall(r"^Refused: ", out, re.M)) == 1
    assert "and run this cell again." in out


def test_refused_byod_manifest_is_one_message_and_no_kernel_error(carried_root, tmp_path, capsys) -> None:
    record = {"id": "a_0", "audio": "a.wav", "label": "cat", "group_id": "g", "role": "train"}
    byod = _manifest(tmp_path / "my_sounds", ["bird", "car"], [record])
    assert _run_stage(carried_root, "prepare", "--byod", str(byod)) is False
    _assert_one_clean_refusal(capsys.readouterr().out, "label 'cat' is not in class_names")
    receipts = json.loads((carried_root / "stage_receipts.json").read_text())
    assert receipts["stages"]["prepare"]["status"] == "refused"


def test_refused_unlabelled_recording_is_one_message_and_no_kernel_error(
    carried_root, tmp_path, capsys
) -> None:
    _wav(tmp_path / "too_short.wav", 0.2, 1)
    assert _run_stage(carried_root, "infer", "--audio", str(tmp_path / "too_short.wav")) is False
    _assert_one_clean_refusal(capsys.readouterr().out, "WAV duration must be 0.5-10 seconds")


def test_refused_byod_group_rule_after_decoding_is_one_message(carried_root, tmp_path, capsys) -> None:
    pytest.importorskip("scipy")
    records = []
    for label in ("bird", "car"):
        for role, count in (("train", 4), ("validation", 2), ("test", 2)):
            for k in range(count):
                name = f"{label}_{role}_{k}"
                _wav(tmp_path / "my_sounds" / "wav" / f"{name}.wav", 1.0, len(records))
                same = role == "validation" and label == "bird"
                records.append(
                    {
                        "id": name,
                        "audio": f"wav/{name}.wav",
                        "label": label,
                        "role": role,
                        "group_id": "bird_val_same_source" if same else f"{label}_{role}_g{k % 2}",
                    }
                )
    byod = _manifest(tmp_path / "my_sounds", ["bird", "car"], records)
    assert _run_stage(carried_root, "prepare", "--byod", str(byod)) is False
    _assert_one_clean_refusal(capsys.readouterr().out, "recording group(s) in validation")


def test_genuine_failures_still_raise(tmp_path, capsys) -> None:
    (tmp_path / "workshop.py").write_text("import sys\nprint('marker-line')\nsys.exit(1)\n")
    with pytest.raises(RuntimeError, match="failed with exit 1"):
        _run_stage(tmp_path, "train")
    assert capsys.readouterr().out.count("marker-line") == 2  # streamed, then the tail next to the error


def test_byod_stage_passes_a_refused_child_on_without_a_traceback(carried_root, tmp_path) -> None:
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are not permitted on this host (the byod stage creates one)")
    record = {"id": "a_0", "audio": "a.wav", "label": "cat", "group_id": "g", "role": "train"}
    byod = _manifest(tmp_path / "my_sounds", ["bird", "car"], [record])
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    done = subprocess.run(
        [
            sys.executable,
            str(carried_root / "workshop.py"),
            "--root",
            str(carried_root),
            "--stage",
            "byod",
            "--byod",
            str(byod),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert done.returncode == REFUSED_EXIT
    assert "Traceback" not in done.stderr and "CalledProcessError" not in done.stderr
    assert done.stdout.count("label 'cat' is not in class_names") == 1


@pytest.mark.parametrize(
    ("cell", "switch", "path"),
    [("code-18", "RUN_BYOD", "BYOD_PATH"), ("code-20", "RUN_UNLABELLED", "AUDIO_PATH")],
)
def test_optional_cells_show_nothing_further_after_a_refusal(tmp_path, capsys, cell, switch, path) -> None:
    shown = _Display()
    namespace = _helpers(tmp_path, shown)
    namespace["run_stage"] = lambda *args, **kwargs: False
    source = (
        _cell(cell).replace(f"{switch} = False", f"{switch} = True").replace(f"{path} = ''", f"{path} = 'x'")
    )
    exec(compile(source, cell, "exec"), namespace)  # must not raise or read any stale output
    assert shown.items == [] and capsys.readouterr().out == ""


# ---------------------------------------------------------------- SEC2-m3: exported summary


class _Axes:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _Figure(_Axes):
    def savefig(self, path, *args, **kwargs) -> None:
        Path(path).write_bytes(b"png")


def _fake_matplotlib(monkeypatch) -> None:
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda *args, **kwargs: (_Figure(), (_Axes(), _Axes()))
    pyplot.close = lambda *args: None
    package = types.ModuleType("matplotlib")
    package.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", package)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)


def _run_report(carried_root, monkeypatch, private_byod=False, records=()) -> str:
    import importlib.metadata
    import importlib.util

    spec = importlib.util.spec_from_file_location("sec2_worker", carried_root / "workshop.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    _fake_matplotlib(monkeypatch)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0")
    root = carried_root
    for name in worker.STAGE_FILES["train"] + [
        "reload_scores.json",
        "noise_records.json",
        "noise_summary.json",
        "data_manifest.json",
        "baseline.json",
        "cache_identity.json",
    ]:
        _write(root, name, {})
    names = ["dog", "rain"]
    cm = [[1, 0], [1, 0]]
    score = {"accuracy": 0.5, "macro_f1": 0.33, "confusion_matrix": cm}
    _write(root, "metrics.json", {m: {"test": score} for m in ("majority", "initial", "selected")})
    rows = [
        {
            "id": f"t{i}",
            "model": model,
            "role": "test",
            "label": i,
            "predicted": 0,
            "logits": [0.5, 0.25],
            "probabilities": [0.6, 0.4],
        }
        for model in ("initial", "selected")
        for i in range(2)
    ]
    _write(root, "scores.json", rows)
    _write(root, "records.json", list(records))
    _write(root, "configuration.json", {"class_names": names})
    _write(root, "reload_parity.json", {"passed": True})
    _write(root, "prepare.json", {"private_byod": private_byod})
    history = [{"epoch": 0, "validation_macro_f1": 0.05, "optimizer_steps": 0}]
    history += [{"epoch": e, "validation_macro_f1": 1.0, "optimizer_steps": 60 * e} for e in range(1, 11)]
    _write(root, "training_history.json", history)
    _write(root, "selection.json", {"selected_epoch": 1, "validation_macro_f1": 1.0, "optimizer_steps": 600})
    (root / "stage_receipts.json").write_text("{}", encoding="utf-8")
    worker.report(root)
    return (root / "outputs" / "summary.md").read_text(encoding="utf-8")


def test_exported_summary_states_updates_in_the_selected_head(carried_root, monkeypatch) -> None:
    summary = _run_report(carried_root, monkeypatch)
    assert "Selected epoch: 1 (validation macro-F1 1.000000)." in summary
    assert "Updates contained in the selected head: 60 of 600 performed in total." in summary
    assert "Epochs tied at the best validation macro-F1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]." in summary
    assert "10 epochs tie, so validation cannot tell them apart" in summary
    assert "actual updates: 600" not in summary


# ---------------------------------------------------------------- SEC2-m4: registry row


def test_registry_row_names_the_hosted_run_of_blob_586a465() -> None:
    readme = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    row = next(
        line for line in readme.splitlines() if "`DIMER_Sound_Event_Classification_Workshop.ipynb`" in line
    )
    assert "`586a465`" in row and "2026-10-03" in row and "`92cda1c`" in row
    assert "needs its own hosted run" not in row


# ---------------------------------------------------------------- SEC2-m5: claim level


def test_prose_states_the_audioset_label_coverage_and_bounds_the_claims() -> None:
    assert "already include all ten ESC-10 categories" in _cell("md-00")
    labels = (
        "Chainsaw",
        "Tick",
        "Crackle",
        "Fire",
        "Baby cry, infant cry",
        "Dog",
        "Helicopter",
        "Rain",
        "Chicken, rooster",
        "Waves, surf",
        "Sneeze",
    )
    assert all(label in _cell("md-07") for label in labels)
    assert "remains the evidence of adaptation" not in _cell("md-11")
    assert "**Classes outside AudioSet:**" in _cell("md-17")
    assert "include all ten ESC-10 categories" in _cell("md-21")


# ---------------------------------------------------------------- SEC2-m6: Section 10's head


def test_section_10_names_its_head(carried_root) -> None:
    assert "**ESC-10 head selected in Section 5**" in _cell("md-19")
    assert "Scored with the ESC-10 head selected in Section 5" in _cell("code-20")


def test_infer_requires_complete_prepare_and_train_receipts(carried_root, tmp_path, monkeypatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("sec2_worker_infer", carried_root / "workshop.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(worker, "read_audio", lambda path: (np.zeros(16000, np.float32), {}))
    monkeypatch.setattr(
        worker, "load_pipe", lambda root: pytest.fail("loaded a model without a train receipt")
    )
    (carried_root / "outputs").mkdir(exist_ok=True)
    (tmp_path / "x.wav").write_bytes(b"")  # exists; read_audio is a stand-in
    with pytest.raises(RuntimeError, match=r"stage 'infer' needs stage 'prepare'.*\(Section 2\)"):
        worker.infer(carried_root, tmp_path / "x.wav")
    receipts = {
        "stages": {
            "prepare": {"status": "complete", "identity": worker.input_identity(carried_root), "outputs": {}}
        }
    }
    (carried_root / "stage_receipts.json").write_text(json.dumps(receipts), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"stage 'infer' needs stage 'train'.*\(Section 5\)"):
        worker.infer(carried_root, tmp_path / "x.wav")


# ---------------------------------------------------------------- SEC2-m7: write-up layer and kind


def test_notice_references_and_kind_are_declared() -> None:
    notebook = _notebook()
    assert notebook["metadata"]["dimer"]["notebook_kind"] == "WORKSHOP"
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.2"
    first_code = next(i for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code")
    before = "".join("".join(c["source"]) for c in notebook["cells"][:first_code])
    assert re.search(r"^> \*\*⚠️ Notice\.\*\* For research, training and evaluation only\.", before, re.M)
    assert "without warranty of any kind" in before and "accept no liability" in before
    markdown = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"]
    refs = markdown[-1][markdown[-1].index("## References") :]
    entries = set(re.findall(r"^\[(\d+)\] ", refs, re.M))
    assert entries == {"1", "2", "3", "4", "5"}
    assert "https://doi.org/10.21437/Interspeech.2021-698" in refs
    assert "https://doi.org/10.1145/2733373.2806390" in refs
    body = "\n".join(markdown).replace(refs, "")
    assert set(re.findall(r"\[(\d+)\]", body)) == entries  # every citation resolves; every entry is cited


def test_post_generation_revision_2026_10_13_is_recorded() -> None:
    revisions = _notebook()["metadata"]["dimer"]["generated_from"]["post_generation_revisions"]
    revision = next(r for r in revisions if r["date"] == "2026-10-13")
    assert revision["carried_files_changed"] == ["workshop.py", "source.json"]
    worker = _carried()["workshop.py"]
    assert "REFUSED_EXIT = 65" in worker


# ---------------------------------------------------------------- peer-review follow-up (CX1-CX9)

_TEST_RECORDS = [
    {"role": "test", "group_id": "a"},
    {"role": "test", "group_id": "a"},
    {"role": "test", "group_id": "b"},
    {"role": "train", "group_id": "c"},
]


def test_default_summary_carries_its_own_qualifications(carried_root, monkeypatch) -> None:
    summary = _run_report(carried_root, monkeypatch, records=_TEST_RECORDS)
    for needle in (
        "Backbone: frozen MIT/ast-finetuned-audioset-10-10-0.4593 at revision f826b80d2822",
        "whose 527 labels include all ten ESC-10 categories",
        "only the new classifier head (its LayerNorm and linear layer)",
        "'initial' is that head at its seeded random initialization",
        "Scope: 3 test clips from 2 source recordings on this fixed split",
        "with one training seed (42)",
        "training data is unresolved",
        "beating the random head is not an improvement over pretrained AST",
    ):
        assert needle in summary


def test_byod_summary_uses_dataset_specific_wording(carried_root, monkeypatch) -> None:
    summary = _run_report(carried_root, monkeypatch, private_byod=True, records=_TEST_RECORDS)
    assert "whether your classes are covered by those labels is not checked here" in summary
    assert "Scope: 3 test clips from 2 recording groups in your manifest's split" in summary
    assert "ESC" not in summary


_MISSING_AUDIO = {
    "class_names": ["bird", "car"],
    "records": [{"id": "a", "audio": "missing.wav", "label": "bird", "group_id": "g", "role": "train"}],
}


@pytest.mark.parametrize(
    ("manifest", "rule"),
    [
        ({"records": []}, "missing required key(s) ['class_names']"),
        ({"class_names": ["bird", "car"]}, "missing required key(s) ['records']"),
        ({"class_names": ["bird", "car"], "records": None}, "records must be a list of 1-400 record objects"),
        (
            {"class_names": ["bird", "car"], "records": {"a": 1}},
            "records must be a list of 1-400 record objects",
        ),
        ({"class_names": "bird", "records": [{}]}, "class_names must be a list"),
        ({"class_names": ["bird", "car"], "records": [None]}, "manifest record 0 must be an object"),
        ([1, 2], "must be a JSON object"),
        (_MISSING_AUDIO, "audio file 'missing.wav' not found under BYOD_PATH"),
    ],
)
def test_malformed_manifest_is_refused_cleanly(carried_root, tmp_path, capsys, manifest, rule) -> None:
    byod = tmp_path / "my_sounds"
    byod.mkdir()
    (byod / "manifest.json").write_text(json.dumps(manifest))
    assert _run_stage(carried_root, "prepare", "--byod", str(byod)) is False
    _assert_one_clean_refusal(capsys.readouterr().out, rule)


def test_invalid_json_manifest_is_refused_cleanly(carried_root, tmp_path, capsys) -> None:
    byod = tmp_path / "my_sounds"
    byod.mkdir()
    (byod / "manifest.json").write_text("{not json")
    assert _run_stage(carried_root, "prepare", "--byod", str(byod)) is False
    _assert_one_clean_refusal(capsys.readouterr().out, "manifest.json is not valid UTF-8 JSON")


def _load_worker(carried_root: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, carried_root / "workshop.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def _worker_main(carried_root: Path, tmp_path: Path, monkeypatch, worker, failure: Exception):
    _wav(tmp_path / "my_sounds" / "a.wav", 1.0, 3)
    record = {"id": "a", "audio": "a.wav", "label": "bird", "group_id": "g", "role": "train"}
    byod = _manifest(tmp_path / "my_sounds", ["bird", "car"], [record])

    def failing_prepare(root, directory):
        raise failure

    monkeypatch.setattr(worker, "prepare", failing_prepare)
    argv = ["workshop.py", "--root", str(carried_root), "--stage", "prepare", "--byod", str(byod)]
    monkeypatch.setattr(sys, "argv", argv)


def test_failure_after_successful_validation_still_raises(
    carried_root, tmp_path, monkeypatch, capsys
) -> None:
    worker = _load_worker(carried_root, "sec2_worker_main")
    _worker_main(carried_root, tmp_path, monkeypatch, worker, ValueError("figure backend failed"))
    with pytest.raises(ValueError, match="figure backend failed"):
        worker.main()
    assert "Refused" not in capsys.readouterr().out
    receipts = json.loads((carried_root / "stage_receipts.json").read_text())
    assert receipts["stages"]["prepare"]["status"] == "failed"


def test_wrapped_recording_violation_is_still_a_refusal(carried_root, tmp_path, monkeypatch, capsys) -> None:
    worker = _load_worker(carried_root, "sec2_worker_cause")
    try:
        try:
            raise worker.InputRefused("WAV duration must be 0.5-10 seconds")
        except worker.InputRefused as cause:
            raise ValueError("record 'a' (a.wav): WAV duration must be 0.5-10 seconds") from cause
    except ValueError as wrapped:
        failure = wrapped
    _worker_main(carried_root, tmp_path, monkeypatch, worker, failure)
    with pytest.raises(SystemExit) as exit_info:
        worker.main()
    assert exit_info.value.code == REFUSED_EXIT
    assert capsys.readouterr().out.count("WAV duration must be 0.5-10 seconds") == 1
    receipts = json.loads((carried_root / "stage_receipts.json").read_text())
    assert receipts["stages"]["prepare"]["status"] == "refused"


def test_missing_unlabelled_recording_is_refused_cleanly(carried_root, tmp_path, capsys) -> None:
    assert _run_stage(carried_root, "infer", "--audio", str(tmp_path / "nowhere.wav")) is False
    _assert_one_clean_refusal(capsys.readouterr().out, "nowhere.wav: file not found")


def _noise_rows() -> list[dict]:
    predictions = {"clean": (0, 0), "20": (0, 0), "10": (0, 0), "0": (1, 1)}
    rows = []
    for index, (clip, label) in enumerate((("a", 0), ("b", 1))):
        for condition, predicted in predictions.items():
            rows.append(
                {
                    "id": clip,
                    "label": label,
                    "condition": condition,
                    "predicted": predicted[index],
                    "gain": 0.5 if clip == "a" else 0.25,
                    "peak": 0.4,
                    "clipped_samples": 0,
                    "actual_snr_db": None if condition == "clean" else float(condition) + 0.01,
                }
            )
    return rows


def test_noise_display_shows_controls_and_paired_outcomes(tmp_path, capsys) -> None:
    _write(tmp_path, "noise_records.json", _noise_rows())
    summary = [{"condition": "clean", "accuracy": 0.5, "macro_f1": 0.33, "changed_predictions": 0}]
    _write(tmp_path, "noise_summary.json", summary)
    shown = _Display()
    _helpers(tmp_path, shown)["show_noise"]()
    text = shown.markdown()
    assert "| clean | 2 | none added (matched clean) | 0.2500 to 0.5000 | 0.4000 | 0 |" in text
    assert "| 0 | 2 | 0.01 to 0.01 | 0.2500 to 0.5000 | 0.4000 | 0 |" in text
    assert "| 0 | 1 | 1 | 0 | 0 |" in text  # clip b corrected, clip a newly wrong
    assert "| 20 | 0 | 0 | 1 | 1 |" in text
    out = capsys.readouterr().out
    assert "These are 2 reused validation clips (two per class), not new held-out evidence." in out
    assert "show_noise()" in _cell("code-14") and "show_noise(byod_root)" in _cell("code-18")
    assert "so it is not a count of new errors" in _cell("md-15")


def test_audio_figures_use_fixed_shared_scales() -> None:
    source = _carried()["workshop.py"]
    assert "WAVEFORM_LIMITS = (-1.0, 1.0)" in source and "FBANK_LIMITS = (-16.0, 11.0)" in source
    assert "ylim=WAVEFORM_LIMITS" in source and "vmin=FBANK_LIMITS[0], vmax=FBANK_LIMITS[1]" in source
    assert 'label="Log filterbank energy (fixed scale)"' in source


def test_follow_up_prose() -> None:
    md11 = _cell("md-11")
    assert "If epoch zero is selected, the exported head is the unchanged initial head." in md11
    assert "If test performance worsens, report that outcome." in md11
    assert "remains the evidence that the new head learned" not in md11
    assert "beating the random head is not an improvement over pretrained AST" in _cell("md-07")
    assert "**How precise are these numbers?**" in _cell("md-17")
    md21 = _cell("md-21")
    assert "233 clips tagged CC0, 165 tagged CC BY and two tagged CC Sampling+" in md21
    assert "304 unique source recordings: 180 tagged CC0, 122 CC BY and 2 CC Sampling+" in md21
    assert "is unresolved" in md21
