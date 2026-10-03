"""Regression tests for the 2026-10-02 review of tutorials/ast_audio_classification_colab.ipynb (AST-M1..M3, AST-m1..m6).

Each test names the finding and the acceptance check it guards. Model-dependent checks use the small randomly
initialised AST stand-in from conftest (not the pretrained weights) and skip cleanly where torch is absent.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

from conftest import ROOT, load_tool, pcm_wav, run_stage_inproc

sys.path.insert(0, str(ROOT / "src"))

build = load_tool("build_notebook")
TEMPLATE = load_tool("notebook_template").TEMPLATE
stages = load_tool("tutorial_stages")
validator = load_tool("validate_release_assets")
CLASSES = ("geophony", "biophony", "anthrophony")


@pytest.fixture(scope="module")
def notebook() -> dict:
    return build.render(ROOT, TEMPLATE, "test-revision")


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


def _section(notebook: dict, number: int) -> str:
    """The markdown of one numbered section (heading up to the next numbered heading)."""
    text = _markdown(notebook)
    start = text.index(f"## {number}. ")
    nxt = re.search(r"^## (\d+\.|Troubleshooting|Interpretation)", text[start + 5 :], re.M)
    return text[start : start + 5 + nxt.start()] if nxt else text[start:]


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as handle:
        for name, payload in members.items():
            handle.writestr(name, payload)
    return path


def _valid_members(prefix: str = "") -> dict[str, bytes]:
    return {f"{prefix}{c}/{c}_{i}.wav": pcm_wav(freq=200 + 300 * k + 7 * i, seed=10 * k + i) for k, c in enumerate(CLASSES) for i in range(2)}


# ---- AST-M1: one-pass Run all — nothing installed into the kernel, no restart -------------------------------------------


def test_m1_no_kernel_install_and_no_restart_instruction(notebook: dict) -> None:
    code = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources"))
    assert "sys.executable" not in code and "'-m', 'pip'" not in code
    assert "'venv', '--managed-python', '--python', '3.12.12'" in code
    assert "'--require-hashes', '--only-binary', ':all:'" in code
    assert "Restart the runtime" not in code
    markdown = _markdown(notebook).lower()
    assert "restart" not in markdown.replace("no runtime restart", "").replace("no restart", "")
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


def test_m1_release_record_states_the_restart_and_pass_count() -> None:
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "after one expected restart" not in record
    assert "not a one-pass `Run all`" in record
    assert "2 passes" in record


# ---- AST-M2: distinct clips, no train/evaluation duplicates, duplicates refused -----------------------------------------


def test_m2_default_dataset_has_24_distinct_waveforms_and_no_split_overlap() -> None:
    from ast_audio_classification_pipeline import split_dataset, validate_dataset

    records = stages.default_dataset()
    validate_dataset(records)
    digests = {stages.sha256_array(r["waveform"]) for r in records}
    assert len(records) == 24 and len(digests) == 24
    train, evaluation = split_dataset(records, val_fraction=0.25, seed=42)
    train_digests = {stages.sha256_array(r["waveform"]) for r in train}
    assert sum(stages.sha256_array(r["waveform"]) in train_digests for r in evaluation) == 0
    assert stages.sha256_array(stages.unseen_clip()) not in digests


def test_m2_duplicate_clip_in_a_byod_zip_is_refused_by_name(tmp_path: Path) -> None:
    members = _valid_members()
    members["geophony/copy_of_first.wav"] = members["geophony/geophony_0.wav"]
    records, _source = stages.load_zip_dataset(_zip(tmp_path / "dup.zip", members), CLASSES)
    with pytest.raises(ValueError, match=r"identical waveforms found \(geophony/geophony_0\.wav = geophony/copy_of_first\.wav\)"):
        stages.refuse_duplicates(records)


def test_m2_dataset_stage_reports_zero_duplicates(ast_stages) -> None:
    st, run_root, weights = ast_stages
    run_stage_inproc(st, run_root, weights, "pretrained")
    run_stage_inproc(st, run_root, weights, "dataset")
    split = json.loads((run_root / "outputs" / f"{st.STEM}_split.json").read_text(encoding="utf-8"))
    assert split["distinct_waveforms"] == 24 and split["train_evaluation_duplicates"] == 0
    assert split["counts"]["evaluation"] == {c: 2 for c in CLASSES}


# ---- AST-M3: the guided layer and a working Predict → Change → Run activity ---------------------------------------------


def test_m3_guided_layer_present(notebook: dict) -> None:
    markdown = _markdown(notebook)
    missing = [m for m in validator.GUIDED_MARKDOWN_MARKERS if m not in markdown]
    assert not missing
    for marker, minimum in validator.GUIDED_MIN_COUNTS.items():
        assert markdown.count(marker) >= minimum, marker
    activity = next(c["source"] for c in notebook["cells"] if c["cell_type"] == "code" and "RUN_ACTIVITY = False" in c["source"])
    assert "if RUN_ACTIVITY:" in activity and "run_stage('activity', '--snr-db', ACTIVITY_SNR_DB)" in activity


def test_m3_activity_control_reaches_the_computation_and_leaves_canonical_outputs(ast_stages, monkeypatch) -> None:
    st, run_root, weights = ast_stages
    for stage, options in (("pretrained", ()), ("dataset", ()), ("finetune", ("--epochs", "1"))):
        run_stage_inproc(st, run_root, weights, stage, *options)
    seen = []
    original = st.add_noise_at_snr
    monkeypatch.setattr(st, "add_noise_at_snr", lambda w, snr, rng: seen.append(snr) or original(w, snr, rng))
    before = {p.name: p.read_bytes() for p in (run_root / "outputs").iterdir() if p.is_file()}
    run_stage_inproc(st, run_root, weights, "activity", "--snr-db", "-10")
    run_stage_inproc(st, run_root, weights, "activity", "--snr-db", "20")
    assert seen == [-10.0] * 6 + [20.0] * 6
    records = sorted((run_root / "outputs" / "activity").glob("*.json"))
    assert [p.name for p in records] == [f"{st.STEM}_activity_snr_20db.json", f"{st.STEM}_activity_snr_minus10db.json"]
    assert {p.name: p.read_bytes() for p in (run_root / "outputs").iterdir() if p.is_file()} == before


# ---- AST-m1: argmax decision rule and uncalibrated scores ---------------------------------------------------------------


def test_m1_minor_decision_rule_and_calibration_stated(notebook: dict) -> None:
    for number in (10, 11):
        text = _section(notebook, number)
        assert "argmax" in text and "uncalibrated" in text, number
    assert "softmax probability" not in _markdown(notebook)
    assert "uncalibrated" in stages.SCORE_SEMANTICS and "argmax" in stages.DECISION_RULE


# ---- AST-m2: metrics explained; baseline fitted on training labels; independence stated ---------------------------------


def test_m2_minor_baseline_is_fitted_on_training_labels() -> None:
    baseline = stages.training_majority_baseline([0, 0, 0, 1, 2], [1, 1, 2], list(CLASSES))
    assert baseline["predicted_class"] == "geophony" and baseline["fitted_on"] == "train"
    assert baseline["accuracy"] == 0.0  # an evaluation-label oracle would have predicted biophony and scored 0.6667


def test_m2_minor_metrics_and_split_assumption_explained(notebook: dict) -> None:
    section10 = _section(notebook, 10)
    for needle in ("**Accuracy**", "**Macro-F1**", "**confusion matrix**, rows are the true class and columns the predicted class", "*precision*", "*recall*"):
        assert needle in section10, needle
    section7 = _section(notebook, 7)
    assert "**assumes the clips are independent**" in section7 and "**held-out evaluation set**" in section7


# ---- AST-m3: archive layouts and member-named refusals ------------------------------------------------------------------


def test_m3_minor_parent_folder_and_macosx_are_handled(tmp_path: Path) -> None:
    members = _valid_members("mydata/")
    members["__MACOSX/mydata/geophony/._geophony_0.wav"] = b"\x00\x05\x16\x07"
    members["mydata/.DS_Store"] = b"junk"
    records, source = stages.load_zip_dataset(_zip(tmp_path / "nested.zip", members), CLASSES)
    assert len(records) == 6 and source["stripped_parent_folder"] == "mydata"
    assert sorted(source["skipped_metadata_entries"]) == ["__MACOSX/mydata/geophony/._geophony_0.wav", "mydata/.DS_Store"]


@pytest.mark.parametrize(
    ("member", "wav", "needles"),
    [
        ("geophony/loud_44k.wav", {"rate": 44_100}, ("geophony/loud_44k.wav", "44100 Hz", "resample the clip to 16000 Hz")),
        ("biophony/long_11s.wav", {"seconds": 11.0}, ("biophony/long_11s.wav", "11.00 s", "trim the clip to at most 10.24 s")),
    ],
    ids=["sample-rate", "duration"],
)
def test_m3_minor_rate_and_duration_refusals_name_the_member_and_the_fix(tmp_path: Path, member, wav, needles) -> None:
    members = _valid_members()
    members[member] = pcm_wav(**wav)
    with pytest.raises(ValueError) as caught:
        stages.load_zip_dataset(_zip(tmp_path / "bad.zip", members), CLASSES)
    for needle in needles:
        assert needle in str(caught.value)


def test_m3_minor_wrong_layout_says_how_to_zip(tmp_path: Path) -> None:
    members = {f"a/b/{name}": payload for name, payload in _valid_members().items()}
    with pytest.raises(ValueError, match="zip the three class folders geophony/, biophony/, anthrophony/"):
        stages.load_zip_dataset(_zip(tmp_path / "deep.zip", members), CLASSES)


# ---- AST-m4: infrastructure cells titled and collapsed ------------------------------------------------------------------


def test_m4_minor_infrastructure_cells_are_titled_and_collapsed(notebook: dict) -> None:
    infra = [c for c in notebook["cells"] if c["cell_type"] == "code" and c["source"].startswith("# @title Infrastructure: ")]
    assert len(infra) == 4
    for cell in infra:
        assert cell["metadata"]["cellView"] == "form" and cell["metadata"]["jupyter"]["source_hidden"] is True
    carrier = [c for c in notebook["cells"] if c["metadata"].get("dimer", {}).get("embedded_sources")]
    assert len(carrier) == 1 and carrier[0] in infra
    assert "You may run them without studying their implementation" in _markdown(notebook)


# ---- AST-m5: timing labelled, version check real, no claims beyond the evidence -----------------------------------------


def test_m5_minor_claims_and_timing(notebook: dict) -> None:
    markdown = _markdown(notebook)
    assert not validator.UNSUPPORTED_LEARNER_CLAIMS.search(markdown)
    assert "~10–15 s on CPU" not in markdown
    assert "measured 1.3 s for the five epochs in a warm CPU process (Windows, Python 3.12)" in markdown
    assert "a few minutes is an estimate" in markdown
    assert "**stops if any differs**" in _section(notebook, 4)


def test_m5_minor_runtime_stage_refuses_a_version_that_differs_from_the_lock(ast_stages) -> None:
    pytest.importorskip("torchaudio")
    st, run_root, weights = ast_stages
    lock = run_root / "requirements.txt"
    lock.write_text(re.sub(r"^torch==\S+", "torch==0.0.1", lock.read_text(encoding="utf-8"), flags=re.M), encoding="utf-8")
    error = run_stage_inproc(st, run_root, weights, "runtime", expect_ok=False)
    assert "installed versions differ from the carried lock" in error["message"] and "'locked': '0.0.1'" in error["message"]


# ---- AST-m6: BYOD path fields work without google.colab -----------------------------------------------------------------


def _obtain_upload(tmp_path: Path):
    install = next(c["source"] for c in build.render(ROOT, TEMPLATE, "t")["cells"] if c["source"].startswith("# @title Infrastructure: install the locked"))
    ns = {"ROOT": tmp_path, "Path": Path}
    exec(install[install.index("def obtain_upload(") :], ns)  # noqa: S102 - the notebook's own helper
    return ns["obtain_upload"]


def test_m6_minor_path_field_reads_from_disk_without_google_colab(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # importing it would raise ImportError
    obtain_upload = _obtain_upload(tmp_path)
    archive = _zip(tmp_path / "mine.zip", _valid_members())
    assert obtain_upload(str(archive), ".zip", "BYOD_ZIP_PATH") == archive.resolve()
    with pytest.raises(RuntimeError, match="set BYOD_ZIP_PATH to its path"):
        obtain_upload("", ".zip", "BYOD_ZIP_PATH")
    with pytest.raises(FileNotFoundError, match="BYOD_ZIP_PATH"):
        obtain_upload(str(tmp_path / "absent.zip"), ".zip", "BYOD_ZIP_PATH")


def test_m6_minor_byod_zip_runs_through_adaptation_evaluation_and_reload(ast_stages, tmp_path: Path) -> None:
    pytest.importorskip("torchaudio")  # the runtime stage checks it against the lock
    st, run_root, weights = ast_stages
    archive = _zip(tmp_path / "byod.zip", {f"{c}/{c}_{i}.wav": pcm_wav(freq=150 + 400 * k + 9 * i, seed=k * 10 + i) for k, c in enumerate(CLASSES) for i in range(4)})
    wav = tmp_path / "one.wav"
    wav.write_bytes(pcm_wav(rate=16_000, seconds=2.0))
    for stage, options in (("runtime", ()), ("pretrained", ("--wav", wav)), ("dataset", ("--zip", archive)), ("baseline", ()), ("finetune", ("--epochs", "1")), ("evaluate", ()), ("reload", ()), ("bundle", ())):
        run_stage_inproc(st, run_root, weights, stage, *options)
    result = json.loads((run_root / "outputs" / f"{st.STEM}_result.json").read_text(encoding="utf-8"))
    assert result["input_manifest"]["dataset_kind"] == "BYOD" and result["input_manifest"]["dataset_source"]["archive"] == "byod.zip"
    assert result["evaluation_report"]["sample_kind"] == "BYOD_heldout_evaluation"
    assert result["reload_parity"]["reload_verification"] == "PASSED"
    pretrained = json.loads((run_root / "outputs" / f"{st.STEM}_pretrained.json").read_text(encoding="utf-8"))
    assert pretrained["sample_kind"] == "BYOD" and pretrained["name"] == "one.wav"


def test_committed_notebook_passes_the_validator() -> None:
    validator.validate_notebooks()
