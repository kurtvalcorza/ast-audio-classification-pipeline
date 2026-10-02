# AST Acoustic Ecology E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/ast-audio-classification-pipeline`  
**Notebook:** `tutorials/ast_audio_classification_colab.ipynb`  
**Reviewed commit:** `256b732a65087eb68a019986a2b897cce2d22b76` (`main`, confirmed with `gh api repos/kurtvalcorza/ast-audio-classification-pipeline/commits/main`)  
**Notebook Git blob:** `0be72542ca1f99e9d60b64aac59a36727e3c6cf6`  
**Finding prefix:** `AST`

## Executive assessment

The notebook is a well-engineered standalone reference pipeline. It carries the package modules byte for byte (generator `--check` passes), pins and digest-verifies an immutable model revision, keeps both upload branches off by default, loads the adapter with `weights_only=True`, and writes a complete machine-readable output bundle. Its one documented clean-runtime execution (Kaggle T4, this exact blob) completed all 16 code cells.

That execution needed a restart. On a fresh Kaggle image the install cell raised its own "Restart the runtime, then rerun from the top" error, because NumPy 2.0.2 was already imported. The repository records the run as PASS without saying that this breaks the `Run all` contract (AST-M1). The held-out evaluation is also weaker than its prose suggests. All eight `anthrophony` clips are one identical waveform, `biophony` has four distinct waveforms for eight records, and both held-out `anthrophony` clips are byte-identical to training clips (AST-M2). And although the notebook declares `GUIDED` mode, it gives the learner nothing to do beyond running cells: no prediction, no optional experiment, no checkpoint and no troubleshooting (AST-M3).

None of this shows that the adaptation code is wrong. It shows that the default path does not meet `Run all`, that the "held-out" result partly measures memorisation, and that the guided-learning promise is not delivered.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata and opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening, `tutorials/README.md`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** (2026-09-26), `ml-worker` `origin/main` `b1cfe13`, blob `7428d5becb8d…`. 2.2 additions (GDL1–GDL15) are `SHOULD`s and, per spec §33, a 2.0/2.1 notebook is not non-conformant solely for lacking them |
| Intended audience | Stated: "basic Python and PyTorch; familiarity with acoustic spectrograms and classification baselines" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12", CPU or CUDA, float32 |
| Promised outcomes | Pinned AudioSet inference (527 labels, independent sigmoid); 24-clip synthetic three-class dataset; seeded 18/6 split; re-head + freeze; 5-epoch head fine-tune; held-out accuracy / macro-F1 vs majority baseline; unseen-clip inference; classifier-head adapter export; fresh reload with numeric parity; provenance bundle; two optional BYOD branches |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; carried modules from `src/ast_audio_classification_pipeline/` @ `6796ae71` |

### Evidence actually obtained

- **Source inspection:** all 35 cells, the three carried modules, the generator and template, `tutorials/README.md`, `docs/release-verification.md`, and the Kaggle evidence under `docs/verification/2026-09-16-kaggle-t4/`.
- **Documented execution evidence:** Kaggle serial suite v3, Tesla T4, Python 3.12.13, commit `79543f3` / blob `0be7254`, **the same blob as the reviewed revision**. Outcome: 16/16 cells **after one restart**. `executed-pass1.ipynb` records the pass-1 failure: `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.1; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` Recorded metrics: accuracy 1.0, macro-F1 1.0, majority 0.3333; `val_accuracy` was 1.0 from epoch 1 onward; reload parity passed. The Kaggle executor injects a `google.colab` upload shim.
- **Direct execution (this review):** `run_probes.py`, Windows, Python 3.12.14, NumPy 2.5.3, CPU, **no model loaded**. It covers notebook JSON parse and compile of all 16 code cells; generator `--check` (exit 0); the dataset-duplicate and split audit using the notebook's exact `synthetic_audio_dataset(24, 42)` / `split_dataset(..., 0.25, 42)` calls; and cell 17 (dataset-ZIP BYOD) executed verbatim with a `google.colab` stub on seven synthetic ZIPs, through validation and split but not through the model.
- **Not verified:** a Colab run of any revision since the 2026-09-13 inference-only notebook; whether a fresh Colab kernel also triggers the restart; any model forward pass in this review; BYOD branches through adaptation/evaluation/reload; single-WAV BYOD; learner understanding.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| Technical correctness | Strong. Pinned and digest-verified model, safe adapter load with lineage checks, bounded ZIP inspection without extraction, parity-checked generated cells. Defects: the install guard turns a fresh-runtime `Run all` into a two-pass run (AST-M1); minor prose and code mismatches (AST-m5). |
| Scientific / experimental validity | The adaptation workflow is real and the sample-sanity label is honest. The "held-out" split contains byte-identical training clips, and one class has zero within-class variation (AST-M2). The baseline is computed from evaluation labels, and metrics and score semantics are under-explained (AST-m1, AST-m2). |
| Promise fulfilment | Every listed stage executes in the documented run. The `Run all` promise in the opening cell is not met (AST-M1). "Fast and reliable" and "~10–15 s on CPU" are unsupported (AST-m5). |
| Learner experience | A clear reference script with "Look for…" hints in places. No guided layer, no learner activity, and 49k characters of carried module code that are not marked as infrastructure (AST-M3, AST-m4). |
| Spec conformance | Unmet applicable `MUST`s: RUN1/RUN10/ENV6/REL2 (AST-M1); UNC2/UNC3 and §21.1 decision rule (AST-m1); EVAL3 (AST-m2); SPL3 (AST-m2); UX12 (AST-m5); REL12, since BYOD positive/negative evidence is not recorded (AST-m6). `SHOULD` gaps: SPL10, UX5, GDL1–15, EXE2, EVAL11. |

## 3. Prioritized findings

### AST-M1 — Major: fresh-runtime `Run all` stops at the install cell and needs a manual restart

**Cell/section:** Section 1, install cell (cell 3). Generated from `_INSTALL_GUARD` in `tools/build_notebook.py` (lines 47–69).

**Observed issue:** The guard records which distributions are already imported and runs `pip install` of the pins (including `numpy==2.5.3`). If any loaded distribution changed, it raises `RuntimeError(... 'Restart the runtime, then rerun from the top.')`. On the documented fresh Kaggle T4 image, NumPy 2.0.2 and cuda-bindings were already loaded, so pass 1 failed at this cell. The executor then started a second kernel, and that pass completed 16/16. `docs/release-verification.md` and `tutorials/README.md` record this as **PASS** / "verified", and the ledger notes "1 restart after install cell". The opening cell promises that `Run all` completes the default path "without … configuration edit" and cites RUN7.

**Consequence:** Every fresh-runtime learner whose kernel preloads NumPy (or another pinned distribution) gets an error on the first `Run all` and must restart and rerun by hand. The spec states that such a notebook "is not `Run all` conformant" (§5). The release record presents it as passing.

**Evidence:** Documented execution: `docs/verification/2026-09-16-kaggle-t4/suite/dimer-nb2-ast-audio-classification/v3/evidence/executed-pass1.ipynb`, cell 3 error output, plus the `RESTART_MARK` retry loop in the suite kernel. Source inspection of cell 3. Colab behaviour: **not verified**.

**Recommended correction:** Adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. This repository's own workshop notebook (`tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb`) is the closest template. Implement it in `tools/build_notebook.py`, regenerate, re-qualify, and correct the release record so that a restart-dependent run is not reported as a `Run all` PASS.

**Acceptance check:** A fresh Colab runtime and a fresh Kaggle image each complete all code cells in **one** kernel session with `Run all` and no restart. The execution record shows a single pass and no error output in cell 3. `docs/release-verification.md` states the pass count explicitly.

**Spec:** RUN1, RUN10, ENV6, REL2 (MUST); §25.7.

### AST-M2 — Major: the "held-out" evaluation contains byte-identical training clips; one class has no within-class variation

**Cell/section:** Sections 6–10 (cells 16–25). Source: `generate_audio_clip` in `src/ast_audio_classification_pipeline/samples.py` (lines 100–113).

**Observed issue:** Only the `geophony` generator uses the RNG. `biophony` varies only by `index % 4` (and `index % 3`, which is constant within the class because class = `index % 3`). `anthrophony` varies only by `index % 3`, which is the same for every record in that class. Result: the 24-record dataset holds **8 / 4 / 1** distinct waveforms for geophony / biophony / anthrophony. With the default seed, both held-out `anthrophony` clips (`clip-011`, `clip-017`) are byte-identical to training clips. The two held-out `biophony` clips happen not to be. The notebook calls this "a 25% held-out validation set" and the outputs "held-out" metrics. The same split is also printed every epoch as `val_acc`, which reached 1.0 at epoch 1 in the documented run.

**Consequence:** Two of the six "held-out" clips measure memorisation, not generalisation. One class consists of a single repeated signal. The learner sees accuracy 1.0 / macro-F1 1.0 and has no way to know that a third of the evaluation reproduces training data. The `sample-sanity` label mitigates this but does not disclose it.

**Evidence:** Direct execution, `results.json` → `P3_dataset_duplicates` (`distinct_waveforms_per_class = {geophony: 8, biophony: 4, anthrophony: 1}`, `val_identical_to_train = 2`). The unseen clip 99 is distinct from all dataset clips (`false`).

**Recommended correction:** Give every class a seeded per-record variation (frequency jitter, phase, noise floor, envelope) in `generate_audio_clip`, so that no two records are identical. Add a cross-split duplicate check (waveform digest) to the split cell that reports and rejects overlap; the same check should protect BYOD. Describe the split as an evaluation split, distinct from any selection data.

**Acceptance check:** For the default dataset, the number of distinct waveform SHA-256 values equals the number of records (24), and the split cell reports 0 train/evaluation duplicates. A test asserts both. A BYOD ZIP containing a duplicated clip across splits is reported.

**Spec:** SPL10 (SHOULD); framework dimension 3 (scientific validity); EVAL6 wording.

### AST-M3 — Major: declared `GUIDED`, but the notebook has no learner activity or guided scaffolding

**Cell/section:** Whole notebook; opening and closing prose in `tools/notebook_template.py` and `tools/build_notebook.py`.

**Observed issue:** The learning objectives are procedural ("install…, verify…, perform…, execute…, export…"), and the learner's only action is running cells. None of the following exist: How to use this notebook, roadmap, glossary, prediction prompt, What to notice / Expected result, Check your reasoning, an optional experiment that changes one variable (UX5 / GDL10), a troubleshooting section, or a conclusion template. The interpretation section is a fixed statement, not a prompt. Some cells have a one-line "Look for…" hint (sections 1, 4, 5). Sections 6–14 have none, and no section ends with a synthesis (UX8) or a transfer prompt (UX9).

**Consequence:** A self-paced learner can finish the notebook without predicting, changing, comparing or explaining anything. The objectives in the title ("acoustic ecology tutorial") and the declared mode are not exercised. Natural experiments go unused: epochs or learning rate, unfrozen versus frozen, a harder synthetic class, or running the base AudioSet head on an adaptation clip.

**Evidence:** Source inspection; `results.json` → `P6_static.guided_markers` (all false except an incidental "prediction" match).

**Recommended correction:** Add the spec §3.5 guided layer, following the reference notebook (§25.13). At minimum: How to use + roadmap + Input → Model → Output near the top; a prediction before the pre-adaptation baseline and before fine-tuning; What to notice after sections 5, 9, 10 and 13; one bounded **Predict → Change one thing → Run → Observe → Explain** activity (for example, `epochs` 5 → 1, or a noisier `biophony` generator) with exact rerun instructions; troubleshooting covering the restart, the HF download, `google.colab` absence on Jupyter, and BYOD rejections; and an evidence-based conclusion prompt. Update `metadata.dimer.notebook_spec` when migrating.

**Acceptance check:** The notebook contains each GDL item above. The activity's control reaches the computation (a changed value changes the printed history or metrics in a rerun that follows the written instructions). Default `Run all` is unaffected with the activity untouched.

**Spec:** UX5, UX8, UX9 (SHOULD); GDL1–GDL15 (SHOULD); framework dimension 6.

### AST-m1 — Minor: adapted-head scores are called "probabilities" with no calibration caveat; the decision rule is not stated

**Cell/section:** Sections 10, 11 and Interpretation (cells 24–27, 34); `score_semantics` string in cell 25 (template line 345).

**Observed issue:** The base sigmoid head is correctly described as uncalibrated. The adapted head is described as a "softmax probability distribution" and "rank-ordered softmax probabilities", with no statement that the scores are uncalibrated (six evaluation clips cannot calibrate anything) and no statement that the label is the argmax.

**Consequence:** The unseen-clip score (0.987 in the documented run) reads as a calibrated confidence.

**Evidence:** Source inspection; `P6_static.decision_rule_stated_in_markdown = false`, `calibration_caveat_for_softmax_in_markdown = false`.

**Recommended correction:** State in sections 10 and 11 that the prediction is the argmax of softmax scores, that these are uncalibrated model scores, and who owns thresholding or calibration downstream. Change `score_semantics` to say "uncalibrated softmax scores".

**Acceptance check:** Markdown in sections 10–11 names the argmax decision rule and the absence of calibration. No adapted-head output is described as a probability without "uncalibrated".

**Spec:** UNC2, UNC3, §21.1 (MUST).

### AST-m2 — Minor: metrics, baseline and split assumptions are named but not explained

**Cell/section:** Sections 7, 8 and 10 (cells 18–25); `evaluate_classification` in `metrics.py` (line 115).

**Observed issue:** Accuracy, macro-F1, per-class metrics and the confusion matrix are printed as raw JSON without saying what each measures or how to read the matrix. The "majority baseline" is computed from the **evaluation** split's own labels (an oracle majority), not from the training distribution. On balanced data both give 1/3, but on imbalanced BYOD data they can differ. The split prose does not say that random stratified splitting assumes independent clips, and it uses "validation" for what is really the evaluation set.

**Consequence:** The learner cannot interpret the principal result. On imbalanced BYOD data, the baseline would use label information that a real trivial predictor could not have.

**Evidence:** Source inspection; `P4_baseline_source`; `P6_static.macro_f1_explained_in_markdown = false`, `random_split_independence_stated = false`.

**Recommended correction:** Add a short explanation of each metric and of the confusion-matrix axes. Fit the majority baseline on training labels and report it on evaluation labels. State the independence assumption and label the split "held-out evaluation" consistently.

**Acceptance check:** Section 10 explains accuracy, macro-F1 and the confusion matrix. For an imbalanced train/eval fixture, the baseline's predicted class equals the training majority. The split prose mentions the independence assumption.

**Spec:** EVAL3, SPL3 (MUST); EVAL11, SPL6 (SHOULD).

### AST-m3 — Minor: dataset-ZIP rejections for common archive layouts are misleading or point at an index

**Cell/section:** Section 6, cell 17 (template lines 210–230).

**Observed issue:** Cell 17 was executed with seven synthetic ZIPs. Valid flat and stereo layouts were accepted (6 records, split 3/3). A ZIP with a parent folder (`mydata/geophony/…`, which is what OS "compress folder" produces) and one with `__MACOSX/` metadata were both rejected as "unsafe or invalid member path". A 44.1 kHz clip and an 11 s clip were rejected as `record 0 …` instead of naming the member file. A missing class gave a clear message.

**Consequence:** A learner who zips a folder the usual way is told their archive is "unsafe", with no hint to zip the three class folders directly. Rate and duration errors make them search by index.

**Evidence:** Direct execution, `P5_byod_zip_branch`.

**Recommended correction:** Detect a single common parent folder and either strip it or say "zip the three class folders directly". Skip or name `__MACOSX/` and dot-files explicitly. Report the member filename (the record `id`) in sample-rate and duration errors, plus the corrective action (resample to 16 kHz; trim to ≤ 10.24 s).

**Acceptance check:** The nested-parent and `__MACOSX` fixtures are accepted or rejected with a message that names the fix. The 44.1 kHz and 11 s fixtures' messages contain the member path and the corrective action.

**Spec:** DAT19 (MUST), UX10 (SHOULD).

### AST-m4 — Minor: 49k characters of carried module code are not marked as infrastructure

**Cell/section:** Section 2, cells 4–9 (5,632 + 30,610 + 12,840 characters).

**Observed issue:** The carried modules show in full between installation and model staging. The section text says "Nothing in these cells runs a model yet", but it does not say the learner may run them without study, and the cells are not collapsed (`cellView: form` / `# @title Infrastructure: …`).

**Consequence:** The first substantive screen a learner meets is about 1,175 lines of library code, which hides the lesson.

**Evidence:** Source inspection; `P6_static.infrastructure_label_or_cellview_form = false`.

**Recommended correction:** Title the carrier cells `# @title Infrastructure: …`, set `cellView: form`, and add a line saying they can be run without study. Keep the parity hashes valid.

**Acceptance check:** Cells 5, 7 and 9 carry an Infrastructure title and form view. The generator `--check` and parity tests still pass.

**Spec:** GDL11, GDL12 (SHOULD).

### AST-m5 — Minor: unlabelled timing estimate and claims beyond the evidence

**Cell/section:** Opening (cell 0, template line 69), Section 4 (cell 12, template line 95), Interpretation (cell 34, template line 465).

**Observed issue:** "~10–15 s on CPU" is not labelled as an estimate and names no environment (the recorded local pre-flight measured 1.259 s of adaptation in a warm process). Section 4 says "The carried package verifies runtime dependency versions", but the cell only prints them; nothing compares them with `PINS`. The interpretation calls the adaptation "fast and reliable … without requiring GPU compute or external dependencies", but the notebook requires a 346 MB Hugging Face download, and six synthetic clips cannot establish reliability.

**Consequence:** Minor over-promising. A learner may assume a version check protects them when it does not.

**Evidence:** Source inspection; `docs/release-verification.md` pre-flight table.

**Recommended correction:** Label the timing as an estimate, or cite the measured figure with its environment. Either assert the installed versions against `PINS` in cell 13 or change the prose to "prints". Remove "reliable" and "without … external dependencies".

**Acceptance check:** Every timing statement names its environment or says "estimate". Cell 13's prose matches its behaviour. The interpretation makes no reliability claim.

**Spec:** UX12, ENV9 (MUST).

### AST-m6 — Minor: BYOD branches have no recorded execution evidence and depend on `google.colab`

**Cell/section:** Sections 5 and 6 (cells 15, 17).

**Observed issue:** Both BYOD branches import `google.colab.files.upload` and offer no location field. The notebook claims "Google Colab or Jupyter" support, so on Jupyter (or Kaggle without a shim) both branches raise `ImportError`. `docs/release-verification.md` says that "Optional BYOD branches were not exercised". This review exercised only cell 17's archive gate, validation and split (AST-m3), not adaptation, evaluation or reload on user data.

**Consequence:** The required BYOD capability (DAT10/DAT14) is unverified end to end and unusable outside Colab.

**Evidence:** Source inspection; release record; partial direct execution (`P5`).

**Recommended correction:** Add `BYOD_WAV_PATH` / `BYOD_ZIP_PATH` form fields that, when set, read from disk without importing `google.colab`. Record one positive and one negative BYOD run per branch, through the adaptation → evaluation → export → reload stages, on a supported runtime.

**Acceptance check:** With `BYOD_ZIP_PATH` set to a valid fixture, the notebook completes sections 6–14 on CPU without `google.colab`. The release record lists a positive and a negative BYOD run with commit, blob and runtime.

**Spec:** REL12 (MUST); EXE1, EXE2 (SHOULD); DAT14.

### Suggestions

- **AST-S1:** Export the pre-adaptation (re-headed, untrained) result to `ast_audio_classification_result.json`. It is printed but not persisted, so the outputs cannot show the before/after comparison.
- **AST-S2:** Run the reload parity check over all evaluation clips, and compare labels as well as rank-ordered scores, not only the single unseen clip.
- **AST-S3:** After migrating the notebook to 2.2, update the declared `notebook_spec` (metadata, opening cell, `tutorials/README.md`, `build_notebook.py` `NOTEBOOK_SPEC`).
- **AST-S4:** Note that CPU and CUDA float32 paths may give slightly different scores (ENV8), since the notebook runs on either.

## 4. Readiness decision

**Needs revision.** Three Major findings are open (AST-M1, AST-M2, AST-M3), and several applicable `MUST`s are unmet (RUN1/RUN10/ENV6/REL2, UNC2/UNC3/§21.1, EVAL3, SPL3, UX12, REL12). Remaining gates after the fixes:

1. A single-pass fresh `Run all` on Colab, plus the equivalent Kaggle run, recorded for the fixed blob.
2. Default dataset with 24 distinct waveforms and 0 cross-split duplicates.
3. Guided layer and one Predict → Change → Run activity.
4. Positive and negative BYOD runs recorded through the downstream stages.

## 5. Verified versus inferred

- **Verified (direct execution, CPU, no model):** notebook parse and compile; generator parity; dataset duplicates and split overlap; dataset-ZIP gate behaviour on seven fixtures.
- **Verified (documented execution):** the restart on fresh Kaggle and the 16/16 second pass for this exact blob; the recorded metrics.
- **Inferred:** that a fresh Colab kernel also preloads NumPy and hits the same guard (not run here); that the identical clips explain part of the perfect held-out score (no model run here).
- **Only Kurt can confirm:** whether the repository's "Candidate" status should be held back, given that its only clean run needed a restart.
- **Most likely to be wrong:** AST-M1's severity on Colab specifically. If a fresh Colab kernel does not preload NumPy or any other pinned distribution, the guard may not fire there, and M1 would then apply to the Kaggle path only.

---

Probe bundle: `ast_audio_classification_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`). Rerun with `python run_probes.py <repo-root>` (NumPy only).
