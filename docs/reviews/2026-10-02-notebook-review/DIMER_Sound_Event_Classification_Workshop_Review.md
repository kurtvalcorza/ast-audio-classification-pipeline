# Review: DIMER Notebook — Sound Event Classification (Notebook Review Framework v1)

Reviewed 2026-10-02. Read-only review; fixes, if any, are made separately and cite the IDs below (`SEC-…`).

## 1. Scope and evidence

### Review contract

| Field | Value |
|---|---|
| Repository | `kurtvalcorza/ast-audio-classification-pipeline` |
| Notebook | `tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` |
| Reviewed revision | `main` at **`256b732a65087eb68a019986a2b897cce2d22b76`** (equal to GitHub `commits/main` at review time); notebook blob `384d9736a4f5b895c3f6d4f09b0f81b3879424b7` |
| Requirements baseline | `NOTEBOOK_SPEC.md` **2.2** (ml-worker `origin/main` `b1cfe13`) |
| Profile / mode | `E2E` / `WORKSHOP` (declared in metadata and opening cell) |
| Intended learner | Basic Python and Colab familiarity; no prior ML experience (opening cell) |
| Prerequisites | A Colab account and a T4 GPU runtime; no token, upload or clone |
| Supported runtime | Fresh Google Colab, Linux x86_64, Tesla T4; the stages run in an isolated `uv` Python 3.12.12 environment built from a hash-pinned lock |
| Promised outcomes | Inspect waveforms/spectrograms; compare training-majority and seeded-head baselines; train only a new ten-class head on frozen AST features; select the epoch on validation; evaluate held-out recordings; reload the saved head in a fresh process; change background noise with other conditions held fixed; export an evidence bundle; optional labelled BYOD (full workflow) and unlabelled single-clip inference |
| Generator | Metadata names `build_sound_workshop.py/1` (sha256 `337c129c…`); the generator is **not in this repository** (`tutorials/README.md` documents that the notebook is generated outside it). CI checks the notebook through `tools/validate_release_assets.py` (carried-file hashes, package parity, metadata), not generator parity |

Scope: all 22 cells (12 markdown, 10 code incl. the carried-payload cell), the 13 carried files (worker `workshop.py`, `ast_reference/` modules, `requirements.txt` lock, `sample.json`, licences, manifests), the registry row in `tutorials/README.md`, the design document `docs/sound-event-classification-workshop-spec.md` and the workshop section of `docs/release-verification.md`.

### Existing execution evidence

| Evidence | Revision | Covers the reviewed blob? |
|---|---|---|
| `docs/release-verification.md`, "Sound-event classification workshop notebook": fresh Colab T4 `Run all`, 2026-09-27, **PASSED** 11/11 code cells, 233.3 s; test macro-F1 0.9749; reload parity exact; noise activity figures | `df03d82`, blob `384d9736a4f5` | **Yes** — same notebook blob. Text record only: the executed `.ipynb` is not retained in the repository (`docs/execution-evidence/` does not exist). |
| Same document: failed Colab run (MPLBACKEND) | `19a8aec`, blob `ffd263bd` | No (superseded) |
| Same document: Linux-container CPU pre-flight of the stages | pre-merge branch | No; not a supported runtime |
| Labelled BYOD and unlabelled branches | — | **Not exercised** by any recorded run |

### Journeys examined

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection (read top to bottom with the stated prerequisites) | Fails at the evaluation-interpretation and conclusion steps (SEC-M1, SEC-M2); otherwise strong |
| Clean default | Documented execution evidence (Colab T4, 2026-09-27, this blob) + CPU direct execution of model-free worker functions (P00–P03) | Passed as recorded; this review did not re-run it |
| Active learning | Documented execution evidence for the built-in noise activity; source inspection of its controls | The activity runs inside `Run all` with fixed settings; no learner-changeable parameter exists (SEC-S1). Not separately verified |
| Reuse and recovery | CPU direct execution of validation/refusal paths (P04–P07); source inspection of the BYOD and unlabelled cells | **Not verified end to end** (needs GPU). Refusals work but several messages are not actionable (SEC-m2, SEC-m5); BYOD results are not shown (SEC-M3) |

Probes: `run_probes.py` (CPU only, no model loaded) → `results.json`; source digests → `source_manifest.json`; all three are in `DIMER_Sound_Event_Classification_Workshop_Review_Probes.zip` next to this report. Environment for direct execution: Windows, Python 3.12 (`eo-notebook-test` env: numpy 2.5.3, scipy 1.18.1, torch 2.13.0+cpu). **Limitations:** no local GPU, so no stage that loads AST was executed; no learner observation; the CPU environment is not the pinned worker lock.

## 2. Separate judgments

**Technical correctness — sound, with friction.** The worker is careful: pinned and digest-verified data and environment, bounded WAV parsing before allocation (P06), source-group isolation (P02), strict-improvement earliest-tie selection with test extraction only after selection is written, a frozen-backbone hash check, `weights_only=True` reload with schema checks, and stage receipts that invalidate descendants. Metric formulas and the noise construction are correct (P01, P03: measured SNR within 1e-8 dB of nominal, one common gain, zero clipping). Defects are on the learner-facing surface: duplicated stage logs (SEC-m1), unspecific validation and stage errors (SEC-m2, SEC-m5).

**Promise fulfilment — mostly delivered.** Every default stage the opening promises executes (documented evidence). Two promises are only half delivered: "inspect errors on recordings held out from training" / "report corrected errors and new errors together" — the comparison is computed and exported but never shown (SEC-M1); and the labelled BYOD path runs the full workflow but shows the learner nothing except a ZIP link (SEC-M3).

**Learner experience — strong scaffolding, two misleading gaps.** Clear Input → System → Output framing per section, predictions before results, a toy metric example, worked hints, troubleshooting and a bounded conclusion template. But the conclusion template asks for numbers the notebook never displays (SEC-M1), and the default run always selects epoch 1 because validation saturates at macro-F1 1.0 — the notebook prints "Selected epoch: 1" next to "Actual updates: 600" without saying that the exported head holds 60 updates or that ten epochs tie (SEC-M2).

**Specification conformance (2.2).** Unmet applicable `MUST`s: **DAT17/DAT18** (BYOD data-movement statement and sensitive-data warning, SEC-m3), **DAT19** (actionable BYOD validation failures, SEC-m2), **ENV3** (principal library versions displayed: only Python and torch are, SEC-m4), **REL12** (BYOD positive/negative evidence — a known open release gate, recorded in `docs/release-verification.md`). Clean-runtime evidence (REL1) exists for this blob as a text record. Guided-layer `SHOULD`s: GDL10 is met only as a demonstration (SEC-S1); GDL6 has no glossary (SEC-S2). The maintainer's attribution review of the 233 CC0 / 165 CC BY / 2 CC Sampling+ clips is an existing release gate, not a new finding.

## 3. Findings

### Majors

**SEC-M1 — Paired errors, mistakes and the confusion pair are computed but never shown; the conclusion template cannot be completed from the notebook**
- **Cell/section:** §5 notice (cell `md-11`), §8 report cell (`code-16`), conclusion template (`md-17`).
- **Observed issue:** the worker writes `paired_errors.json` (corrected / new errors / unchanged, highest-confidence mistakes, a representative confusion pair, a fixed example), but no learner-facing cell reads it (P09: 0 cells). The template asks the learner to fill in "corrected [count] initial errors and introduced [count] new errors" and "An error worth inspecting is [actual example]"; the §5 hint says "Report corrected errors and new errors together". The §5 notice explains confusion-matrix rows, but the confusion figure is first displayed in §8 (P10: notice cell 11, figure cell 16). The design document (§7, §9 step 7) promises these comparisons in the learner flow.
- **Consequence:** the learner cannot perform the notebook's own interpretation activity without unzipping JSON; the "inspect errors" objective is not exercised. Dimensions 1, 5, 6, 8.
- **Evidence:** source inspection; probes P09, P10.
- **Recommended correction:** after the report stage, display a paired-outcome table (initial vs selected on the same test clips), the highest-confidence mistakes with true/predicted class names and score (or an explicit "no test errors" line), and the representative confusion pair by name (or "none"). Point the §5 notice at the §8 figure.
- **Acceptance check:** a learner-facing code cell reads `paired_errors.json` and displays the four transition counts, the mistakes (or the no-error message) with class names, and the confusion pair (or "none"); every bracketed slot in the conclusion template corresponds to a value printed by the notebook; the confusion-matrix explanation sits next to (or after) the cell that displays `confusion.png`.

**SEC-M2 — Saturated validation selection is presented without explanation**
- **Cell/section:** §5 train cell (`code-10`) and its notice (`md-11`).
- **Observed issue:** validation macro-F1 is 1.0 from epoch 1 to 10 in both recorded runs (Colab T4 and CPU pre-flight, `docs/release-verification.md`), so the earliest-tie rule always selects epoch 1. The cell prints "Selected epoch: 1" and "Actual updates: 600" and the maximum head change during training; it does not say that the exported head is the epoch-1 state (60 updates), nor that ten epochs tie and validation cannot distinguish them.
- **Consequence:** the learner is likely to misread what was selected (a 600-update head) or conclude that one epoch is "best"; the central "select with validation" objective produces an uninterpretable outcome as presented. Dimensions 3, 5.
- **Evidence:** documented execution evidence (both runs); source inspection (P11).
- **Recommended correction:** print the optimizer updates contained in the selected head and the number of epochs whose validation macro-F1 equals the best; when more than one epoch ties, print a sentence that validation cannot distinguish them and the earliest was kept by rule. Explain saturation in the notice.
- **Acceptance check:** the train cell's output states (a) updates in the selected head, taken from `training_history.json`, (b) the count of epochs tied at the best validation macro-F1, and (c) when that count exceeds 1, an explicit tie/saturation sentence; the §5 notice explains what a saturated validation score does and does not show.

**SEC-M3 — The labelled BYOD branch shows no results and does not explain how to place the data**
- **Cell/section:** §9 markdown (`md-17`) and BYOD cell (`code-18`).
- **Observed issue:** after `run_stage('byod', …)` the cell only displays a `FileLink` to the ZIP (P12: none of metrics, selection, reload parity or noise summary displayed). The instructions ask for "a directory with `manifest.json`" but never say how a Colab user gets a folder of WAVs into the runtime (Files pane folder upload or Drive mount).
- **Consequence:** a learner who completes the full BYOD workflow cannot see whether their head beat the baselines, which epoch was selected, whether reload passed or how noise changed predictions — the "practical route to reuse" is not observable. Dimension 8; DAT14 intent.
- **Evidence:** source inspection; P12, P13.
- **Recommended correction:** after a BYOD run, display the same tables as the default path (metrics by system and role, selected epoch and ties, reload parity, noise summary) read from the BYOD run directory, then the ZIP link; add two sentences on creating a folder in the Files pane (or mounting Drive) and setting `BYOD_PATH`.
- **Acceptance check:** the BYOD cell displays metrics, selection, reload parity and noise summary from the BYOD run's own `outputs/` (not the canonical run's); §9 explains how to place a folder in the runtime; the default path is unchanged when `RUN_BYOD = False`.

### Minors

**SEC-m1 — Every stage log is printed twice.** `run_stage` streams each line, then prints the last 7,000 characters of the log again regardless of outcome (P08; the hosted run noted the epoch log printed twice). Doubles output volume and makes the training table harder to find. *Correction:* print the tail only on failure. *Acceptance:* on success each log line appears once; on failure the tail and the error are printed.

**SEC-m2 — Validation failures do not name what failed (DAT19 `MUST`, UX10).** Messages are generic: "class support violates frozen split" (also for BYOD, where nothing is frozen, and without class/role/count), "recording group crosses roles" (no group), "BYOD needs two recording groups per class per role" (no class/role), and missing manifest fields raise raw `KeyError: 'role'` / `KeyError: 'audio'` (P04, P05). *Correction:* check required record fields explicitly and include class name, role, observed count and minimum, and group id in the messages. *Acceptance:* the four P04 cases and the P05 case produce messages that name the class and role (with counts), the crossing group id, and the missing field, with no raw `KeyError`.

**SEC-m3 — BYOD prose lacks the data-movement statement and sensitive-data warning (DAT17, DAT18 `MUST`).** §9 says "Use only recordings you may process in third-party Colab" but does not state whether recordings leave the runtime, nor warn against confidential, personal or regulated data (P13). *Correction:* add both statements. *Acceptance:* §9 states that the BYOD path reads and processes files inside the Colab runtime and sends them to no external service other than what Colab itself does, and warns not to use confidential, personal, sensitive or regulated recordings without authorisation.

**SEC-m4 — Principal library versions are not displayed (ENV3 `MUST`).** The setup cell prints Python and torch only; torchaudio and transformers versions reach only `provenance.json` inside the ZIP (P14). *Correction:* print torch, torchaudio and transformers versions and the CUDA device from the worker environment. *Acceptance:* the setup cell's check prints all three versions and the device name.

**SEC-m5 — Stale-stage errors do not say which stage to re-run.** Re-running §6 (reload) and then §8 (report) raises "missing, failed or stale prerequisite stage" (P07) without naming the stage (`activity`) or the cell. *Correction:* name the stage, its recorded status and the remedy. *Acceptance:* the P07 sequence raises a message naming `activity` and telling the learner to re-run that section's cell (and later ones) in order.

**SEC-m6 — Registry row contradicts the recorded hosted run.** `tutorials/README.md` says "pending — fresh Colab T4 qualification not yet recorded" while `docs/release-verification.md` records a passing Colab T4 run of this exact blob (P15). *Correction:* state the recorded run and its revision. *Acceptance:* the row names the recorded run's revision and does not say "not yet recorded".

### Suggestions (not required)

- **SEC-S1** — Make the noise activity learner-controllable (e.g. a form field choosing one SNR or one clip, rerun in its own output folder) so GDL10's "change one thing" is done by the learner rather than shown.
- **SEC-S2** — Add a collapsible glossary (waveform, sample rate, spectrogram, head, softmax, macro-F1, SNR, source group) per GDL6.
- **SEC-S3** — Optional display of the unchanged AudioSet head's top-5 sigmoid tags on a training clip, as the design document allows, to make §3's contrast concrete.
- **SEC-S4** — Accept `WAVE_FORMAT_EXTENSIBLE` PCM16/float32 headers, which common audio tools write; today they are refused with a message that suggests the encoding is wrong.

## 4. Readiness

**Needs revision.** Three Majors (SEC-M1–M3) are open, and four applicable `MUST`s are unmet (DAT17, DAT18, DAT19, ENV3). Remaining gates after the fixes: a fresh hosted Colab T4 `Run all` of the fixed revision covering all journeys (default, a representative labelled BYOD run with at least one rejected input — REL12 — and the unlabelled branch); the maintainer's per-clip attribution review; the promotion decision itself. Status stays Candidate.

## 5. Verified versus inferred

- **Verified by direct CPU execution:** carried-file integrity; metric formulas; split/group isolation and activity selection; noise SNR/gain/clipping; WAV refusals; the exact text of validation and stale-stage errors (P00–P07).
- **Verified by documented execution evidence (not re-run):** the default Colab T4 path and its numbers for this blob, including validation saturation.
- **Source inspection only:** SEC-M1, SEC-M2 presentation, SEC-M3, SEC-m1, SEC-m3, SEC-m4, SEC-m6.
- **Not verified:** any BYOD or unlabelled run on a GPU; learner understanding (no learner observation).

The finding I'd most expect to be wrong: **SEC-M3's severity.** The BYOD run does stream its stage logs (epoch lines) and produces a complete ZIP, so a reviewer could reasonably call the missing in-notebook results Minor friction rather than an undelivered outcome.
