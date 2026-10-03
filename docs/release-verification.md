# Release verification

`tutorials/ast_audio_classification_colab.ipynb` is a standalone `E2E` Candidate (`GUIDED`, DIMER Notebook Specification 2.2). The 2026-10-03 revision fixes the 2026-10-02 notebook review ([review](reviews/2026-10-02-notebook-review/ast_audio_classification_colab_Review.md); findings AST-M1–M3, AST-m1–m6). Its main change is that nothing is installed into the notebook kernel any more: a pinned `uv` builds an isolated, hash-locked Python 3.12.12 environment from `tutorials/requirements-colab.lock.txt`, and every stage of `tools/tutorial_stages.py` runs there in its own process. The revision has source checks, unit tests, a local CPU pre-flight and one hosted Colab T4 default-path run (one pass, no restart, 0 errors); Kaggle, BYOD and activity runs are not recorded yet. Candidate remains unchanged until hosted one-pass evidence is recorded and an explicit promotion decision is made.

## Automatic coverage

CI and the local validator check that:

- notebook JSON parses, code cells compile, outputs and execution counts are cleared, explanatory markdown precedes code, and `metadata.dimer` declares Notebook Specification 2.2, profile `E2E`, mode `GUIDED`, standalone generation metadata, the isolated environment and the carried-file hashes;
- the notebook is byte-identical to `tools/build_notebook.py` output; its carrier cell holds the package modules (`__init__.py`, `metrics.py`, `samples.py`, `pipeline.py`), the stage runner, the lock, the snapshot manifest and the licence byte for byte, and verifies each against `CARRIED_HASHES`;
- the lock pins every `pyproject.toml` pin, hashes every entry and was compiled wheel-only for manylinux x86_64; the only installer is the pinned `uv` wheel (URL, size and SHA-256), into a separate environment, with `--require-hashes --only-binary :all:`; no kernel cell imports a model library or runs `pip`;
- the learner cells run the stages in order (`weights`, `runtime`, `pretrained`, `dataset`, `baseline`, `finetune`, `evaluate`, `reload`, `bundle`, then the optional `activity`), and the runner uses the public pipeline API for staging, verification, base inference, dataset validation and splitting, re-heading, freezing, fine-tuning, evaluation, export and safe reload;
- the three gates (`USE_BYOD`, `USE_BYOD_DATASET`, `RUN_ACTIVITY`) default off, each BYOD gate has a path field, prohibited direct-library and unsafe deserialization patterns stay out of kernel and carried code, and the guided layer (orientation, predictions, checkpoints, troubleshooting, conclusion, collapsed infrastructure) is present;
- `README.md`, `STATUS.md`, this document, the tutorial registry, model card, weight documentation, and package constants agree on identity and Candidate status;
- `tests/test_tutorial_stages.py` runs every stage, in order and through files only, against a small randomly initialised AST stand-in, and `tests/test_ast_audio_classification_colab_review_fixes.py` guards each review finding's acceptance check;
- the offline tests and lint pass.

These are source and unit checks, not supported-runtime execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab | fresh Linux x86_64 runtime, T4 GPU or CPU | primary supported tutorial path and promotion evidence |
| Kaggle notebook executor | fresh Linux x86_64 image | equivalent clean-room path; BYOD branches use the path fields, so no `google.colab` shim is needed |
| Local Windows/WSL harness | the stage runner driven by the notebook's own kernel helpers, with a local interpreter standing in for the isolated environment | pre-flight only; useful for defect discovery and measurements, not promotion |

## Supported release verification procedure

Before promotion:

1. Resolve the exact commit and notebook Git blob under review. Confirm the generator, validator, lint, and full unit suite pass at that revision.
2. Start a fresh supported Linux x86_64 runtime with no repository checkout and a clean model cache. Run the exact generated notebook with **Run all**, once, without editing implementation cells and **without a restart**. Keep `USE_BYOD = False`, `USE_BYOD_DATASET = False` and `RUN_ACTIVITY = False` for the qualifying default path.
3. Confirm the execution counts run 1..N in one kernel session with no error output, and that `NOTEBOOK_SOURCE` and the isolated environment's versions match the notebook metadata and the lock. Record the pass count explicitly; a run that needed a restart is not a one-pass `Run all`.
4. Confirm every stage completes: runtime check; carried files verified; isolated environment installed; snapshot staged and verified; versions checked against the lock; pretrained inference on the tone; 24 distinct generated clips validated with 0 train/evaluation duplicates; seeded 18/6 split; re-head, freeze, untrained-head and training-majority baselines; five-epoch head fine-tune and adapter export; held-out evaluation and unseen-clip prediction in a fresh process; reload parity on all 21 scores at `rtol=1e-5`, `atol=1e-6`; provenance bundle.
5. Run the optional branches in separate fresh sessions and record each: dataset BYOD with a valid ZIP (positive) and with a ZIP the stage must refuse (negative), single-WAV BYOD, and the activity.
6. Retain the notebook blob, source commit, executor, Python and package versions, device, clean-cache state, cell-by-cell outcome, pass count, duration, output files and hashes, measured metrics, artifact size, reload delta and warnings. Do not retain access tokens.
7. Treat any failing default-path cell, a needed restart, a missing output, an identity mismatch, an unsafe load, or a failed reload comparison as a release blocker. Review the evidence before changing status.

## Recorded executions

### Isolated-environment revision (2026-10-03)

#### Hosted Colab T4 run (2026-10-03)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Code cells | Wall time | Outcome |
|---|---|---|---|---:|---:|---|
| 2026-10-03 | `16eee3985381bd770acde9e1ac2a3a0cb060b76d` / `2f694197e54a2c548108d7be11ee1a32c3914942` | Colab CLI sequential execution (`colab exec -f`), fresh VM, Tesla T4 15,360 MiB; not a browser `Run all`, so the saved notebook has no execution counts; cell order is taken from `exec.log` (cells 1..15 in sequence) | default settings only (`USE_BYOD = False`, `USE_BYOD_DATASET = False`, `RUN_ACTIVITY = False`); no repository checkout | 15/15 | 172.6 s | **one pass, no restart, 0 errors**; executed code-cell sources identical to the blob; [retained evidence](verification/2026-10-03-colab-t4/) |

Printed by the run: kernel Python 3.13.15; carrier verified 9 carried files, `NOTEBOOK_SOURCE` `d20a47c1d65e88b5890fe608fe6e0d309e37c087` (the generator stamp in `metadata.dimer.generated_from.revision`, parent of the run commit); isolated environment Python 3.12.12, torch `2.14.0+cu130`, torchaudio `2.11.0+cu130`, transformers `4.57.6`, 47 locked packages, `versions_match_lock: True`, device `cuda:0`, setup 67 s; snapshot `MIT/ast-finetuned-audioset-10-10-0.4593@f826b80d` 4/4 files verified; tone top label `Sine wave` 0.8288; 24 distinct generated clips (8 per class), over-long probe refused at 121.00 s; seeded 18/6 split, 0 train/evaluation duplicates; 3,843 trainable of 86,187,264 frozen backbone parameters; untrained head accuracy 0.0, training-majority baseline 0.3333; five epochs, train loss 0.9809 → 0.0059, evaluation accuracy 1.0 from epoch 1; 19,103-byte classifier-head adapter; held-out accuracy and macro-F1 1.0 (delta 0.6667 over majority), confusion matrix diagonal 2/2/2; unseen biophony clip ranked first at 0.9718; reload parity 21/21 scores on 7 clips, maximum difference 0.0 (`rtol=1e-5`, `atol=1e-6`), `PASSED`; bundle of 13 output files; activity skipped (optional). These are sample-sanity observations on generated clips, not field-recording or benchmark claims.

| File | SHA-256 |
|---|---|
| `ast_audio_classification_colab_16eee39_colab-cli-t4_output.ipynb` | `63ba7c6a9d57967e37d2ad73688919a10de1d1df7b13bea1335dc90ac02f0be6` |
| `run_summary.json` | `64db654c0d12511b76ce10f51316176feb5057ec3afdc83a707d39a27c32055b` |
| `exec.log` | `065222f945de6b5dd782146b3bff3f3e6f7a780f83ea2ff66de885676f871a45` |

Not exercised by this run: dataset BYOD (positive and negative ZIP), single-WAV BYOD, the evaluation-noise activity, and the equivalent Kaggle run.

#### Local CPU pre-flight

| Date | Source | Executor | Path | Observations | Qualification |
|---|---|---|---|---|---|
| 2026-10-03 | branch `review/ast_audio_classification_colab-2026-10-02` working tree (generator /3.0) | Windows, CPU only. The notebook's own carrier cell and `run_stage` helper were executed by a harness; a local uv virtual environment with CPython 3.12.12 and the CPU builds of the pinned packages (torch 2.14.0+cpu, torchaudio 2.11.0+cpu, transformers 4.57.6) stood in for the hash-locked Linux environment; real pinned weights | all learner cells in order, then the activity at 0, -10, -20 and -30 dB | runtime versions matched the lock; tone top label `Sine wave`; 24 distinct waveforms, 0 train/evaluation duplicates; untrained head accuracy 0.0, training-majority baseline 0.3333; five epochs, evaluation accuracy 1.0 from epoch 1; held-out accuracy and macro-F1 1.0; unseen biophony clip ranked first at 0.9718; reload parity 21/21 scores, maximum difference 0.0; activity: 0 dB unchanged, -10 dB and below moved all four biophony and anthrophony clips to geophony (accuracy 0.3333) | pre-flight only: not Linux, not the hash-locked install, not a hosted runtime, not promotion evidence |

### Superseded in-kernel-install revision (blob `0be7254`)

These runs executed the earlier notebook, which pip-installed the pins into the notebook kernel. On a fresh hosted image that install replaced already-imported NumPy and cuda-bindings, and the install cell stopped with `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` The executor then restarted and ran the notebook again. **Neither run is a one-pass `Run all`** (RUN1, RUN10): each took 2 passes.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall time | Outcome |
|---|---|---|---|---:|---|
| 2026-09-15 | `79543f3bb5a745493cf781fe29278e9b5524b7d3` / `0be72542ca1f99e9d60b64aac59a36727e3c6cf6` | Kaggle serial suite, `kurtvalcorza/dimer-nb2-ast-audio-classification` v3; Tesla T4 15,360 MiB; Python 3.12.13 | unchanged post-review E2E path, no repository checkout, clean Hugging Face cache, pinned install, digest-verified model download, base inference, generated dataset, split, adaptation, held-out evaluation, export, fresh reload | 225.6 s | **2 passes — not a one-pass `Run all`**: pass 1 stopped at the install cell (restart required); pass 2 in a new kernel completed 16/16 cells; five outputs preserved and hashed; [retained evidence](verification/2026-09-16-kaggle-t4/README.md) |
| 2026-09-15 | `b939aa3cd30cbe334c43f87b96eb96ed8e16180e` / `3a99508abd3b3308174a5f8346e685fca771bbf4` | Kaggle serial suite, `kurtvalcorza/dimer-nb2-ast-audio-classification` v2; Tesla T4 15,360 MiB; Python 3.12.13 | unchanged default E2E path, no repository checkout, clean Hugging Face cache, pinned install, digest-verified model download, base inference, generated dataset, split, adaptation, held-out evaluation, export, fresh reload | 204.3 s | **2 passes — not a one-pass `Run all`**: pass 1 stopped at the install cell (restart required); pass 2 completed 16/16 cells; five outputs preserved and hashed; [retained evidence](verification/2026-09-16-kaggle-t4/README.md) |

The installed notebook environment was PyTorch `2.14.0+cu130`, torchaudio `2.11.0+cu130`, and Transformers `4.57.6`. All four manifest-listed model files were downloaded from the immutable model revision and digest-verified. The six-clip synthetic held-out evaluation reported accuracy and macro-F1 `1.0`, compared with majority accuracy `0.3333`; the unseen generated biophony clip scored `0.987455`; the 19,103-byte classifier-head artifact reloaded over a fresh base instance with the same checked score. The 2026-10-02 review found that this generated dataset held only 13 distinct waveforms and that two of the six evaluation clips were identical to training clips, so part of that held-out score measured memorisation. These are execution and sample-sanity observations, not field-recording or benchmark claims.

### Superseded E2E pre-flight

| Date | Source | Executor | Path | Observations | Qualification |
|---|---|---|---|---|---|
| 2026-09-16 | local uncommitted `feat/ast-e2e-finetuning` working tree | Windows, Python 3.12, CPU, pinned local snapshot and exact dependencies preinstalled | all 16 generated notebook code cells, including carried modules, base inference, generate, split, re-head, freeze, adapt, evaluate, unseen predict, export, and fresh reload | 24 clips; 18/6 split; 5 epochs; 25 optimizer steps; 1.259 s adaptation in the warm process; held-out accuracy 1.0 and macro-F1 1.0; majority baseline 0.3333; unseen biophony predicted correctly at 0.9874; 19,103-byte artifact; reload maximum absolute score difference 0 | pre-flight only; installation was intentionally skipped, and this is not a fresh supported-runtime run |

### Historical inference-only evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Cell wall / total | Outcome |
|---|---|---|---|---|---|
| 2026-09-13 | `749fbf6b9a87bbf67394aad2e338252d29d5910b` / `59edb7e523cd419cd64e49624e793d9de205b033` | Colab CLI to a fresh Python 3.12.3 interpreter; Tesla T4 | unchanged older eight-cell inference sample; no repository checkout; empty per-model cache | 134.118 s / 138.366 s | pass for that historical blob; [retained record](verification/2026-09-13/README.md) |

The historical run used PyTorch `2.14.0+cu130`, verified every snapshot digest, and completed base AudioSet inference. It did not generate a labelled dataset, train a head, compute adapted metrics, export an adapter, or reload it, so it is not qualification evidence for the current `E2E` notebook.

## Current status

**Candidate.** A hosted Colab T4 run of the isolated-environment revision (commit `16eee39`, blob `2f69419`) is recorded: default path, 15/15 code cells in one pass with no restart and 0 errors (Colab CLI sequential execution, not a browser `Run all`). The earlier Kaggle T4 runs of blob `0be7254` needed a restart (2 passes) and do not meet the one-pass `Run all` requirement. Before promotion: the equivalent Kaggle run, recorded positive and negative dataset-BYOD runs and a single-WAV BYOD run through the downstream stages, the activity run, and an explicit promotion decision.

## Sound-event classification workshop notebook

`tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` (`E2E` / `WORKSHOP`, DIMER Notebook Specification 2.2) is a **Candidate**. It is recorded separately from `ast_audio_classification_colab.ipynb`, whose status it does not change. It carries the AST package modules, the model manifest and code licence, the pinned ESC-50 metadata and attribution (`karolpiczak/ESC-50@33c8ce9`), a 400-clip ESC-10 sample manifest, a runner and a hash-pinned dependency lock, and installs them into an isolated `uv` Python 3.12.12 environment. Its design is `docs/sound-event-classification-workshop-spec.md`.

| Check | Automatic (every pull request) | Manual (before promotion) |
|---|---|---|
| Metadata, opening declaration, no persisted outputs, every code cell plain Python | `tools/validate_release_assets.py` | — |
| Each carried file matches `CARRIED_HASHES`; carried `ast_reference/`, AST manifest and code licence equal the package; `source.json` agrees with the metadata; the carried ESC-50 metadata and licence are the files the sample manifest pins | `tools/validate_release_assets.py`, `tests/test_workshop_notebook.py` | — |
| Default `Run all` on a fresh Colab T4 runtime without a restart, with total time, peak GPU memory and disk recorded | — | **recorded 2026-09-27** (blob `384d9736a4f5`, PASSED, 233.3 s total; peak GPU memory and disk not transcribed from this run) |
| Labelled BYOD (representative real data, required by the design for qualification) and the unlabelled-recording branch | — | not yet exercised |

| Date (UTC) | Notebook source | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-03 | `92cda1c` / blob `586a465800fd` (downloaded from GitHub at the PR head and blob-verified before the session; every cell equals the PR head; executed file `execution-evidence/2026-10-03/DIMER_Sound_Event_Classification_Workshop_92cda1c_colab-cli-t4.ipynb`, sha256 `ff9169e2b969…`) | Google Colab CLI 0.7.4 on a fresh Colab **Tesla T4** session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`). Code cells ran in order in one kernel; this is not a browser Run all, and order is evidenced by the CLI's `Executing cell k/11` log (it records no execution counts) | Default path: `prepare`, `train`, `reload`, `activity` and `report` stages; BYOD and unlabelled branches left disabled | 297.4 s wall (notebook: setup 68.3 s, total 283.6 s) | **PASSED**: 11/11 code cells, no errors. Every reported figure equals the 2026-09-27 run of blob `384d9736a4f5`: 240 / 80 / 80 clips (188 / 59 / 57 recordings); test accuracy / macro-F1 majority 0.100 / 0.018, initial head 0.0875 / 0.0592, selected head 0.975 / 0.9749 (epoch 1, 600 updates; 71 test clips corrected, 0 new errors); reload parity passed (4 probes, 80 clips, max logit and probability deltas 0.0); noise activity clean 1.00, 20 dB 1.00, 10 dB 0.85 (3 changed), 0 dB 0.60 / 0.5497 (8 changed). Saved outputs inspected |
| 2026-09-27 | `df03d82` / blob `384d9736a4f5` (the executed file's 22 cells equal this blob apart from a Colab `# @title` line in cell 3; uploaded file sha256 `3af4d2c86c38…`) | Google Colab, fresh **Tesla T4** runtime (reported by cell 2) | Default `Run all`, no restart: carried files verified; isolated environment installed; `prepare`, `train`, `reload`, `activity` and `report` stages each in their own process; BYOD and unlabelled branches left disabled | 233.3 s total (setup 67.8 s) | **PASSED** — 11/11 code cells, execution counts 1–11, no errors. 240 / 80 / 80 clips (188 / 59 / 57 source recordings). Test accuracy / macro-F1: majority 0.100 / 0.018, untrained head 0.0875 / 0.0592, selected head 0.975 / 0.9749 (epoch 1 of 10; validation macro-F1 saturates at 1.0 from epoch 1; 600 updates). Reload parity passed (4 probes, 80 test clips, max logit and probability deltas 0.0, metrics identical). Noise activity (20 selected clips): clean 1.00, 20 dB 1.00, 10 dB 0.85 (3 changed), 0 dB 0.60 / 0.5497 (8 changed). Figures match the CPU pre-flight below. The epoch log is printed twice in cell 10; cosmetic, the tables are single |
| 2026-09-27 | `19a8aec` / blob `ffd263bdeea4` (the executed file's code cells equal this blob apart from a Colab `# @title` line in cell 3) | Google Colab, fresh **Tesla T4** runtime (reported by cell 2) | Default `Run all`: cells 2–4 completed (carried files verified, isolated environment installed); `prepare` failed | — | **FAILED** in `prepare` at the first figure: `ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend`. Colab's kernel exports `MPLBACKEND=module://matplotlib_inline.backend_inline`; the runner passed `os.environ` to the isolated worker, whose pinned matplotlib 3.10.8 cannot import that backend. The CPU pre-flight did not set `MPLBACKEND`, so it did not reproduce there; reproduced locally with the pinned matplotlib and that variable. Fixed: the runner cell sets `ENV['MPLBACKEND'] = 'Agg'` for the worker (the worker only writes figures to files); `tests/test_workshop_notebook.py` guards it. No carried file or hash changed |
| 2026-09-27 | This branch (carried `workshop.py` sha256 `dc7c97d7e7fd…`) | Builder pre-flight in a Linux container, CPU only (4 cores). The exact hash-pinned lock was installed with `uv` 0.12.15 into managed Python 3.12.12 (torch 2.11.0+cu130, torchaudio 2.11.0+cu130, transformers 4.57.6). The runner's GPU-required `require_gpu()`, synchronize calls and peak-memory read were patched in a copy for CPU | `prepare` (400 ESC-10 WAVs downloaded and digest-verified; 240 / 80 / 80 clips in 188 / 59 / 57 source groups), `train` (frozen-feature cache, 10 epochs, 600 optimizer updates), `reload` (new process), `activity`, `report`. The notebook's display, worked-example and report cells were then run against those outputs; the optional branches were left at their defaults (off) | prepare 162 s, train 251 s (feature cache 190 s), reload 52 s, activity 51 s, report 1 s (CPU float32) | **PASS**. Validation macro-F1 was 0.0458 at epoch 0 and 1.0000 from epoch 1 onward, so the earliest-tie rule selected **epoch 1**. Test accuracy / macro-F1: training-majority 0.100 / 0.018, seeded initial head 0.088 / 0.059, selected head **0.975 / 0.975**. Backbone SHA-256 was identical before and after training. Reload parity passed on 4 probes and all 80 test clips (max logit and probability delta 0.0, metric parity true). Noise activity on 20 validation clips (accuracy / changed predictions vs gain-matched clean): clean 1.00 / 0, 20 dB 1.00 / 0, 10 dB 0.85 / 3, 0 dB 0.60 / 8. `sound_results.zip` was exported. **Not a supported runtime and not promotion evidence**: CPU float32 differs from the T4 path, so Colab figures may differ |

On 2026-10-02 the notebook was reviewed under Notebook Review Framework v1 ([review](reviews/2026-10-02-notebook-review/DIMER_Sound_Event_Classification_Workshop_Review.md); findings SEC-M1–M3, SEC-m1–m6) and the fixes changed the notebook blob and the carried `workshop.py`. The 2026-09-27 hosted run above covers blob `384d9736a4f5` only; the fixed revision has CPU source and helper tests but no hosted run yet. Before promotion it needs a fresh Colab T4 `Run all` that also exercises a representative labelled BYOD directory, at least one rejected BYOD input, and the unlabelled branch.

Validation saturates at macro-F1 1.0 after one epoch, so epoch selection is uninformative on this sample; the test comparison is the evidence of adaptation. A fresh Colab T4 `Run all` of blob `384d9736a4f5` passed on 2026-09-27 (first row above). The notebook stays **Candidate** until the ESC-10 per-clip attribution review it names (233 CC0, 165 CC BY, 2 CC Sampling+) and a representative labelled BYOD run are recorded here.

### Notebook source layout change (2026-10-03)

The workshop's `CARRIED_FILES` literal in cell 3 was one 959,724-character line. `tools/split_workshop_carrier.py` rewrote it as parenthesised runs of short string pieces (at most 1,000 characters each), and the change is logged in `metadata.dimer.generated_from.post_generation_revisions`. Python joins the pieces back into the same text: the carried files, `CARRIED_HASHES`, the carried `source.json` and `generated_from.files` are unchanged, and no cell line is now longer than 2,000 characters (`python tools/split_workshop_carrier.py --check`). The notebook blob changes from `f1acaaf4d88e` (the 2026-10-02 review-fix revision, which has no hosted run) to `586a465800fd`. A hosted re-run of the new blob passed on 2026-10-03 through the Colab CLI (first row of the table above); its figures equal the 2026-09-27 run of blob `384d9736a4f5`. It exercised the default path only: a representative labelled BYOD run, a rejected BYOD input and the unlabelled branch are still required before promotion. Status stays **Candidate**.
