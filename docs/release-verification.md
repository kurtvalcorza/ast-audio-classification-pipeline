# Release verification

`tutorials/ast_audio_classification_colab.ipynb` is a standalone `E2E` Candidate. The current revision has source checks and local CPU evidence only. The 2026-09-13 clean GPU record belongs to an older inference-only notebook and does not satisfy the execution gate for this changed workflow.

## Automatic coverage

CI and the local validator check that:

- notebook JSON parses, code cells compile, outputs and execution counts are cleared, explanatory markdown precedes code, and `metadata.dimer` declares Notebook Specification 2.0, profile `E2E`, mode `GUIDED`, standalone generation metadata, and module hashes;
- the notebook is byte-identical to `tools/build_notebook.py` output and carries repository-matching `metrics.py`, `samples.py`, and `pipeline.py`, immutable model identity, snapshot manifest, and dependency pins;
- the default path uses public pipeline APIs for base inference, dataset generation and validation, stratified splitting, re-heading, freezing, fine-tuning, evaluation, artifact export, safe reload, and adapted inference;
- both upload gates default off, prohibited direct-library and unsafe deserialization patterns stay outside learner code, and the archive path has bounded member, compressed-size, expanded-size, path, encoding, sample-rate, duration, and class checks;
- the five expected outputs are named: input manifest, evaluation report, result/provenance JSON, rank-ordered CSV, and classifier-head adapter;
- `README.md`, `STATUS.md`, this document, the tutorial registry, model card, weight documentation, and package constants agree on identity and Candidate status;
- the offline tests and lint pass.

These are source and unit checks, not supported-runtime execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab | fresh Python 3.12 runtime, CPU or CUDA | primary supported tutorial path and promotion evidence |
| Kaggle notebook executor | fresh Python 3.12 image with a minimal `google.colab` upload shim | equivalent clean-room path for the standalone notebook |
| Local Windows/WSL harness | pinned project environment, sequential or direct API execution | pre-flight only; useful for defect discovery and measurements, not promotion |

## Supported release verification procedure

Before promotion:

1. Resolve the exact commit and notebook Git blob under review. Confirm the generator, validator, lint, and full unit suite pass at that revision.
2. Start a fresh supported Python 3.12 runtime with no repository checkout and a clean model cache. Run the exact generated notebook from top to bottom without editing implementation cells. Keep `USE_BYOD = False` and `USE_BYOD_DATASET = False` for the qualifying default path.
3. Confirm the runtime pins and `NOTEBOOK_SOURCE.repository_revision` match the notebook metadata.
4. Confirm all 14 stages complete:
   - install the pinned environment;
   - execute the three carried package modules;
   - write and validate the immutable model manifest;
   - print runtime versions, device, and audio ceilings;
   - load the verified `MIT/ast-finetuned-audioset-10-10-0.4593` revision and demonstrate unchanged 527-label independent-sigmoid inference;
   - generate and validate 24 balanced `geophony`, `biophony`, and `anthrophony` records under `io.github.kurtvalcorza.dataset.audio.waveform-classification.v1`, while recording the expected rejected over-long probe;
   - create a seeded, disjoint, stratified 18/6 train/validation split;
   - install the seeded three-class head, freeze the transformer backbone, and record the pre-adaptation result and majority baseline;
   - cache frozen backbone features and run five classifier-head epochs at batch size 4 and learning rate `1e-3`;
   - evaluate the held-out split with accuracy, macro-F1, per-class metrics, confusion matrix, and baseline delta;
   - classify the newly generated unseen biophony clip with three-class softmax scores;
   - export the classifier-only `org.valcorza.ast-audio.adapter.v1` artifact;
   - load the artifact safely over a fresh pinned base instance and pass the numerical score comparison at `rtol=1e-5`, `atol=1e-6`;
   - write the complete output and provenance bundle.
5. Verify the adapter does not contain the frozen backbone and records the ordered classes, base model ID and revision, activation, artifact version, and training configuration.
6. Retain the notebook blob, source commit, executor, Python and package versions, device, clean-cache state, cell-by-cell outcome, duration, output files and hashes, measured metrics, artifact size, reload delta, warnings, and cleanup evidence. Do not retain access tokens.
7. Treat any failing default-path cell, missing output, identity mismatch, unsafe load, or failed reload comparison as a release blocker. Review the evidence before changing status.

Optional single-WAV and dataset-ZIP branches should be tested separately. Their failure does not alter what the default-path run exercised, but a known defect must be recorded and repaired before claiming those branches are supported.

## Recorded executions

### Current E2E pre-flight

| Date | Source | Executor | Path | Observations | Qualification |
|---|---|---|---|---|---|
| 2026-09-16 | local uncommitted `feat/ast-e2e-finetuning` working tree | Windows, Python 3.12, CPU, pinned local snapshot and exact dependencies preinstalled | all 16 generated notebook code cells, including carried modules, base inference, generate, split, re-head, freeze, adapt, evaluate, unseen predict, export, and fresh reload | 24 clips; 18/6 split; 5 epochs; 25 optimizer steps; 1.259 s adaptation in the warm process; held-out accuracy 1.0 and macro-F1 1.0; majority baseline 0.3333; unseen biophony predicted correctly at 0.9874; 19,103-byte artifact; reload maximum absolute score difference 0 | pre-flight only; installation was intentionally skipped, and this is not a fresh supported-runtime run |

This evidence verifies the real model path locally and guided the bounded default schedule. It does not show generalisation to field recordings and cannot support a benchmark claim.

### Historical inference-only evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Cell wall / total | Outcome |
|---|---|---|---|---|---|
| 2026-09-13 | `749fbf6b9a87bbf67394aad2e338252d29d5910b` / `59edb7e523cd419cd64e49624e793d9de205b033` | Colab CLI to a fresh Python 3.12.3 interpreter; Tesla T4 | unchanged older eight-cell inference sample; no repository checkout; empty per-model cache | 134.118 s / 138.366 s | pass for that historical blob; [retained record](verification/2026-09-13/README.md) |

The historical run used PyTorch `2.14.0+cu130`, verified every snapshot digest, and completed base AudioSet inference. It did not generate a labelled dataset, train a head, compute adapted metrics, export an adapter, or reload it, so it is not qualification evidence for the current `E2E` notebook.

## Current status

The current generated notebook remains Candidate. A fresh supported-runtime run and evidence review are still open gates.
