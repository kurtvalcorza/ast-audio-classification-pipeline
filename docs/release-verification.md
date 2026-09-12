# Release verification

`tutorials/ast_audio_classification_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.0. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile and the notebook-spec version; `metadata.dimer` declares that profile and spec `1.0`;
- the fresh-runtime bootstrap (clone by canonical URL, `DIMER_TUTORIAL_REF`, detached checkout of
  the requested revision, restart-on-stale-import guard) and the recorded `REPO_SHA` in exports;
- `MODEL_ID`/`MODEL_REVISION` are imported from the package rather than hard-coded, the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `ASTAudioClassificationPipeline.from_pretrained(weights_dir=...)`, `predict(audio, sample_rate=..., top_k=5)`), the ceiling print (`SAMPLE_RATE`,
  `MAX_AUDIO_SECONDS`, `MAX_INPUT_SECONDS`, `NUM_LABELS`), the pre-inference truncation and
  input-ceiling checks, the exports, the learner-facing statements (independent sigmoid, not a
  calibrated probability, no shipped threshold, rank-ordered scores, no metric) and the gated-off
  BYOD default listed in the validator; forbidden patterns (credential-in-URL, direct
  `from transformers import` / `ASTForAudioClassification` / `ASTFeatureExtractor` /
  `torchaudio.load(` / `torchaudio.functional` / `from huggingface_hub import` use that bypasses the
  pipeline,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also installs the pinned CPU-only `torch`/`torchaudio` wheels plus `transformers`, runs `ruff`
and the offline unit suite (`tests/test_pipeline.py`, injected runner, no weights). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab`, sets `DIMER_TUTORIAL_REF`, and chdirs to a scratch directory so the bootstrap clones the candidate |
| Local WSL harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with `DIMER_TUTORIAL_REF` set to the candidate commit and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`);
4. verify that Section 1 reports `repository_revision` equal to the candidate commit and that the
   installed core package versions equal the `pyproject.toml` pins;
5. verify every default-path stage completes:
   - fresh bootstrap from GitHub at the candidate revision;
   - synthetic 3 s, 440 Hz, amplitude-0.5 float32 sine at 16 kHz generated in code with its
     SHA-256 printed, `will_truncate: False`, `will_resample: False`, and the ceilings
     (`SAMPLE_RATE` 16000, `MAX_AUDIO_SECONDS` 10.24, `MAX_INPUT_SECONDS` 120.0, `NUM_LABELS` 527)
     surfaced;
   - pinned `MIT/ast-finetuned-audioset-10-10-0.4593` acquisition at the immutable revision
     through the package: `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports
     `['model.safetensors']` on a clean runtime (the clone carries the manifest and the two config
     files only), `verify_snapshot` returns the manifest dict, and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loads with `local_files_only=True`;
   - classification through `predict(audio, sample_rate=sample_rate, top_k=5)` with
     `activation == 'sigmoid'`, `truncated == False`, `resampled == False`, and a rank-ordered
     top-5 list; record the top label and score (the card-pass smoke on CUDA ranked `Sine wave`
     first at 0.84 — a different top label on CPU is a finding to record, not a failure by itself,
     because no expected metric is asserted);
   - no metric reported (no helper, no ground truth), stated as such;
   - `outputs/ast_audio_classification_result.json` and `outputs/ast_audio_classification_top_k.csv`
     written with repository SHA, model revision, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, torchaudio,
   Transformers, device), model identifier and immutable revision, whether the model cache was
   clean, outcome, produced outputs, and any warning or applicable `SHOULD` deviation in the table
   below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/ast_audio_classification_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/ast_audio_classification_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Default sample path | | pending — queued to the GPU lane |

## Current status

No clean-runtime execution of the notebook has been recorded yet; the run is **pending** and queued
to the GPU lane. Static validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains
**Candidate** until a reviewer confirms a recorded run against the notebook blob under review and
an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh:
the card pass executed the pipeline only from the verified local snapshot on CUDA (`returns/L3`:
CPU path not executed) and `stage_missing_files` was exercised only with an injected downloader in
the unit suite, so the clean run will be the first real execution of the notebook's staging path and
of CPU inference against the real weights; and
`torchaudio==2.11.0` is required at runtime because the Transformers feature extractor uses
`torchaudio.compliance.kaldi.fbank` when torchaudio is importable.
