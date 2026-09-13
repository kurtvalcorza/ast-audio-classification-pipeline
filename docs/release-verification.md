# Release verification

`tutorials/ast_audio_classification_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.1. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec
  `1.1`, `standalone: true` and `generated_from` (repository, generating revision, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary
  path; exactly one cell tagged `embedded_module` equal to `src/ast_audio_classification_pipeline/pipeline.py`
  after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and
  the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to
  `tools/build_notebook.py` output; the pinned-install cell with its restart-on-stale-import guard;
  `NOTEBOOK_SOURCE` recorded in the exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `ASTAudioClassificationPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`,
  `predict(audio, sample_rate=..., top_k=5)`, `evaluation_report`), the ceiling print (`SAMPLE_RATE`,
  `MIN_AUDIO_SECONDS`, `MAX_AUDIO_SECONDS`, `MAX_INPUT_SECONDS`, `NUM_LABELS`), the truncation flag read from
  the input manifest, the four exports, the learner-facing statements (independent sigmoid, not a
  calibrated probability, no shipped threshold, rank-ordered scores, always `not-measurable`) and the gated-off
  BYOD default listed in the validator; forbidden patterns (credential-in-URL, any `git clone` / `github.com` /
  repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `ASTForAudioClassification` / `ASTFeatureExtractor` /
  `torchaudio.functional.resample(` / `from huggingface_hub import` use **outside the carried module cell**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also installs the pinned CPU-only `torch`/`torchaudio` wheels plus `transformers`, runs `ruff`,
`tools/build_notebook.py --check`, and the offline unit suite (`tests/test_pipeline.py`,
`tests/test_role_helpers.py`, `tests/test_notebook_parity.py`; injected runner, no weights). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Local WSL harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines the pipeline class and both role helpers) with no import of the
     repository package;
   - synthetic 3 s, 440 Hz, amplitude-0.5 float32 sine at 16 kHz generated in code with its
     SHA-256 printed, and the ceilings
     (`SAMPLE_RATE` 16000, `MIN_AUDIO_SECONDS` 0.025, `MAX_AUDIO_SECONDS` 10.24, `MAX_INPUT_SECONDS` 120.0,
     `NUM_LABELS` 527) surfaced;
   - pinned `MIT/ast-finetuned-audioset-10-10-0.4593` acquisition at the immutable revision
     through the package: the inline `MANIFEST` is asserted against the module identity and written to
     `weights/ast-audioset/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all four
     manifest entries (`README.md`, `config.json`, `model.safetensors`, `preprocessor_config.json`) on a clean
     runtime, `verify_snapshot` returns the manifest dict, and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loads with `local_files_only=True`;
   - `validate_inputs` writes `outputs/ast_audio_classification_input_manifest.json` (verdict `accepted`,
     `will_truncate: false`, `will_resample: false`, one recorded rejection finding from the over-long probe);
   - classification through `predict(audio, sample_rate=sample_rate, top_k=5)` with
     `activation == 'sigmoid'`, `truncated == False`, `resampled == False`, and a rank-ordered
     top-5 list; record the top label and score (the card-pass smoke on CUDA ranked `Sine wave`
     first at 0.84 — a different top label on CPU is a finding to record, not a failure by itself,
     because no expected metric is asserted);
   - `evaluation_report` writes `outputs/ast_audio_classification_evaluation_report.json` with verdict
     `not-measurable` (no metric helper, no ground truth), stated as such;
   - `outputs/ast_audio_classification_result.json` and `outputs/ast_audio_classification_top_k.csv`
     written with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
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

A third fact a reviewer should weigh from the standalone pass: the carrier itself — executing the
carried module cell in a runtime that has no repository checkout — has been validated statically only
(parity PASS plus an offline carrier probe that stopped before any fetch), never run.
