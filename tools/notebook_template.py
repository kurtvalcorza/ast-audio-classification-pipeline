"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone carrier, isolated environment).

Only the task-specific prose and the learner cells live here. The runtime check, the carrier of the package / stage
runner / lock / manifest, the isolated install and the weight staging are produced by the generator from repository
sources, so they cannot drift from the package. Every learner cell runs one stage of ``tools/tutorial_stages.py`` with
``run_stage`` and prints what that stage wrote.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

COLAB_URL = "https://colab.research.google.com/github/kurtvalcorza/ast-audio-classification-pipeline/blob/main/tutorials/ast_audio_classification_colab.ipynb"

TEMPLATE = {
    "package": "ast_audio_classification_pipeline",
    "repo_name": "ast-audio-classification-pipeline",
    "stem": "ast_audio_classification",
    "notebook_name": "ast_audio_classification_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "weights_key": "ast-audioset",
    "carried": {
        "src/ast_audio_classification_pipeline/__init__.py": "src/ast_audio_classification_pipeline/__init__.py",
        "src/ast_audio_classification_pipeline/metrics.py": "src/ast_audio_classification_pipeline/metrics.py",
        "src/ast_audio_classification_pipeline/samples.py": "src/ast_audio_classification_pipeline/samples.py",
        "src/ast_audio_classification_pipeline/pipeline.py": "src/ast_audio_classification_pipeline/pipeline.py",
        "tutorial_stages.py": "tools/tutorial_stages.py",
        "requirements.txt": "tutorials/requirements-colab.lock.txt",
        "weights/ast-audioset/dimer-base-manifest.json": "weights/ast-audioset/dimer-base-manifest.json",
        "LICENSE": "LICENSE",
    },
    "stage_runner": "tutorial_stages.py",
    "lock": "requirements.txt",
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "disk_gib": {"weights": 0.5, "environment": 10},
    "runtime_modules": ["torch", "torchaudio", "transformers"],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime checks the runtime, writes and hash-verifies the carried files, builds "
        "an isolated hash-locked Python environment (nothing is installed into the notebook kernel, so no restart is needed), "
        "stages and digest-verifies the pinned AST snapshot, demonstrates the unchanged 527-label AudioSet head, builds and "
        "validates the 24-clip acoustic-ecology dataset, refuses duplicate clips, creates a seeded stratified split with a "
        "held-out evaluation set, re-heads the classifier onto three classes, freezes the transformer backbone, measures the "
        "untrained head and a training-majority baseline, runs the bounded classifier-head fine-tune and exports the adapter, "
        "evaluates the held-out clips and an unseen clip in a fresh process, reloads the adapter in another fresh process and "
        "checks numeric parity, and writes machine-readable outputs. The default path needs no repository clone, DIMER worker "
        "or service, credential, upload dialog, or configuration edit (NOTEBOOK_SPEC 2.2 §5, RUN1, RUN7, RUN10, FT2)."
    ),
    "byod": (
        "Two optional branches are disabled by default. `USE_BYOD = True` (Section 5) scores one PCM WAV with the unchanged "
        "AudioSet head. `USE_BYOD_DATASET = True` (Section 6) replaces the generated dataset with one ZIP of three folders, "
        "`geophony/`, `biophony/` and `anthrophony/` (at the top of the archive or inside one enclosing folder), with at least two "
        "16 kHz mono or multichannel PCM WAV files per class, each 0.025–10.24 s; the ZIP then goes through the same validation, "
        "duplicate refusal, seeded split, adaptation, evaluation, export and reload stages. Each branch reads the file named in "
        "its path field (`BYOD_WAV_PATH`, `BYOD_ZIP_PATH`) on any runtime; with the field empty, Google Colab shows an upload "
        "dialog. Files stay in this runtime."
    ),
    "title": "AST Audio Spectrogram Transformer — DIMER Acoustic Ecology E2E Tutorial (Standalone)",
    "badges": [
        ("GitHub", "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white", "https://github.com/kurtvalcorza/ast-audio-classification-pipeline"),
        ("Open In Colab", "https://colab.research.google.com/assets/colab-badge.svg", COLAB_URL),
        ("Hugging Face", "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MIT%2Fast--finetuned--audioset-ffcc4d?style=flat", "https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593"),
        ("Upstream", "https://img.shields.io/badge/Upstream-YuanGongND%2Fast-181717?style=flat&logo=github&logoColor=white", "https://github.com/YuanGongND/ast"),
        ("arXiv", "https://img.shields.io/badge/arXiv-2104.01778-b31b1b.svg", "https://arxiv.org/abs/2104.01778"),
    ],
    "capability": "pretrained AudioSet classification & supervised acoustic ecology adaptation with `MIT/ast-finetuned-audioset-10-10-0.4593`",
    "intro": (
        "The Audio Spectrogram Transformer (AST) applies a Vision Transformer (ViT) architecture directly to audio "
        "spectrograms. At inference, input audio is resampled to 16 kHz, converted into a 128-bin Kaldi filterbank with "
        "a 10 ms hop, padded or cropped to 1024 frames (10.24 s), and processed by the transformer backbone.\n\n"
        "**Supervised adaptation in this notebook:** besides pretrained 527-class AudioSet event inference, the notebook adapts "
        "the model to a three-class acoustic ecology task (`ADAPT_CLASSES = ('geophony', 'biophony', 'anthrophony')`). The "
        "backbone is frozen and the classifier head is re-headed (`Linear(768, 3)` behind the head's LayerNorm), so only "
        "3,843 parameters are trained, with bounded AdamW on cached backbone features. The training itself is short: the "
        "2026-09-16 local pre-flight measured 1.3 s for the five epochs in a warm CPU process (Windows, Python 3.12). On a "
        "fresh hosted runtime, building the isolated environment and downloading the model take longer — a few minutes is an "
        "estimate that depends on the network. Evaluation against a training-majority baseline and a portable adapter export "
        "(`ast-audio-adapter-v1.pt`) complete the end-to-end workflow."
    ),
    "learning_objectives": (
        "after this notebook you should be able to (1) explain how AST turns a waveform into a spectrogram and why the "
        "pretrained AudioSet head's sigmoid scores are not calibrated probabilities; (2) explain why a held-out evaluation needs "
        "distinct, independent clips, and check a dataset for duplicates before splitting; (3) adapt a pretrained audio model "
        "to new classes by re-heading it and training only the head on a frozen backbone; (4) read accuracy, macro-F1, per-class "
        "scores and a confusion matrix, and compare them with a baseline fitted on training labels; (5) export the adapted head, "
        "reload it in a new process and verify that it reproduces the trained model's scores; and (6) change one variable "
        "(evaluation noise) and explain what the change does to the metrics."
    ),
    "exclusions": (
        "speech transcription, speaker identification, temporal localisation of events inside the window, source "
        "separation, calibrated confidence, or unconstrained multi-hour training. Adaptation is bounded to the classifier head."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh Linux x86_64 runtime — Google Colab, Kaggle, or a Linux Jupyter server — with about 11 GB of free disk. A GPU is used automatically when present (the default runtime type is a T4) but is not required: every stage also runs on CPU, in float32 on both. Windows and macOS kernels are not supported, because the hash-locked environment is built for manylinux x86_64.",
        "- **Knowledge:** basic Python; what a classifier and a train/test split are. Spectrograms, re-heading and the metrics are explained where they are first used; the glossary collects the terms.",
        "- **Data:** the default path is self-contained and generates its audio in code. Optional single-clip BYOD accepts one PCM WAV. Optional dataset BYOD accepts one ZIP of `geophony/`, `biophony/` and `anthrophony/` folders (at the top level or inside one enclosing folder; `__MACOSX/` and dot-files are skipped) with at least two 16 kHz PCM WAV files per class; clips must be 0.025–10.24 s and no clip may appear twice. The ZIP is capped at 100 files, 64 MiB compressed, and 256 MiB expanded, and it is read in memory, never extracted. Do not upload confidential or restricted audio to a hosted runtime unless authorized.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can run cells in a hosted notebook and read short Python, and who want "
                "to see how a pretrained audio model is evaluated and adapted to new classes. No prior experience with audio "
                "models is assumed: each term is explained where it is first needed, and the glossary below collects them.\n\n"
                "**Running it.** In Colab, choose *Runtime → Run all*. The default path needs no edit, no upload, no account, no "
                "token and no runtime restart. Sections 1–3 build an isolated environment from hash-locked packages and download "
                "about 346 MB of verified weights, so they take the longest before any model runs; read ahead while they finish, "
                "or run one cell at a time with *Shift + Enter*.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files in the run directory — the verified snapshot, the "
                "dataset, the adapter and JSON records.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–14) are the machine-learning workflow; each runs one stage or "
                "reads one record and prints compact dictionaries for you to read. *Infrastructure cells* (Sections 1–3: the "
                "runtime check, the carried code, the isolated install and the pinned-weight staging) are collapsed and titled "
                "**Infrastructure**. You may run them without studying their implementation: they exist for reproducibility and "
                "provenance, not as prerequisite machine-learning knowledge.\n\n"
                "**Form controls.** Some learner cells start with fields that Colab renders as a form: `USE_BYOD` and "
                "`BYOD_WAV_PATH` (Section 5); `USE_BYOD_DATASET` and `BYOD_ZIP_PATH` (Section 6); and `RUN_ACTIVITY` and "
                "`ACTIVITY_SNR_DB` in the optional activity (Section 14). Section 9 also sets `EPOCHS`, `BATCH_SIZE` and "
                "`LEARNING_RATE`. Leave them at their defaults for the first run: the notes and sample answers describe the "
                "default path.\n\n"
                "**Section tags.** Each numbered heading carries one tag. **[Concept]** — what the model does and why. "
                "**[Evaluation practice]** — how the evidence is produced and how to read it. **[Engineering]** — "
                "reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to "
                "an expectation; after it, **What to notice** describes normal output, and a collapsed **Check your reasoning** "
                "answer follows each checkpoint. Write your own answer first, then open it. Exact numbers can vary between CPU "
                "and GPU and between library builds, so the notes describe the shape of a normal result rather than fixed values."
            ),
            (
                "## The task: Input → Model/System → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Pretrained inference** | one 16 kHz waveform | filterbank spectrogram → AST backbone → the original 527-class AudioSet head, sigmoid per class | five AudioSet labels with independent scores |\n"
                "| **Adaptation** | 18 labelled training clips | frozen backbone features → a new 3-class head trained with AdamW and cross-entropy | a small adapter file holding only the head |\n"
                "| **Evaluation** | 6 held-out labelled clips and one unseen clip | the frozen backbone plus the reloaded head, softmax over three classes, argmax decision | accuracy, macro-F1, per-class scores, a confusion matrix, and a comparison with a baseline |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux and disk checked; a fresh run directory | the GPU (or CPU) |\n"
                "| 2. Carry the code, install the runtime | [Engineering] | carried files verified; an isolated hash-locked environment | versions |\n"
                "| 3. Pin, stage and verify | [Engineering] | the snapshot downloaded and digest-checked | file count |\n"
                "| 4. Confirm the runtime | [Engineering] | versions checked against the lock | device, ceilings |\n"
                "| 5. The pretrained head | [Concept] | AudioSet labels for a 440 Hz tone | top-5 labels |\n"
                "| 6. The dataset | [Evaluation practice] | 24 clips built, validated, duplicates refused | distinct waveforms |\n"
                "| 7. The split | [Evaluation practice] | 18 train / 6 held-out evaluation | counts, duplicates = 0 |\n"
                "| 8. Re-head and baselines | [Concept] | new head, frozen backbone, two baselines | the score to beat |\n"
                "| 9. Fine-tune and export | [Concept] | five epochs on the head; adapter saved | the epoch history |\n"
                "| 10. Held-out evaluation | [Evaluation practice] | the adapter scored in a fresh process | the principal result |\n"
                "| 11. An unseen clip | [Evaluation practice] | one new clip, three scores | the decision rule |\n"
                "| 12. Reload parity | [Engineering] | another fresh process reproduces every score | `PASSED` |\n"
                "| 13. Provenance bundle | [Engineering] | one JSON linking every record | the file list |\n"
                "| 14. Optional activity | [Concept] | add noise to the evaluation clips (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | common failures and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n\n"
                "**Fast path.** Short on time? Run all, then read Sections 8, 10 and 12 and the conclusion: they carry the "
                "principal results. The canonical path ends with Section 13; Section 14 changes nothing unless you switch it on."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Waveform** | The audio as a sequence of numbers between -1 and 1, 16,000 per second (16 kHz). |\n"
                "| **Spectrogram / filterbank** | A time × frequency picture of the sound: 128 frequency bands every 10 ms. AST reads this picture. |\n"
                "| **Backbone** | The transformer that turns the spectrogram into one 768-number summary (a *feature*). Frozen here. |\n"
                "| **Head** | The small layer that turns the feature into one score per class. Replaced and trained here. |\n"
                "| **Re-heading** | Swapping the 527-class AudioSet head for a new, randomly initialised 3-class head. |\n"
                "| **Freezing** | Marking the backbone's parameters as not trainable, so only the head changes. |\n"
                "| **Sigmoid score** | The pretrained head's per-label score in 0–1; labels are scored independently and the scores do not sum to one. |\n"
                "| **Softmax score** | The adapted head's scores; they are positive and sum to one over the three classes, but they are not calibrated probabilities. |\n"
                "| **Argmax** | The decision rule: the class with the highest score is the prediction. |\n"
                "| **Calibration** | Whether a score of 0.9 is right about 90% of the time. Nothing here calibrates the scores. |\n"
                "| **Epoch** | One pass over the 18 training clips. |\n"
                "| **Held-out evaluation set** | Clips kept out of training and used only to measure the result. No separate validation set is used to choose anything. |\n"
                "| **Accuracy** | The share of evaluation clips whose predicted class is correct. |\n"
                "| **Macro-F1** | The F1 score (balance of precision and recall) computed per class, then averaged with equal weight per class. |\n"
                "| **Confusion matrix** | A table of counts: rows are the true class, columns the predicted class; the diagonal holds the correct predictions. |\n"
                "| **Majority baseline** | A trivial rule that always predicts the most common class in the training split. |\n"
                "| **SNR (dB)** | Signal-to-noise ratio: how much louder the clip is than the added noise; 0 dB means equally loud. |\n"
                "| **Adapter** | The exported file holding only the trained head, class names and the base model's identity. |\n"
                "| **Digest (SHA-256)** | A fingerprint of a file's bytes or a waveform's samples; a single changed value changes it. |\n"
                "| **Hash-locked environment** | A separate Python environment built from a requirements file that pins every package to one version and one set of SHA-256 digests; the installer refuses anything else. |\n"
                "| **Stage** | One step of the workflow run as its own process by `run_stage`; it reads the files earlier stages wrote and writes its own. |\n"
                "| **BYOD** | Bring Your Own Data: optional switches to run the same workflow on your own audio. |\n\n"
                "</details>"
            ),
        ],
    },
    "setup": [
        {
            "cell": "check",
            "md": (
                "## 1. Check the runtime · [Engineering]\n\n"
                "> **Infrastructure.** The code cells in Sections 1–3 are collapsed. You may run them without studying their "
                "implementation; they exist for reproducibility and provenance. The learning activities start in Section 4.\n\n"
                "**Input:** a fresh hosted runtime. **System:** checks that it is Linux x86_64 with enough free disk, reports the "
                "GPU if there is one, and creates a new run directory. **Output:** the directories this run will use. Each run "
                "writes to a new directory under `outputs/{stem}/`, so an earlier export cannot be mistaken for a current result. "
                "The verified snapshot is kept in `weights/` and reused by a later run."
            ),
            "after": (
                "**Expected result:** one dictionary naming the GPU (for example `Tesla T4, 15360 MiB`) or `none (the stages run "
                "on CPU)`, the kernel's Python version, the run directory, the weights directory, the isolated environment's "
                "directory and the free disk. If the cell stops with a platform or disk message, see **Troubleshooting**."
            ),
        },
        {
            "cell": "carrier",
            "md": (
                "## 2. Carry the code and install the locked runtime · [Engineering]\n\n"
                "> **Infrastructure.** The next two code cells are collapsed. The first **is** the repository's code, carried so "
                "that this notebook works on its own; the second builds the environment every stage runs in.\n\n"
                "The first cell holds, as text, the files the workflow needs: the package's four modules under "
                "`src/ast_audio_classification_pipeline/` (identity constants, snapshot verification and staging, validation, the "
                "pipeline class, the generated sample data and the metrics), the stage runner `tutorial_stages.py`, the hash-locked "
                "`requirements.txt` ({n_locked} packages), the snapshot manifest and the licence. It writes each file into the run "
                "directory and checks its SHA-256 against `CARRIED_HASHES`, stopping on any mismatch. The text is the repository's "
                "files byte for byte; the repository's parity test (`tests/test_notebook_parity.py`) fails whenever the two "
                "diverge, so what runs here is what the repository tests. Nothing in this cell runs a model."
            ),
            "after": (
                "**Expected result:** `carried_files`, `verified: True`, and the repository revision the notebook was generated "
                "from.\n\n"
                "The next cell installs nothing into this notebook's kernel. It downloads one pinned file — the `uv` installer "
                "wheel, refused unless its size and SHA-256 match — creates a separate virtual environment with its own CPython "
                "3.12.12, and installs `requirements.txt` into it with `--require-hashes --only-binary :all:`: every package must "
                "be the locked version, a prebuilt wheel, and match a locked digest. The hosted runtime's own packages are never "
                "replaced, which is why no restart is needed. The cell also defines `run_stage`, `load_record` and "
                "`obtain_upload`, the helpers the learner cells use."
            ),
        },
        {
            "cell": "install",
            "md": (
                "**Infrastructure: the isolated environment.** Installation messages from `uv` are normal and can take a few "
                "minutes. A failed download or a hash mismatch stops the cell; never remove a pin or a hash to get past one."
            ),
            "after": (
                "**Expected result:** one dictionary with the generating revision, the isolated environment's Python (3.12.12), "
                "the `torch`, `torchaudio` and `transformers` versions, `cuda` (`True` on a GPU runtime, `False` on CPU — both "
                "are supported), the number of locked packages and the setup time."
            ),
        },
        {
            "cell": "weights",
            "md": (
                "## 3. Pin, stage and verify the model · [Engineering]\n\n"
                "> **Infrastructure.** The next code cell is collapsed. It downloads about 346 MB of pinned weights and checks every "
                "file's size and SHA-256; you may run it without studying its implementation.\n\n"
                "The model identity is carried twice — `MODEL_ID`/`MODEL_REVISION` in the carried `pipeline.py` and the snapshot "
                "manifest (paths, byte sizes, SHA-256) — and the `weights` stage first checks that they agree. It installs the "
                "carried manifest into `weights/`, fetches exactly the files that are absent from the Hugging Face Hub **at the "
                "pinned revision** of `{MODEL_ID}` (never `main`), and re-hashes every file, raising on the first size or digest "
                "mismatch. There is no fallback to a different download, and no remote model code is executed. Every later stage "
                "verifies the snapshot again before loading it."
            ),
            "after": (
                "**What to notice:** the model id, revision, licence and file count; a `fetched` list of the files downloaded on "
                "this run (empty on a rerun, because staging only fetches files that are absent); and a count of verified files. "
                "A size or SHA-256 mismatch stops the cell with an error naming the file — see **Troubleshooting**, and never edit "
                "a manifest to get past one."
            ),
        },
    ],
    "cells": [
        # ------------------------------------------------------------------ 4. runtime
        {
            "md": (
                "## 4. Confirm the isolated runtime · [Engineering]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage` (or reads a record a stage "
                "wrote). This cell runs the `runtime` stage. It compares the `torch`, `torchaudio` and `transformers` versions "
                "installed in the isolated environment with the versions in the carried lock and **stops if any differs**; it "
                "then prints the device the stages will use and the input ceilings the pipeline enforces.\n\n"
                "**Expected result:** `versions_match_lock: True`, the device (`cuda:0` on a GPU runtime, `cpu` otherwise) and "
                "the ceilings: 16 kHz input, at least 0.025 s, a 10.24 s model window, at most 120 s of input, and 527 AudioSet labels."
            ),
            "code": "run_stage('runtime')",
        },
        # ------------------------------------------------------------------ 5. pretrained
        {
            "md": (
                "## 5. The pretrained AudioSet head · [Concept]\n\n"
                "Before adapting anything, the `pretrained` stage runs the unchanged model on a 3-second 440 Hz sine tone (or, "
                "with `USE_BYOD = True`, on your own PCM WAV). `predict` turns the clip into a spectrogram, runs the backbone and "
                "the original 527-class AudioSet head, and applies an independent **sigmoid** to every label. Each score is **not "
                "a calibrated probability**, the 527 scores do not sum to one, and no decision threshold is shipped, so the list "
                "is a ranking, not a verdict.\n\n"
                "**BYOD:** set `USE_BYOD = True` and either put a WAV path in `BYOD_WAV_PATH` (any runtime) or leave it empty to "
                "get the Colab upload dialog. A rate other than 16 kHz is resampled; clips longer than 10.24 s are cropped to the "
                "model window and reported as `truncated`.\n\n"
                "**Predict before running:** a pure 440 Hz tone is the note A. Which kinds of AudioSet labels do you expect near "
                "the top — speech, music, environmental sounds? Write one guess down."
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_WAV_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "pretrained_options = []\n"
                "if USE_BYOD:\n"
                "    pretrained_options = ['--wav', obtain_upload(BYOD_WAV_PATH, '.wav', 'BYOD_WAV_PATH')]\n"
                "run_stage('pretrained', *pretrained_options)"
            ),
        },
        {
            "md": (
                "**What to notice:** `sample_kind`, the clip's sample count and digest, `activation: sigmoid`, and five ranked "
                "labels with their scores. Several unrelated labels can all have moderate scores at once, because each label is "
                "scored on its own.\n\n"
                "**Checkpoint:** the top label for the tone has a score of, say, 0.4. Does that mean the model is 40% sure? Could "
                "two labels both score above 0.5?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "No. A sigmoid score is a per-label output of the training objective, not a calibrated probability: nothing "
                "here measured how often a score of 0.4 is right. And yes — labels are scored independently, so several can be "
                "high at once (AudioSet clips often contain several sounds), and the scores do not sum to one. A tone usually "
                "draws musical or tonal labels (sine wave, music, tuning fork and similar), but the list is a ranking for "
                "inspection; this notebook makes no claim that the base model recognises any particular sound.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 6. dataset
        {
            "md": (
                "## 6. Build and validate the acoustic-ecology dataset · [Evaluation practice]\n\n"
                "The adaptation task is the Krause/Pijanowski acoustic ecology split of a soundscape into three sources: "
                "**geophony** (non-biological natural sound — rain, wind, water), **biophony** (animal sound — bird song, insects) "
                "and **anthrophony** (human-made sound — engines, machinery). The `dataset` stage builds 24 balanced 3-second "
                "16 kHz clips with the package's generator (`synthetic_audio_dataset`): filtered noise for geophony, frequency-"
                "modulated chirps for biophony and a harmonic drone for anthrophony. The generator alone repeats itself — "
                "anthrophony would be one waveform eight times — so the stage gives **every clip its own seeded variation** "
                "(`vary_waveform`): a speed change that shifts every frequency by up to ±8%, a time shift, a gain and a quiet "
                "noise floor. It then validates every record (`validate_dataset`: sample rate, mono float32, finite samples in "
                "[-1, 1], duration, labels), and **refuses any waveform that occurs twice**, naming the copies, because a copy "
                "could otherwise end up both in training and in evaluation. The records follow the owner-namespaced "
                "`io.github.kurtvalcorza.dataset.audio.waveform-classification.v1` representation.\n\n"
                "**BYOD:** set `USE_BYOD_DATASET = True` and put a ZIP path in `BYOD_ZIP_PATH`, or leave the path empty for the "
                "Colab upload dialog. The ZIP must hold the three class folders, at the top level or inside one enclosing folder "
                "(the folder name is stripped and reported); `__MACOSX/` and dot-files are skipped and listed. Every refusal "
                "names the file and what to do: resample to 16 kHz, trim to 10.24 s, remove a duplicate, add clips to a class.\n\n"
                "**Expected result:** `records: 24`, `distinct_waveforms: 24`, eight clips per class, `verdict: accepted`, and a "
                "refusal probe showing that a 121-second input is rejected before any model runs. The input manifest is written "
                "to `outputs/{stem}_input_manifest.json` in the run directory."
            ),
            "code": (
                "USE_BYOD_DATASET = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_ZIP_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "dataset_options = []\n"
                "if USE_BYOD_DATASET:\n"
                "    dataset_options = ['--zip', obtain_upload(BYOD_ZIP_PATH, '.zip', 'BYOD_ZIP_PATH')]\n"
                "run_stage('dataset', *dataset_options)"
            ),
        },
        {
            "md": (
                "**What to notice:** `distinct_waveforms` equals `records`. The generated clips are synthetic stand-ins with "
                "clear acoustic signatures, so they test the workflow, not field performance.\n\n"
                "**Checkpoint:** why check for identical waveforms *before* splitting, rather than trusting that 24 records "
                "with 24 different ids are 24 different clips?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Ids are labels on records, not evidence about their content. Before this check was added, the generator "
                "produced 24 ids but only 13 distinct waveforms (8 geophony, 4 biophony, 1 anthrophony), and two of the six "
                "evaluation clips were byte-identical to training clips. A model can score perfectly on such a clip by "
                "remembering it, so part of the held-out result would have measured memorisation. Checking digests before the "
                "split guarantees that no clip can be on both sides; the next section confirms it.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 7. split
        {
            "md": (
                "## 7. The held-out evaluation split · [Evaluation practice]\n\n"
                "The `dataset` stage also split the records with `split_dataset(..., val_fraction=0.25, seed=42)`: per class, "
                "a seeded shuffle puts 25% of the clips into a **held-out evaluation set** and the rest into training — 18 "
                "training clips (6 per class) and 6 evaluation clips (2 per class). Stratifying keeps the class balance "
                "identical on both sides. The evaluation clips are used only to measure the result; nothing is tuned or "
                "selected on them, and no separate validation set exists. A random split like this **assumes the clips are "
                "independent** — no two clips cut from the same recording or source. That holds for the generated data; for "
                "your own recordings, put clips from one recording on the same side.\n\n"
                "**Expected result:** 18 / 6 clips, 6 / 2 per class, and `train_evaluation_duplicates: 0`."
            ),
            "code": (
                "split = load_record('{stem}_split.json')\n"
                "print({{key: split[key] for key in ('split', 'train_clips', 'evaluation_clips', 'counts', 'train_evaluation_duplicates')}})\n"
                "print('evaluation clips:', split['eval_ids'])"
            ),
        },
        {
            "md": (
                "**Checkpoint:** with only two evaluation clips per class, how much does one wrong prediction move the accuracy?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "One clip out of six is about 0.17 of accuracy, and it halves that class's recall. Six clips are enough to "
                "check that the workflow behaves sensibly — a *sample-sanity* check — but far too few to estimate how well the "
                "model would do on new recordings, which is why the reports below say `sample-sanity`, not a benchmark result.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 8. baseline
        {
            "md": (
                "## 8. Re-head, freeze, and two baselines · [Concept]\n\n"
                "`pipe.rehead(ADAPT_CLASSES, seed=42)` replaces the 527-class AudioSet head with a new, randomly initialised "
                "3-class head (`Linear(768, 3)`, seed 42), and `pipe.freeze_backbone()` freezes the 85.5 M backbone parameters, "
                "so only the head — 3,843 parameters — can change. Before any training, the `baseline` stage scores two trivial "
                "references on the evaluation clips: the **untrained head** (random weights) and the **majority baseline**, a "
                "rule fitted on the *training* labels that always predicts the training split's most common class. It does not "
                "look at the evaluation labels, so it is a predictor you could actually build; with balanced classes it scores "
                "one third.\n\n"
                "**Predict before running:** roughly what accuracy do you expect from the untrained head on three balanced "
                "classes?"
            ),
            "code": "run_stage('baseline')",
        },
        {
            "md": (
                "**What to notice:** the frozen and trainable parameter counts, the untrained head's accuracy and macro-F1, and "
                "the majority rule's class and accuracy. These are the numbers the fine-tuned head has to beat.\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Around one third — chance level for three balanced classes — though with six clips it can easily land at 0, "
                "0.17, 0.33 or 0.5. A random head often predicts one class for most clips, so its macro-F1 can be far lower than "
                "its accuracy. The majority rule scores exactly 2/6 = 0.3333 here because every class has two evaluation clips.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 9. finetune
        {
            "md": (
                "## 9. Fine-tune the head and export the adapter · [Concept]\n\n"
                "The `finetune` stage re-heads and freezes a fresh model exactly as in Section 8, runs every training and "
                "evaluation clip through the frozen backbone **once** and caches the 768-number features, then trains only the "
                "head with `torch.optim.AdamW` and cross-entropy for `EPOCHS = 5` epochs at `BATCH_SIZE = 4` and `LEARNING_RATE "
                "= 1e-3` (5 optimizer steps per epoch, 25 in total). After each epoch it prints the training loss and the "
                "accuracy on the evaluation clips, for information only: nothing is chosen from those numbers. It then records "
                "the trained model's scores for later comparison and saves the head as `outputs/ast-audio-adapter-v1.pt`: the "
                "trained weights, the class names, the training settings and the exact base-model identity. The 85 M-parameter "
                "backbone is not duplicated in this file; it is rebuilt from the pinned base revision when the adapter is loaded.\n\n"
                "**Predict before running:** will the training loss fall every epoch? How many epochs will the head need to "
                "classify all six evaluation clips correctly?"
            ),
            "code": (
                "EPOCHS = 5\n"
                "BATCH_SIZE = 4\n"
                "LEARNING_RATE = 1e-3\n"
                "run_stage('finetune', '--epochs', EPOCHS, '--batch-size', BATCH_SIZE, '--learning-rate', LEARNING_RATE)"
            ),
        },
        {
            "md": (
                "**What to notice:** five epoch lines with the training loss, the evaluation accuracy and `steps=5`, then the "
                "adapter's size (about 19 KB) and digest. The notebook does not promise a monotonic loss or any quality "
                "threshold on this sample.\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "The loss usually falls, but with mini-batches of four it can rise for an epoch. On these clearly separable "
                "synthetic classes the head often classifies all evaluation clips correctly within one or two epochs — which is "
                "exactly why the evaluation accuracy is printed for information only: if it were used to pick an epoch, it "
                "would no longer be a held-out measurement.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 10. evaluate
        {
            "md": (
                "## 10. Held-out evaluation in a fresh process · [Evaluation practice]\n\n"
                "The `evaluate` stage starts a new process, rebuilds the base model from the verified snapshot, loads the adapter "
                "with `torch.load(..., weights_only=True)`, and scores the six evaluation clips. For each clip the adapted head "
                "produces three **softmax scores**: positive numbers that sum to one. The **decision rule is argmax** — the class "
                "with the highest score is the prediction; there is no threshold and no \"unsure\" answer. These scores are "
                "**uncalibrated**: six clips cannot show how often a score of 0.9 is right, so do not read them as "
                "probabilities. Calibration and any threshold are the responsibility of whoever uses the model, on their own "
                "data.\n\n"
                "How to read the report:\n\n"
                "- **Accuracy** is the share of evaluation clips predicted correctly.\n"
                "- **Per class**, *precision* is the share of clips predicted as that class that really are it, *recall* the "
                "share of that class's clips found, *F1* their harmonic mean, and *support* the number of clips of that class.\n"
                "- **Macro-F1** averages the three F1 scores with equal weight, so a class the model ignores pulls it down "
                "even when accuracy looks good.\n"
                "- In the **confusion matrix**, rows are the true class and columns the predicted class, in the order printed; "
                "the diagonal counts correct predictions and every off-diagonal count is a specific mistake.\n"
                "- The **baseline** is the training-majority rule from Section 8; `delta_vs_baseline` is the accuracy gain over it.\n\n"
                "The machine-readable report is written to `outputs/{stem}_evaluation_report.json` with verdict `sample-sanity`.\n\n"
                "**Predict before running:** will the adapted head beat the majority baseline? On which class, if any, do you "
                "expect mistakes?"
            ),
            "code": "run_stage('evaluate')",
        },
        {
            "md": (
                "**What to notice:** accuracy and macro-F1 against the baseline's 0.3333, the per-class lines, and where any "
                "off-diagonal counts fall in the confusion matrix.\n\n"
                "**Question tested:** on clips that played no part in training, does the adapted head separate the three "
                "sources better than a rule that ignores the audio?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "A perfect or near-perfect score is the expected outcome here: the generated classes differ in obvious ways "
                "(broadband noise, chirps, a low drone), and the evaluation clips are distinct from the training clips. That "
                "shows the adaptation workflow works end to end and that the head learned *these* signatures. It does not show "
                "that the model would separate real field recordings, where sources overlap, vary and are recorded on different "
                "equipment.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 11. unseen
        {
            "md": (
                "## 11. An unseen clip · [Evaluation practice]\n\n"
                "The `evaluate` stage also scored one newly generated biophony clip (`unseen_synthetic_biophony_99`) that is not "
                "in the dataset — its digest is checked against all 24 records. The three uncalibrated softmax scores are "
                "exported, rank-ordered, to `outputs/{stem}_top_k.csv`. The prediction is the argmax; the top score is a ranking "
                "margin, not a confidence.\n\n"
                "**Expected result:** `in_dataset: False`, the true class `biophony`, and three scores that sum to one."
            ),
            "code": (
                "unseen = load_record('{stem}_unseen_prediction.json')\n"
                "print({{'clip': unseen['clip'], 'true_class': unseen['true_class'], 'in_dataset': unseen['in_dataset'], 'decision_rule': 'argmax'}})\n"
                "for rank, item in enumerate(unseen['result']['predictions'], start=1):\n"
                "    print(f\"  {{rank}}. {{item['label']:<12}} uncalibrated softmax score {{item['score']:.4f}}\")"
            ),
        },
        {
            "md": (
                "**Checkpoint:** the top score is close to 1. Would you report \"the model is 99% confident this is biophony\"?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "No. Softmax always produces scores that sum to one, and a head trained to separate three easy synthetic "
                "classes tends to push one score close to 1 whatever the input — even for a sound that belongs to none of the "
                "classes, because argmax over a closed set has no \"other\" answer. The honest statement is \"the adapted head "
                "ranks biophony first\"; how often such a ranking is right would have to be measured on real, labelled data.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 12. reload
        {
            "md": (
                "## 12. Fresh reload and numeric parity · [Engineering]\n\n"
                "Section 10 already loaded the adapter in a new process. The `reload` stage checks that this is faithful: it "
                "checks the adapter file's digest, rebuilds the pipeline from it in another fresh process, scores every "
                "evaluation clip and the unseen clip again, and compares all 21 scores with the ones the trained model produced "
                "in Section 9 (`rtol=1e-5`, `atol=1e-6`), together with the predicted labels. A mismatch stops the notebook.\n\n"
                "**Expected result:** `reload_verification: PASSED`, `predicted_labels_identical: True` and a maximum absolute "
                "score difference at or near 0."
            ),
            "code": "run_stage('reload')",
        },
        {
            "md": (
                "**What to notice:** parity shows that the export is complete and loads safely — not that the predictions are "
                "right. A saved model reproduces its mistakes just as faithfully as its successes."
            ),
        },
        # ------------------------------------------------------------------ 13. bundle
        {
            "md": (
                "## 13. Export the provenance bundle · [Engineering]\n\n"
                "The `bundle` stage writes `outputs/{stem}_result.json`, which links every record of this run — input manifest, "
                "split, untrained-head and baseline results, training history, evaluation report, unseen prediction and reload "
                "parity — with the notebook's source revision, the model id, revision and licence, the runtime versions, and the "
                "SHA-256 of every output file.\n\n"
                "**Expected result:** a list of the run's output files, including `{stem}_input_manifest.json`, "
                "`{stem}_evaluation_report.json`, `{stem}_result.json`, `{stem}_top_k.csv` and `ast-audio-adapter-v1.pt`."
            ),
            "code": "run_stage('bundle')",
        },
        # ------------------------------------------------------------------ 14. activity
        {
            "md": (
                "## 14. Optional activity: change one thing — evaluation noise · [Concept]\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.** This activity is off by default and changes nothing "
                "the canonical path produced: with `RUN_ACTIVITY = False` the next cell only prints how to switch it on. When "
                "on, the `activity` stage loads the exported adapter, adds seeded white noise to each of the six evaluation "
                "clips at `ACTIVITY_SNR_DB`, and scores clean and noisy clips with the same model. It writes only under "
                "`outputs/activity/` and stops if a canonical output changed.\n\n"
                "**Change one thing:** the noise level. 0 dB means the noise is as loud as the sound; 20 dB means it is ten "
                "times quieter (in amplitude); -10 dB means it is louder. The adapter, the clips and the noise seed stay fixed. "
                "Set `RUN_ACTIVITY = True` and run the cell with the default `ACTIVITY_SNR_DB = -10.0`; then change the value to "
                "0 and to 20 and run the cell again for each.\n\n"
                "**Predict before running:** at -10 dB, which classes do you expect to be misclassified, and as what? Will 0 dB "
                "and 20 dB differ from it? Write it down."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_SNR_DB = -10.0  # @param {{type:\"number\"}}\n\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', '--snr-db', ACTIVITY_SNR_DB)\n"
                "else:\n"
                "    print({{'activity': 'skipped (optional)', 'to_run': 'set RUN_ACTIVITY = True and an ACTIVITY_SNR_DB value, then run this cell'}})"
            ),
        },
        {
            "md": (
                "**Observe:** clean and noisy accuracy, macro-F1 and confusion matrices, and how many confusion-matrix rows "
                "changed.\n\n"
                "**Explain:** did the result match your prediction? Which true class moved off the diagonal, and into which "
                "predicted class?\n\n"
                "**Question tested:** how robust is a head trained on clean synthetic clips when the recording conditions change?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "White noise is broadband, which is what the geophony clips are made of. Once the noise is louder than the "
                "sound, noisy biophony and anthrophony clips are typically pulled to geophony: in the local CPU pre-flight of "
                "this notebook, -10 dB moved all four biophony and anthrophony clips to geophony (accuracy 0.33), while 0 dB "
                "and 20 dB changed nothing. Your runtime may differ slightly, and six clips at one noise seed are a tiny "
                "sample: a result against that direction is an observation about this run, not a failed activity. Because only "
                "the noise level changed, any difference is caused by it. The lesson carries over to real data: a head trained "
                "and evaluated on clean recordings says little about noisy ones.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ troubleshooting
        {
            "md": (
                "## Troubleshooting · [Engineering]\n\n"
                "| Observation | Action |\n"
                "|---|---|\n"
                "| Section 1 stops: not Linux x86_64 | Use Google Colab, Kaggle or a Linux Jupyter server; the locked environment is built for manylinux x86_64. |\n"
                "| Section 1 stops: not enough disk | Start a fresh runtime, or delete earlier `outputs/{stem}/` run directories. |\n"
                "| Section 2: download, `uv` or hash failure | Retry once on a stable connection. Never remove a pin or a hash; a hash mismatch means the file is not the locked one. |\n"
                "| Section 3: Hugging Face download fails or a digest mismatches | Retry; delete `weights/ast-audioset/` and rerun Section 3 if a partial file remains. Never edit the manifest. |\n"
                "| A stage fails | The cell repeats the stage's error message; the full log is in the run directory under `logs/<stage>.log`. Fix the cause and rerun from that section. |\n"
                "| \"… is missing: run the stage that writes it\" | A later cell ran before an earlier one. Run the notebook from the top. |\n"
                "| `The upload dialog needs Google Colab` | Outside Colab, put the file on the machine and set `BYOD_WAV_PATH` or `BYOD_ZIP_PATH` to its path. |\n"
                "| BYOD ZIP refused | The message names the file and the fix: resample to 16 kHz, trim to 10.24 s, keep exactly the three class folders, remove duplicate clips, or add a second clip to a class. |\n"
                "| Reload parity fails | Do not use the export. Rerun from Section 9; if it persists, keep the logs and report it. |\n\n"
                "Nothing is installed into the kernel, so no runtime restart is ever needed."
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The adapted classifier produces uncalibrated softmax scores over `ADAPT_CLASSES` (`geophony`, `biophony`, "
        "`anthrophony`) and predicts by argmax. The baseline comparison measures performance relative to a trivial "
        "majority rule fitted on the training labels, on six held-out synthetic clips. Training is bounded to the classifier "
        "head with frozen backbone parameters; it runs on CPU or GPU. The generated clips are distinct and the split has no "
        "duplicates, but they remain simple synthetic signatures, and six evaluation clips are a sample-sanity check, not an "
        "estimate of field performance.\n\n"
        "Successful execution proves that the recorded repository revision's package and stage runner, carried in this "
        "standalone notebook, can build a hash-locked environment without touching the kernel, acquire and digest-verify the "
        "pinned model, validate the demonstrated dataset contract, execute supervised head adaptation, evaluate it against a "
        "training-majority baseline, reload the adapter with numeric parity, and emit the shown machine-readable artifacts — "
        "without the repository being reachable. It does **not** establish benchmark superiority, calibrated confidence, "
        "production fitness, or performance on real field recordings.\n\n"
        "## Conclude with evidence · [Evaluation practice]\n\n"
        "Complete this paragraph with the numbers your run printed:\n\n"
        "> On [number] held-out synthetic clips that are distinct from the training clips, the adapted head reached accuracy "
        "[value] and macro-F1 [value], against [value] for a majority rule fitted on the training labels and [value] for the "
        "untrained head. The reloaded adapter reproduced the trained model's scores within [max difference]. With noise at "
        "[SNR] dB (optional activity), accuracy was [value]. These results show [what they do show] and do not show [one thing "
        "they cannot show].\n\n"
        "**Transfer:** switch on dataset BYOD (Section 6) with your own short recordings of the three sources — at least two "
        "per class, clips from one recording kept together, nothing confidential on a hosted runtime — and compare your "
        "result with the synthetic one. Which assumption from Section 7 is hardest to keep with real recordings?\n\n"
        "**AI Assistance Disclosure:** this notebook's code and explanations were developed with generative AI assistance under "
        "maintainer direction. The maintainer remains responsible for reviewing implementation, validating results and making "
        "release decisions.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/ast-audio-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/ast-audio-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/ast-audio-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/YuanGongND/ast\n"
        "- AST: Audio Spectrogram Transformer (Gong, Chung, Glass, 2021): https://arxiv.org/abs/2104.01778\n"
        "- Acoustic ecology (Krause, 2008; Pijanowski et al., 2011)"
    ),
}
