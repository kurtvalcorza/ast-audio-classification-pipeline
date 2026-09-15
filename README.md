# AST Audio Classification Pipeline

DIMER-oriented Audio Spectrogram Transformer pipeline with both pinned AudioSet inference and bounded classifier-head adaptation. The standalone tutorial turns the pretrained representation into a three-class acoustic-ecology classifier entirely inside one Python 3.12 runtime.

## Upstream alignment

- Model: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Revision: `f826b80d28226b62986cc218e5cec390b1096902`
- Upstream weight license: BSD-3-Clause
- Base task: multi-label AudioSet classification over 527 labels using independent sigmoid scores
- Tutorial adaptation: single-label `geophony` / `biophony` / `anthrophony` classification with a seeded three-class head and frozen AST backbone

## Quick start

```python
from ast_audio_classification_pipeline import (
    ADAPT_CLASSES,
    ASTAudioClassificationPipeline,
    split_dataset,
    synthetic_audio_dataset,
)

records = synthetic_audio_dataset(n_samples=24, seed=42)
train_records, val_records = split_dataset(records, val_fraction=0.25, seed=42)

pipe = ASTAudioClassificationPipeline.from_pretrained()  # verifies weights first
pipe.rehead(ADAPT_CLASSES, seed=42)
pipe.freeze_backbone()
history = pipe.finetune(
    train_records,
    val_records,
    epochs=5,
    batch_size=4,
    learning_rate=1e-3,
    seed=42,
)
report = pipe.evaluate(val_records)
pipe.save_artifact("outputs/ast-audio-adapter-v1.pt")
```

The default sample has 24 generated clips and a deterministic stratified 18/6 train/validation split. Frozen backbone features are cached once; AdamW updates only the 3,843-parameter classifier head. `evaluate()` reports accuracy, macro-F1, per-class metrics, a confusion matrix, and the delta from the majority-class baseline.

The generated dataset uses the owner-namespaced representation `io.github.kurtvalcorza.dataset.audio.waveform-classification.v1`. It is a transparent tutorial fixture, not a real acoustic-ecology benchmark. The optional notebook BYOD path accepts one bounded ZIP with exactly `geophony/`, `biophony/`, and `anthrophony/` top-level directories and at least two 16 kHz PCM WAV files per class.

## Base inference contract

Before re-heading, `predict()` takes a 1-D float array in `[-1, 1]` plus its source sample rate and returns ranked independent sigmoid scores over the 527 AudioSet labels. Non-16 kHz input is resampled. Only the first 10.24 seconds reach the model, clips longer than 120 seconds are rejected, and truncation is reported explicitly. Because the repository has no AudioSet-labelled evaluation corpus, `evaluation_report()` for this base path remains `not-measurable`.

After re-heading, predictions are a softmax distribution over the ordered target vocabulary. The saved `org.valcorza.ast-audio.adapter.v1` artifact contains only the classifier state, target classes, adaptation settings, and exact base-model lineage; a fresh load reconstructs the pinned base model and safely reads the artifact with `weights_only=True`.

## Weights layout

```text
weights/ast-audioset/
  README.md  config.json  model.safetensors  preprocessor_config.json  dimer-base-manifest.json
```

`from_pretrained()` verifies the size and SHA-256 of every manifest entry and loads the local snapshot with `local_files_only=True`. Without a verified snapshot it raises unless `allow_download=True`, which fetches the immutable revision above. See `docs/WEIGHTS.md`.

## Tests

```text
pip install -e . --no-deps
pytest -q
python tools/build_notebook.py --check
python tools/validate_release_assets.py
ruff check src tests tools
```

The unit suite is offline and uses injected or tiny models. A separate local CPU check of the real pinned model is recorded in the model card; source checks and local execution do not replace a fresh supported-runtime notebook run.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/ast-audio-classification-pipeline/blob/main/tutorials/ast_audio_classification_colab.ipynb)

`tutorials/ast_audio_classification_colab.ipynb` declares the `E2E` profile under DIMER Notebook Specification 2.0 and is a generated standalone carrier. Its 14 stages install pinned dependencies, verify and demonstrate the unchanged base model, validate the dataset, split, re-head, freeze, adapt, evaluate, predict an unseen clip, export the adapter, reload a fresh base-model instance, check numerical parity, and export provenance. Edit `tools/notebook_template.py` or the package modules, then regenerate with `python tools/build_notebook.py`; do not hand-edit the notebook.

## Release status

**Candidate.** Source validation, offline tests, and a local CPU run of the current E2E implementation are recorded. The prior 2026-09-13 clean Colab execution covered an older inference-only notebook blob and does not qualify this E2E revision. Promotion requires an unchanged top-to-bottom run of the current generated notebook in a fresh supported runtime and review of its retained evidence. See `docs/release-verification.md`.

## Licensing

Repository code is Apache-2.0 (`LICENSE`). The upstream weights are BSD-3-Clause; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository's code and documentation were developed with generative AI assistance under maintainer direction. The maintainer remains responsible for review, validation, and release decisions; AI assistance is not independent verification or provider endorsement.
