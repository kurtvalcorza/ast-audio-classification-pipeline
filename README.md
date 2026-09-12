# AST Audio Classification Pipeline

DIMER-oriented inference wrapper for the **MIT Audio Spectrogram Transformer fine-tuned on AudioSet**, pinned to an immutable Hugging Face revision. The repository exposes multi-label audio event classification over the 527 AudioSet classes, a supply-chain check of the local weight snapshot, and machine-readable provenance.

## Upstream alignment

- Model: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Revision: `f826b80d28226b62986cc218e5cec390b1096902`
- Upstream weight license: BSD-3-Clause
- Upstream task: audio classification (AudioSet ontology, 527 labels, sigmoid per label)
- Repository adaptation: **none**; inference only

## Quick start

```python
import numpy as np
from ast_audio_classification_pipeline import ASTAudioClassificationPipeline, SAMPLE_RATE

pipe = ASTAudioClassificationPipeline.from_pretrained()          # verifies weights/ast-audioset first
t = np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE
sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
result = pipe.predict(sine, sample_rate=SAMPLE_RATE, top_k=5)
print(result["predictions"][0])   # {'label': 'Sine wave', 'index': ..., 'score': ...}
```

`predict()` takes a 1-D float array plus the rate it was captured at; input at a rate other than 16 kHz is resampled with `torchaudio.functional.resample`. Only the first 10.24 s reach the model (`MAX_AUDIO_SECONDS`); longer clips are cropped and flagged `truncated: True`, and clips above `MAX_INPUT_SECONDS` (120 s) are rejected so the caller chunks them.

## Weights layout

```
weights/ast-audioset/
  README.md  config.json  model.safetensors  preprocessor_config.json  dimer-base-manifest.json
```

`from_pretrained()` calls `verify_snapshot()` (size + SHA-256 of every manifest entry) and loads with `local_files_only=True`. Without a verified snapshot it raises unless `allow_download=True`, in which case it pulls the pinned revision from the Hub. See `docs/WEIGHTS.md`.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: they use an injected fake runner and a temporary manifest, never the weights.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/ast-audio-classification-pipeline/blob/main/tutorials/ast_audio_classification_colab.ipynb)

`tutorials/ast_audio_classification_colab.ipynb` is declared `TASK-INFERENCE` under DIMER Notebook Specification 1.0. Its default path generates a 3 s, 440 Hz sine tone in code at 16 kHz (no download, no ground truth), surfaces the pipeline ceilings, resolves the pinned model through the public API, ranks the 527 AudioSet labels by independent sigmoid score (uncalibrated, no shipped threshold), reports no metric (none is shipped and no labelled audio exists), and exports JSON plus a rank-ordered CSV. BYOD (one PCM WAV file) is optional and gated off by default. See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Candidate.** Static/unit checks do not constitute clean-runtime notebook evidence. The clean-runtime run of the tutorial is pending; complete `docs/release-verification.md` against the exact release revision before calling the notebook release-grade.

## Licensing

This repository's code is Apache-2.0 (`LICENSE`). The packaged upstream weights are BSD-3-Clause; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.
