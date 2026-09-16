# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/ast-audio-classification-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/ast-audio-classification-pipeline/blob/main/tutorials/ast_audio_classification_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MIT%2Fast--finetuned--audioset--10--10--0.4593-ffcc4d?style=flat)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

Notebook specification: **DIMER Notebook Specification 2.0**. The notebook is a **standalone** generated carrier in `GUIDED` pedagogical mode. It embeds the package modules, immutable model identity, snapshot manifest, and runtime pins, so the exported notebook does not need this repository at execution time. Edit the package or `tools/notebook_template.py`, regenerate with `python tools/build_notebook.py`, and use `--check` for parity.

| Notebook | Profile | Mode | Carrier | Capability | Default runtime | Samples | BYOD | Run-all | Release status |
|---|---|---|---|---|---|---|---|---|---|
| `ast_audio_classification_colab.ipynb` | `E2E` | `GUIDED` | standalone (generated) | pinned 527-label AudioSet inference, followed by a frozen-backbone three-class acoustic-ecology adaptation, held-out evaluation, classifier-head artifact export, fresh reload, and numerical parity check | CPU; CUDA used automatically | generated 24-clip balanced dataset, 18 train / 6 validation | optional single PCM WAV for base inference; optional bounded three-directory ZIP for adaptation; both off by default | verified — clean post-review Kaggle T4 Run all recorded for commit `79543f3`, blob `0be7254` (16/16 cells) | Candidate — clean-room execution recorded in [release verification](../docs/release-verification.md); promotion decision pending |

## Conformance notes

- **Standalone and parity:** `tests/test_notebook_parity.py` and `tools/validate_release_assets.py` verify that the carried `metrics.py`, `samples.py`, and `pipeline.py`, inline manifest, dependency pins, and generated notebook bytes match repository sources.
- **Run-all:** the default path stages and verifies `MIT/ast-finetuned-audioset-10-10-0.4593` at revision `f826b80d28226b62986cc218e5cec390b1096902`; demonstrates the unchanged independent-sigmoid AudioSet head; validates the deterministic dataset; performs the seeded split, re-head, freeze, five-epoch cached-feature fine-tune, held-out evaluation, unseen prediction, adapter export, fresh base reload, numerical comparison, and provenance export.
- **Dataset contract:** the default generated records use `io.github.kurtvalcorza.dataset.audio.waveform-classification.v1`. The optional dataset upload is one ZIP with exactly `geophony/`, `biophony/`, and `anthrophony/` top-level directories, at least two 16 kHz PCM WAV files per class, 0.025–10.24 seconds per clip, at most 100 files, 64 MiB compressed, and 256 MiB expanded. Members are inspected and decoded in memory; the archive is not extracted.
- **Evaluation:** base AudioSet inference is still not measurable without ontology-aligned labels. The adapted path evaluates six held-out clips with multiclass accuracy, macro-F1, per-class statistics, confusion matrix, and the delta from a majority-class baseline. This small synthetic result is a sample-sanity check, not a real-world benchmark.
- **Artifact:** `outputs/ast-audio-adapter-v1.pt` stores only the 3,843-parameter classifier head plus ordered classes, adaptation configuration, and exact base lineage. Reload uses `torch.load(..., weights_only=True)` over a fresh pinned base-model instance.
- **Outputs:** the run writes an input manifest, evaluation report, result/provenance JSON, rank-ordered CSV, and adapter artifact under `outputs/`.
- **Release boundary:** the exact current notebook passed unchanged from top to bottom on a clean Kaggle T4. It remains Candidate until the retained evidence is reviewed and an integrator explicitly approves promotion.

## AI Assistance Disclosure

The code and documentation were developed with generative AI assistance under maintainer direction. The maintainer remains responsible for review, validation, and release decisions.
