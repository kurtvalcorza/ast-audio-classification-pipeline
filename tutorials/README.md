# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/ast-audio-classification-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/ast-audio-classification-pipeline/blob/main/tutorials/ast_audio_classification_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MIT%2Fast--finetuned--audioset--10--10--0.4593-ffcc4d?style=flat)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
[![Upstream](https://img.shields.io/badge/Upstream-YuanGongND%2Fast-181717?style=flat&logo=github&logoColor=white)](https://github.com/YuanGongND/ast)
[![arXiv](https://img.shields.io/badge/arXiv-2104.01778-b31b1b.svg)](https://arxiv.org/abs/2104.01778)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `ast_audio_classification_colab.ipynb` | `TASK-INFERENCE` | multi-label audio event classification over the 527 AudioSet labels with `MIT/ast-finetuned-audioset-10-10-0.4593`; rank-ordered independent sigmoid scores, no shipped threshold, no metric (no helper, no labelled audio) | CPU (CUDA used automatically when available) | single PCM WAV file, gated off by default | **Candidate** — static checks pass; the clean-runtime execution run is pending and will be recorded in `../docs/release-verification.md`, which must be reviewed for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `ASTAudioClassificationPipeline` from the repository public API rather than reimplementing model loading; model acquisition goes through the package: `stage_missing_files(WEIGHTS_DIR, allow_download=True)` fetches only the manifest entries a fresh clone lacks, at the pinned revision, `verify_snapshot` re-hashes every entry, and `from_pretrained(weights_dir=WEIGHTS_DIR)` loads the verified files (`local_files_only=True`, `trust_remote_code=False`; the notebook never calls `huggingface_hub`).
- The default sample is a synthetic 3 s, 440 Hz sine tone generated in code at 16 kHz (no download, no ground truth); its ranking is sanity evidence for the input contract, feature extraction and forward pass, not a correctness or benchmark claim. The smoke observation quoted from the model card (`Sine wave` first at 0.84) is one observation, not a calibration point.
- Score semantics: every `score` is an independent sigmoid, described as uncalibrated and not a probability; no threshold is shipped and the caller owns per-label thresholds and calibration; exported top-k scores keep their rank ordering in both JSON and CSV.
- Ceilings (`SAMPLE_RATE` 16 kHz, `MAX_AUDIO_SECONDS` 10.24 s window, `MAX_INPUT_SECONDS` 120 s, `NUM_LABELS` 527) are printed before the model runs; a clip longer than the window is announced as truncated before inference and the result carries `truncated: True`.
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog; BYOD decodes PCM WAV with the standard-library `wave` module because no audio decoder beyond `torchaudio.functional.resample` is pinned.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
