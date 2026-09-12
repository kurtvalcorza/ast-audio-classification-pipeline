---
license: bsd-3-clause
model_card_spec: "1.1"
pipeline_tag: audio-classification
base_model: MIT/ast-finetuned-audioset-10-10-0.4593
---

# Audio Spectrogram Transformer, AudioSet fine-tune (DIMER package v0.1.0) — Audio Event Classification

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MIT%2Fast--finetuned--audioset--10--10--0.4593-ffcc4d?style=flat)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-YuanGongND%2Fast-181717?style=flat&logo=github&logoColor=white)](https://github.com/YuanGongND/ast)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2104.01778-b31b1b.svg)](https://arxiv.org/abs/2104.01778)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Pipeline](https://img.shields.io/badge/Pipeline-ast--audio--classification--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/ast-audio-classification-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, resolve and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/ast-audio-classification-pipeline/blob/main/tutorials/ast_audio_classification_colab.ipynb) [`ast_audio_classification_colab.ipynb`](https://github.com/kurtvalcorza/ast-audio-classification-pipeline/blob/main/tutorials/ast_audio_classification_colab.ipynb)  
  *Multi-label audio event classification over the 527 AudioSet labels with the pinned `MIT/ast-finetuned-audioset-10-10-0.4593` weights: 16 kHz waveform → 128-bin filterbank → one logit per label; no adaptation occurs.*

---

###### Description

This package wraps the Audio Spectrogram Transformer (AST) checkpoint `MIT/ast-finetuned-audioset-10-10-0.4593`, pinned to revision `f826b80d28226b62986cc218e5cec390b1096902`. AST is a Vision-Transformer-style encoder applied to a log-mel spectrogram: the pinned `config.json` describes 12 layers of hidden size 768 over 16x16 patches with stride 10 in both time and frequency, and a 527-way classification head over the AudioSet ontology (Gong et al., 2021). At inference the feature extractor converts 16 kHz mono audio into a 128-bin Kaldi filterbank, pads or crops it to 1,024 frames, and the encoder emits one logit per label; no adaptation happens at inference and this repository trains nothing. What the repository adds is the `ASTAudioClassificationPipeline` class in `src/ast_audio_classification_pipeline/pipeline.py`: manifest-based snapshot verification (`verify_snapshot`), a loader that refuses unverified or remote code, input validation with named ceilings, an explicit sigmoid over the logits with top-k selection, and provenance fields in every result.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is multi-label audio event tagging: input is a one-dimensional float waveform with its sample rate, output is a ranked list of `{label, index, score}` over the 527 AudioSet classes, each score an independent sigmoid in [0, 1] (`activation: "sigmoid"` in the result). Envisioned applications are coarse content tagging of short clips: flagging which segments of a field recording contain speech, music, vehicles, animals, or alarms; pre-filtering audio archives before manual review; and producing weak labels or features for a downstream model trained on the operator's own data. The pipeline is meant as a zero-configuration baseline and as an inference component inside a larger system, not as a stand-alone decision system.

###### Primary Intended Users

Intended users are machine-learning engineers, data scientists, and application developers integrating audio tagging into research or internal enterprise systems, including public-service teams triaging recordings they are authorised to hold. The pipeline assumes its users understand sample rates and mono conversion, know that only the first 10.24 s of a clip are scored, can read a multi-label sigmoid score as a per-label confidence rather than a probability distribution, and accept responsibility for validating label quality on their own recordings before acting on any tag.

###### Out-of-scope use cases

1. **Capability boundary:** the model does not transcribe speech, identify speakers, localise sounds in time within the window, separate sources, or detect events outside the 527 AudioSet labels. Transcription belongs to the sibling `whisper-asr-pipeline`; there is no sibling for speaker identity and none is planned.
2. **Input boundary:** input must be a 1-D float `numpy.ndarray` (`TypeError` otherwise) of at least 0.025 s (`MIN_AUDIO_SECONDS`) and at most 120 s (`MAX_INPUT_SECONDS`, `ValueError` above it). Only the first 10.24 s (`MAX_AUDIO_SECONDS`) reach the model; longer clips are cropped and returned with `truncated: True`. Audio at a rate other than 16 kHz is resampled with `torchaudio.functional.resample`, which cannot restore content above the original Nyquist frequency. Stereo, integer PCM, and NaN/inf samples are rejected.
3. **Decision boundary:** not for autonomous safety, security, medical, or legal action on a tag (gunshot, glass breaking, cough, scream) without a human listening to the clip; the score is uncalibrated and the labels were learned from YouTube-derived weak labels.

#### Factors

###### Groups

The pipeline is human-centric in part: AudioSet is drawn from YouTube and its ontology contains speech, singing, laughter, crying, and demographic labels such as "Male speech, man speaking", "Female speech, woman speaking", and "Child speech, kid speaking" (indices 1-3 in the pinned `config.json`). Neither the upstream checkpoint README nor this repository reports group-level performance by speaker age, gender, accent, language, or recording context, and no such audit was performed here. An operator applying the model to recordings of people must measure per-label precision and recall across the speaker groups present in their own data and must not use the gendered or age-related speech labels as demographic classifiers.

###### Instrumentation

The training data behind the checkpoint is AudioSet: audio tracks of YouTube videos, captured by whatever consumer microphones, cameras, phones, and encoders the uploaders used, then transcoded by YouTube and resampled to 16 kHz by the AST authors. The instrument characteristics that reach the model are therefore heterogeneous: lossy codecs, automatic gain control, clipping, and mixed mono/stereo down-mixes. At inference the pipeline fixes the representation (16 kHz mono, 25 ms Hann-window frames with a 10 ms hop, 128 mel bins, normalised with the mean -4.2677 and standard deviation 4.5690 from `preprocessor_config.json`). It validates dtype, shape, finiteness, duration, and sample rate, and resamples when told the true rate, but it cannot detect a caller that passes the wrong rate, clipped or gain-distorted input, or a different codec chain; those errors propagate straight into the spectrogram.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0` (CUDA 13.0 build), `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3` (`pyproject.toml`). The smoke run recorded under "Runtime" used an NVIDIA RTX 5070 Ti (16 GB, sm_120) in float32 with a peak of 414 MiB allocated; the model is 346 MB of float32 weights and runs on CPU with the same code path (`device="cpu"`), though CPU latency was not measured in this pass. Data environment: the model assumes clips resembling AudioSet's YouTube distribution of everyday sounds at consumer quality, scored in a fixed 10.24 s window. Long clips are cropped, so events after the window are invisible; quiet events under louder ones, unusual microphones, ultrasonic or sub-bass content, and sound classes absent from the ontology degrade to low or wrong scores rather than to an error.

#### Metrics

###### Performance Measures

The pipeline reports no performance measure. `predict()` returns per-label sigmoid `score` values and their ranking; it does not compute mean average precision, per-label AUC, or top-k accuracy because none can be computed without labelled clips, and the repository ships no labelled audio. The upstream README states no number for this checkpoint beyond the name's "0.4593", which the AST paper reports as AudioSet mAP for this configuration; that figure is upstream-reported and was not reproduced here. An operator who needs a measure should collect clips labelled against the same 527-class ontology, run `predict()` with `top_k=527`, and compute mAP (the standard AudioSet measure, because it is threshold-free and averages over classes of very different frequency) alongside per-label precision at their chosen threshold, since mAP alone hides which classes fail.

###### Decision thresholds

No decision threshold is shipped. `predict()` applies a sigmoid to each logit and ranks labels by score; the ranking with `top_k` (default 5, ceiling 527) is a presentation choice, not an acceptance rule, and no label is asserted present or absent. The threshold was withheld because the costs are asymmetric per label and per deployment: a missed "Smoke detector, smoke alarm" tag in a monitoring system is expensive, a spurious "Music" tag in an archive search is cheap. The operator owns the threshold and should set it per label from precision-recall curves on their own labelled clips, choosing a lower threshold where false negatives cost more than false positives and a higher one where review effort dominates. The smoke run's top score of 0.84 for "Sine wave" on a synthetic tone is one observation, not a calibration point.

###### Approaches to uncertainty and variability

No metric is reported, so no estimation procedure or dispersion applies; the operator who computes mAP on their own clips owns the split design and any resampling. Sources of run-to-run variability in the pipeline itself: the forward pass is deterministic on a fixed device and dtype (no sampling, no dropout at eval, `torch.inference_mode`), but CUDA kernel selection, float32 versus other precisions, and the resampler can shift scores in the third or fourth decimal place across hardware; no seed is set because nothing in the path is stochastic. The sigmoid outputs are not calibrated probabilities: they were trained with binary cross-entropy on weak, incomplete labels, so a score of 0.5 does not mean a 50% chance the event is present. A caller needing calibrated confidence must fit a per-label calibrator (for example temperature scaling or isotonic regression) on labelled clips from their own distribution.

#### Ethical considerations and biases

###### Data

The checkpoint was fine-tuned on AudioSet, a dataset of weakly labelled 10 s excerpts from YouTube videos, after ImageNet pretraining of the vision backbone according to the AST paper; the upstream README does not enumerate the clips, their uploaders, or any consent, and this repository cannot. The data contains human speech, voices, and identifiable recordings of people, so personal data is present, not merely unruled out; whether copyrighted or otherwise restricted material is included is not disclosed upstream. This repository distributes code, tests, and documentation; it does not vendor the weights in Git (the snapshot under `weights/ast-audioset/` is git-ignored and reproduced from the pinned revision), and ships no audio samples. The operator must audit the audio they submit for personal, confidential, or legally restricted content and for recording consent; the pipeline performs no such check.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by anyone. The only validation performed is the offline unit suite and one smoke inference on a synthetic tone recorded under "Runtime". Use in a sensitive domain is foreseeable — cough or breathing labels in health screening, gunshot or scream labels in security monitoring — and would be admissible only with a human listening to every flagged clip, an independent domain validation of per-label error rates on that deployment's own recordings, and whatever regulatory clearance the domain requires.

###### Mitigations

1. **Supply-chain integrity:** `MODEL_ID` and the 40-hex `MODEL_REVISION` are module constants; `verify_snapshot()` reads `weights/ast-audioset/dimer-base-manifest.json`, checks `modelId` and `revision` against those constants, and checks the byte size and SHA-256 of all four listed files before any load, raising on the first mismatch. The loader passes `local_files_only=True` for the verified directory and `trust_remote_code=False` always; `allow_download=True` is the only path to the Hub and it pins `revision=MODEL_REVISION`. A test flips one hex digit of a manifest digest and asserts the check raises.
2. **Input integrity:** `predict()` rejects non-`ndarray`, non-1-D, non-float, NaN/inf input, non-integer or non-positive sample rates, clips under `MIN_AUDIO_SECONDS` or over `MAX_INPUT_SECONDS`, and `top_k` outside `[1, 527]` before the model runs (one test per check); the backend's logit shape is checked after.
3. **Statistical mitigations:** none are implemented; there is no class balancing or subsampling because the pipeline does no training.
4. **Reproducibility:** every runtime dependency is pinned with `==` in `pyproject.toml`; each result carries `model_id`, `model_revision`, `duration_seconds`, `window_seconds`, `resampled`, `input_sample_rate` and `truncated`.
5. **Refusals:** no argmax or presence threshold is applied; labels are never asserted as present. No file loading, URL fetching, or streaming is exposed: the caller decodes audio and passes an array.

###### Risks and harms

1. **Silent cropping:** only the first 10.24 s are scored; an event later in a clip is missed. The `truncated` flag is set, but an operator who ignores it bears the harm; likely whenever clips exceed the window.
2. **Wrong sample rate:** a caller passing the wrong `sample_rate` gets a pitch-shifted spectrogram and wrong labels with no error; the pipeline cannot detect it.
3. **Out-of-distribution overconfidence:** sounds outside the ontology or unlike YouTube audio still receive scores; an operator treating a high score as evidence is misled, and the data subject bears the consequence if the tag is about a person.
4. **Bias transfer:** speech-related and demographic labels reflect AudioSet's uploader population and annotators; error rates by voice type are unmeasured, so a deployment that filters or prioritises by those labels can disadvantage speakers whose voices are tagged less reliably.
5. **Automation bias and privacy:** a reviewer may trust tags over listening; processing recordings of people exposes their content to whatever system stores the results. Magnitude ranges from wasted review time to a wrongful security or health escalation when the decision boundary above is ignored.

###### Use cases

The pipeline must not be used for covert surveillance of people through their recorded environment, for inferring gender, age, or health from voice-related labels, for demographic profiling or social scoring, or for any unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access. It must not be used to fabricate evidence that a sound occurred, to monitor individuals without consent where consent is required, or in any way that breaches the BSD-3-Clause terms of the upstream weights (including implying MIT endorsement) or the DIMER deployment terms. These prohibitions hold even where the model would produce a confident tag.

## Immutable provenance

- Model: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Revision: `f826b80d28226b62986cc218e5cec390b1096902`
- Manifest: `weights/ast-audioset/dimer-base-manifest.json`, format `dimer_hf_snapshot` v1, 4 files, `totalBytes` 346433173
- `model.safetensors` (346,404,948 bytes) SHA-256: `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d`
- `config.json` (26,763 bytes) SHA-256: `a93d525511d77e8ecc933d09674b85099815bbbb417c228a4edd655e252fb9ff`
- `preprocessor_config.json` (297 bytes) SHA-256: `8d04ba5a9c6fca5d39d0de2b1fd05ecf79deb589fbba279728bbebac39934231`
- Upstream reference: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593

## Input/output contract

- `ASTAudioClassificationPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`: verifies and loads the local snapshot; `device` defaults to `cuda:0` when available, else `cpu`.
- `predict(audio: np.ndarray, sample_rate: int, top_k: int = 5) -> dict` with keys `predictions` (list of `{label: str, index: int, score: float}` sorted by score, length `top_k`), `activation` (`"sigmoid"`), `truncated` (bool), `duration_seconds` (float), `window_seconds` (10.24), `resampled` (bool), `input_sample_rate` (int), `model_id`, `model_revision`.
- Constants: `SAMPLE_RATE = 16000`, `MAX_AUDIO_SECONDS = 10.24`, `MAX_INPUT_SECONDS = 120.0`, `MIN_AUDIO_SECONDS = 0.025`, `NUM_LABELS = 527`, `DEFAULT_TOP_K = 5`.
- No metric helper is shipped; mAP requires labelled clips the repository does not have.

## Runtime

- Pins (`pyproject.toml`): `torch==2.14.0` (cu130 build in the venv), `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`; dev `pytest==8.4.2`, `ruff==0.16.6`. Python 3.12.
- Executed 2026-09-12 in the `dimer-next16` Linux venv (WSL, claude-science): `pytest -q -o addopts= tests` — 18 passed, exit 0; `ruff check src tests` clean.
- Smoke, executed: `from_pretrained()` on the verified snapshot, `predict()` on a 3 s 440 Hz sine at 16 kHz, `device="cuda:0"`, float32; load 4.56 s, inference 0.29 s, 4.85 s total, peak 414 MiB; top-5 `Sine wave` 0.8419, `Dial tone` 0.0325, `Chirp tone` 0.0105, `Beep, bleep` 0.0097, `Busy signal` 0.0073; `truncated: false`.
- Not executed: the CPU path, the `allow_download=True` Hub path, inputs longer than the window against the real model, and any labelled evaluation.

## References

- Gong, Y., Chung, Y.-A., Glass, J. (2021). AST: Audio Spectrogram Transformer. https://arxiv.org/abs/2104.01778
- Gemmeke, J. F. et al. (2017). Audio Set: An ontology and human-labeled dataset for audio events. ICASSP.
- Upstream checkpoint card: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 (pinned README, revision above)
- Original code: https://github.com/YuanGongND/ast
