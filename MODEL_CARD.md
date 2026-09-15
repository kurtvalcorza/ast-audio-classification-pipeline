---
license: bsd-3-clause
model_card_spec: "1.1"
pipeline_tag: audio-classification
task: "Others - Audio Event Classification"
base_model: MIT/ast-finetuned-audioset-10-10-0.4593
date_published: "2022-11-14"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint"
---

# Audio Spectrogram Transformer — AudioSet Inference and Acoustic-Ecology Head Adaptation

> [!WARNING]
> Provided for research, training, and evaluation. The upstream weights are supplied under BSD-3-Clause and the repository code under Apache-2.0, without warranty. This pipeline has not been validated for production, clinical, safety-critical, legal, or other high-impact decision use.

## Interactive Colab Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/ast-audio-classification-pipeline/blob/main/tutorials/ast_audio_classification_colab.ipynb)

The standalone `E2E` tutorial verifies the pinned base checkpoint, demonstrates its unchanged AudioSet head, adapts a seeded three-class head over frozen AST features, evaluates a held-out generated split, exports a classifier-only artifact, and checks a fresh reload.

#### Description

This package wraps `MIT/ast-finetuned-audioset-10-10-0.4593` at immutable revision `f826b80d28226b62986cc218e5cec390b1096902`. The base model applies a Vision Transformer encoder to 128-bin audio filterbanks and emits 527 AudioSet logits. Base inference applies independent sigmoids. The adaptation API replaces that head with an ordered multiclass classifier, freezes the transformer backbone, caches its pooled features, and updates only the new head with AdamW. Adapted inference uses softmax over the target vocabulary.

The default tutorial task has three acoustic-ecology teaching labels: `geophony`, `biophony`, and `anthrophony`. Its 24 audio clips are deterministic procedural signals designed to make the execution path inspectable. They are not recordings of real habitats and do not establish environmental-audio quality.

#### Intended Use and Limitations

###### Primary Intended Uses

The unchanged base path is suitable for exploratory tagging of short audio against the AudioSet ontology. The adaptation path is a compact example of supervised transfer learning: validate a labelled waveform dataset, create a disjoint split, replace the classifier, freeze the backbone, fit the head, compare held-out metrics with a majority baseline, and move the head through an explicit artifact boundary. It is intended for teaching, integration testing, and prototyping before a user supplies a representative labelled corpus.

###### Primary Intended Users

Intended users are ML engineers, data scientists, educators, and application developers who understand waveform sampling, supervised splits, and the difference between a demonstration result and deployment evidence. Users must validate label definitions, collection consent, distribution coverage, and error costs for their own audio.

###### Out-of-scope use cases

The pipeline does not transcribe speech, identify speakers, localise events in time, separate sources, or learn the transformer backbone. It is not a validated acoustic-ecology classifier: the default classes are represented by easily separated synthetic constructions. It must not make autonomous health, safety, security, employment, credit, housing, education, insurance, or legal decisions. The classifier-head artifact is not standalone; it requires the exact pinned base architecture and revision.

#### Factors

###### Groups

AudioSet includes human vocal categories and YouTube-derived recordings, but neither the upstream checkpoint nor this repository reports performance by age, gender, accent, language, disability, geography, or recording context. The synthetic adaptation dataset contains no human speakers and cannot reveal group disparities. Any use on recordings of people requires consent review and subgroup evaluation on the intended population.

###### Instrumentation

The base checkpoint was trained from heterogeneous consumer recordings and codecs. The pipeline expects finite mono float32 samples in `[-1, 1]`; dataset adaptation requires 16 kHz and clips from 0.025 through 10.24 seconds. Base inference can resample other rates and accepts up to 120 seconds, while only the first 10.24 seconds reach AST. Incorrect declared sample rate, clipping, gain changes, microphone response, codec loss, and background mixing can alter predictions.

###### Environment

The declared environment is Python 3.12 with pinned PyTorch, torchaudio, torchvision, Transformers, safetensors, NumPy, and huggingface-hub versions from `pyproject.toml`. The current E2E path was exercised locally on CPU with float32 and the verified snapshot. A fresh supported-runtime execution of the current notebook has not yet been recorded. CUDA numerical results can differ slightly while still satisfying the reload tolerance.

#### Metrics

###### Performance Measures

For unchanged AudioSet inference, the repository has no ontology-aligned labelled corpus and reports no reproduced base-model metric; `evaluation_report()` therefore returns `not-measurable`. For the adapted multiclass path, `evaluate()` reports accuracy, macro-F1, per-class precision/recall/F1/support, a confusion matrix, majority-class accuracy, and accuracy delta from that baseline.

On the deterministic default split, the local CPU pre-flight observed accuracy `1.0` and macro-F1 `1.0` on six held-out generated clips, versus majority accuracy `0.3333`. This tiny, constructed sample is deliberately separable and is only a functional sanity result. It is not an estimate for real acoustic-ecology recordings, and no confidence interval is meaningful at this size.

###### Decision thresholds

The base path ships no label-presence threshold: its independent sigmoid values are uncalibrated ranking scores. The adapted path reports an argmax class and a softmax distribution over the ordered three-class vocabulary; no abstention or minimum-confidence threshold is provided. A real deployment must select thresholds or abstention rules from representative validation data and its false-positive and false-negative costs.

###### Approaches to uncertainty and variability

The tutorial fixes dataset, split, head-initialisation, training, and per-epoch shuffle seeds. Its default schedule is five epochs, batch size 4, learning rate `1e-3`, and 25 optimizer steps over 18 training records. Determinism on one device does not address sampling uncertainty, distribution shift, label ambiguity, or hardware-level floating-point variation. Real evaluation should use a larger independent test set, repeated seeds, class-stratified intervals, and calibration checks.

#### Ethical considerations and biases

###### Data

The upstream checkpoint was fine-tuned on weakly labelled YouTube excerpts from AudioSet and may inherit representation, annotation, consent, copyright, and geographic biases. The repository does not vendor audio. Its default adaptation records are generated sine, noise, chirp, harmonic, and amplitude-gated signals with no people or personal data. The optional upload paths process user-provided audio inside the notebook runtime; users remain responsible for authority, privacy, retention, and third-party hosted-runtime policies.

###### Human Life

No result here is suitable for autonomous decisions about a person's health, safety, identity, intent, or legal status. A high-scoring alarm, gunshot, cough, scream, or speech label is not verified evidence that the event occurred. Sensitive use would require human review of every relevant clip, independent domain evaluation, monitoring, and any applicable regulatory approval.

###### Mitigations

1. The package pins the model ID and 40-character revision, checks every local snapshot file against recorded byte length and SHA-256, loads locally with remote code disabled, and only permits a revision-pinned download when explicitly enabled.
2. Input and dataset validators reject malformed shape, dtype, amplitude, finiteness, rate, duration, duplicate ID, label, class-name, or class-coverage conditions before adaptation.
3. The split is seeded, stratified, disjoint, and guarantees at least one record from every class on each side.
4. Adaptation refuses to begin while the backbone remains trainable. The exported artifact contains only classifier parameters and records the class order, activation, training settings, and exact base lineage.
5. Artifact reload uses `torch.load(..., weights_only=True)`, validates its schema and identity, reconstructs the head, and can be checked numerically against pre-export predictions.
6. The notebook labels synthetic metrics as sample sanity, reports the majority baseline, keeps uploads disabled by default, bounds ZIP resources, rejects unsafe member paths, and never extracts the archive.

###### Risks and harms

Likely failure modes include silent loss of events after the 10.24-second model window, wrong spectra from a falsely declared sample rate, confident-looking scores on out-of-distribution sound, overfitting a small adaptation set, misleadingly strong synthetic results, class-order mismatch in downstream consumers, and privacy exposure when uploading recordings to a hosted notebook. Treat provenance fields and warnings as required controls, not optional metadata.

###### Use cases

Prohibited uses include covert surveillance, speaker or demographic profiling, fabricating evidence that a sound occurred, and unlawful discrimination. Do not infer identity, age, gender, health, criminality, or intent from AudioSet labels. Do not imply endorsement by MIT, the upstream authors, DIMER, or any model provider.

## Immutable provenance

- Model: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Revision: `f826b80d28226b62986cc218e5cec390b1096902`
- Manifest: `weights/ast-audioset/dimer-base-manifest.json`, format `dimer_hf_snapshot` v1, four files, `totalBytes` 346433173
- `model.safetensors` SHA-256: `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d`
- `config.json` SHA-256: `a93d525511d77e8ecc933d09674b85099815bbbb417c228a4edd655e252fb9ff`
- `preprocessor_config.json` SHA-256: `8d04ba5a9c6fca5d39d0de2b1fd05ecf79deb589fbba279728bbebac39934231`
- Adapter format: `org.valcorza.ast-audio.adapter.v1`, format version `1.0`; classifier state only, not a standalone model

## Input/output contract

- `from_pretrained(...)` verifies and loads the local base snapshot; remote download is opt-in and revision-pinned.
- Base `predict(audio, sample_rate, top_k=5)` returns ranked AudioSet labels with `activation: "sigmoid"`, truncation and resampling flags, duration, and base provenance.
- `synthetic_audio_dataset`, `validate_dataset`, and `split_dataset` implement the owner-namespaced waveform dataset contract `io.github.kurtvalcorza.dataset.audio.waveform-classification.v1`.
- `rehead(class_names, seed)`, `freeze_backbone()`, and `finetune(...)` perform classifier-head adaptation. The training records must already be valid 16 kHz labelled records covering every declared class.
- Adapted `predict()` returns rank-ordered `softmax` scores across the active class vocabulary; `evaluate()` consumes labelled records and returns multiclass metrics plus the majority baseline.
- `save_artifact(path)` writes the classifier-only adapter. `load_artifact(path)` rejects a format, version, kind, model ID, revision, activation, vocabulary, or tensor-shape mismatch and refreshes inference after loading.
- Base ceilings: `SAMPLE_RATE = 16000`, `MAX_AUDIO_SECONDS = 10.24`, `MAX_INPUT_SECONDS = 120.0`, `MIN_AUDIO_SECONDS = 0.025`, `NUM_LABELS = 527`.

## Runtime

- Declared pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, and `huggingface-hub==0.36.2`; Python `>=3.12,<3.13`.
- Local CPU E2E pre-flight on 2026-09-16: all 16 generated code cells ran with the exact pins already installed; the real pinned base loaded, 86,187,264 backbone parameters were frozen, and the three-class head had 3,843 trainable parameters. Five cached-feature epochs over 18 records took 1.259 seconds in that warm-process run, held-out accuracy/macro-F1 were `1.0`/`1.0` on six generated clips, and the unseen generated biophony clip was classified as biophony.
- The exported classifier-only artifact was approximately 19 KB. Loading it over a fresh base instance reproduced the checked three-class scores exactly.
- The current E2E notebook has not been executed in a fresh supported runtime. BYOD branches and generalisation to real field recordings are also unverified.

## References

- Gong, Y., Chung, Y.-A., and Glass, J. (2021). AST: Audio Spectrogram Transformer. https://arxiv.org/abs/2104.01778
- Gemmeke, J. F. et al. (2017). Audio Set: An ontology and human-labeled dataset for audio events. ICASSP.
- Upstream checkpoint: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
- Original implementation: https://github.com/YuanGongND/ast
