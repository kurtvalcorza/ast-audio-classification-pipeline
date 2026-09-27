# DIMER Sound Event Classification — Notebook Specification

Status: **Implementation proposal. No new notebook, model result or hosted qualification is claimed.**

Prepared: 2026-09-27. Normative basis: DIMER NOTEBOOK_SPEC 2.2. Scope of this task: this design document only.

## 1. Learning unit

**DIMER Notebook: Sound Event Classification — Learning to Recognize What We Hear**

Driving question: **What sound is present, and how much should we trust the model when the recording changes?**

Place beside Whisper in **Level 2 — Applied tasks**. Whisper introduces speech transcription; this unit introduces clip-level environmental-sound classification. Basic Python and Colab familiarity are sufficient. Explain waveform, sample rate, spectrogram, label and classifier head before using them.

| Field | Proposed decision |
|---|---|
| Host repository | `kurtvalcorza/ast-audio-classification-pipeline` |
| Artifact | `tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` |
| Profile / mode | `E2E` / `WORKSHOP`, standalone and self-paced |
| Backbone | Pinned Audio Spectrogram Transformer, frozen |
| Default adaptation | New ten-class head, including its classifier LayerNorm; no backbone updates |
| Data | All 400 ESC-10 clips, immutable manifest and original recording groups |
| Split | 240 train / 80 validation / 80 test |
| Controlled activity | Fixed synthetic background-noise realization at clean / 20 / 10 / 0 dB SNR |
| Runtime | Fresh Colab T4-class GPU; isolated Python 3.12 worker |
| Release state | Candidate pending exact hosted execution and real labelled BYOD qualification |

Catalogue description:

> Listen to labelled environmental recordings and explore how spectrograms represent sound. Adapt a classifier head on frozen AST features, compare it with simple baselines, and inspect errors on recordings held out from training. Test how controlled additive noise changes predictions, then export and reload the selected classifier and explain the limits of its results.

Completion records are optional personal notes, not required submissions. This specification does not update the published curriculum count or invent a Colab link.

## 2. Verified starting point and required additions

Inspected clean local repository commit: `1988bd6cdda5fb3d1d8619f866f53f7ed9c56f27`. This is a local source identity, not a fresh service-availability claim.

Base checkpoint: `MIT/ast-finetuned-audioset-10-10-0.4593`, revision `f826b80d28226b62986cc218e5cec390b1096902`. The current wrapper uses **527 independent sigmoid scores** for AudioSet inference. A reheaded classifier uses **softmax over the supplied single-label vocabulary**. These are different output contracts, not interchangeable probability tables. [Model card](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

Source inspection confirmed:

- `pipeline.py` supports arbitrary named reheading, freezing the backbone, cached pooled features, classifier-only AdamW training, classification metrics and artifact reload.
- Current `finetune` keeps the last epoch, lacks epoch-zero competition and does not expose checkpoint-selection callbacks. **The workshop must add a tested local trainer with explicit per-epoch snapshots and validation selection.** Do not claim the current method already implements that behavior or repeatedly call it for one epoch while silently resetting optimizer state.
- Current samples are synthetic geophony/biophony/anthrophony clips. This unit uses the ten actual event categories below. It does not relabel recordings into those three acoustic-ecology categories or claim ecological monitoring capability.
- The existing adapter is a classifier-only `.pt` payload, loaded with `torch.load(..., weights_only=True)`, validated against base identity, label order and tensor schema. Preserve that compatibility and add a SHA-256 manifest; do not describe it as a safetensors artifact.
- Source preprocessing supports resampling through torchaudio; the adaptation dataset expects mono float32 at 16 kHz. Source and notebook Python files were AST-parsed during specification work; no model was executed.

New build work includes real-data acquisition/pinning, source-group validation, the selected-checkpoint trainer, full-score exports, the noise activity, stricter artifact/refusal tests and notebook guidance. Carry the changes as documented notebook-local reference code; do not silently change the existing synthetic tutorial or production API.

## 3. Dataset, attribution and split ownership

Use only rows marked `esc10=True` in [ESC metadata](https://github.com/karolpiczak/ESC-50/blob/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/meta/esc50.csv), pinned to commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`.

Verified from that CSV: 400 clips; 304 original `src_file` groups; 80 clips in each official fold; eight examples per class per fold; zero source groups spanning folds. Label indices are newly assigned in this fixed alphabetical order:

`chainsaw`, `clock_tick`, `crackling_fire`, `crying_baby`, `dog`, `helicopter`, `rain`, `rooster`, `sea_waves`, `sneezing`.

| Role | Official folds | Clips | Clips per class | Permitted use |
|---|---|---:|---:|---|
| Training | 1, 2, 3 | 240 | 24 | Fit head and training-only baselines |
| Validation | 4 | 80 | 8 | Select epoch; later fixed descriptive activity |
| Test | 5 | 80 | 8 | One final comparison after selection is locked |

Retain original target IDs separately from the new contiguous indices. Reuse official folds without shuffling clips across them. Enforce `src_file` isolation, unique IDs, decoded-waveform duplicate checks before and after resampling, and explicit group counts. If duplicates or corrupt data are found, stop and document a revised frozen manifest; do not silently drop records to preserve the nominal counts.

The upstream repository describes five-second mono 44.1 kHz WAVs and distinguishes ESC-10's CC BY terms from the larger collection's noncommercial terms. Preserve pinned per-clip attribution and original source links from its LICENSE file. Confirm the exact selected-clip terms at build time. This fixed split is a teaching protocol, not the official five-fold benchmark average. Upstream also documents class-related preprocessing artifacts; report that limitation and do not claim pretraining overlap has been ruled out. [Dataset documentation and license](https://github.com/karolpiczak/ESC-50)

Build a manifest of every selected WAV's immutable URL, byte count, SHA-256, source ID, fold, category, role, format, frame count and attribution. Pin CSV and license hashes too. Download data files only, not repository source or the whole repository archive. Audio bytes and runtime estimates remain unmeasured until the build freezes and verifies the assets. Downloading and hashing the full sample is a build acceptance check, not a claim made here.

## 4. Audio and feature contract

Default decoding: WAV PCM to float32 mono in `[-1,1]`, then deterministic resampling to 16 kHz using the pinned torchaudio implementation. Verify file headers and allocation ceilings before decoding. Record original and resulting sample counts, sample rates, channel policy and hashes. Do not peak-normalize, trim silence or time-stretch clips silently. Downmix BYOD stereo by arithmetic mean and disclose it; reject more than two channels.

Use the pinned AST extractor's exact configuration, including filterbank framing, normalization and padding to 1,024 frames. Export the extractor configuration. Five-second defaults fit the model window; do not invent content in the padded portion or call it observed silence. Show a valid-time/padding boundary on the model-input visualization. Longer-than-window BYOD is rejected by this learning unit rather than silently truncated.

Show waveform in seconds, a log-mel representation with clearly labelled axes and a short explanation of frequency versus time. Distinguish a human-readable pre-normalization spectrogram from the actual normalized model tensor. Include audio playback with **no autoplay**, low-volume guidance and `normalize=False` so playback does not conceal amplitude differences in experiments. The notebook must remain understandable without listening; captions and plots carry the learning point.

Caches may contain frozen-backbone pooled features only. Preserve backbone evaluation mode and inference-only feature extraction. Convert cached features into ordinary detached tensors before gradient training, avoiding inference-tensor backward errors. Bind caches to input hashes, model revision, extractor parameters, code and dtype. Training never updates embeddings, backbone normalization, attention blocks or processor state.

## 5. Baselines and output semantics

Use three directly comparable ten-class systems:

1. **Training-majority predictor:** calculate the majority from training labels only. The default split is balanced; break ties by the fixed class order, disclose the rule and measure the result rather than implying an informed classifier.
2. **Seeded initial head:** frozen AST features with the new random classifier before any updates. Call this an untrained task head, not pretrained ten-class classification.
3. **Validation-selected trained head:** same backbone, feature pipeline and vocabulary, differing only in head parameters.

An optional introductory display runs the unchanged AudioSet head on a small fixed training-only selection. Show its top five labels and independent sigmoid scores, explicitly uncalibrated. Do not map them to ESC-10 by fuzzy strings, compare those scores directly with softmax scores, or report ten-class accuracy without a separately specified ontology mapping. A ten-class softmax always assigns one of its known classes; it does not establish absence of unknown sounds.

## 6. Training and checkpoint selection

Proposed frozen recipe: seed 42; ten epochs; batch size four; AdamW learning rate `1e-3`, weight decay `0.01`; cross-entropy; float32; no augmentation, class reweighting or learning-rate scheduler. Head parameters include the classifier's LayerNorm and linear layer; enumerate exact trainable names/counts. Feature extraction is bounded at batch four, followed by cached-head training. Expected optimizer updates: `10 * ceil(240/4) = 600`.

Initialize the ten-class head once. Evaluate epoch zero. At each epoch compute validation macro-F1 from unrounded counts, retaining a snapshot if it strictly improves; earliest epoch wins ties, including epoch zero. Use one continuous optimizer across epochs and record the deterministic shuffle algorithm. Restore the selected snapshot before any final test evaluation. Test labels must not influence caching, selection, training or the noise recipe. Do not try alternative learning rates after inspecting test performance.

Record actual optimizer steps, finite losses, head weight changes before selection, frozen-backbone hashes before/after, epoch metrics and selected epoch. Nonzero updates are required; a selected epoch-zero head may legitimately have zero final delta. Do not equate training activity with improved accuracy. Report precision, peak GPU memory, synchronized stage time, feature-cache time and setup/download time separately. Change resource defaults only through an explicit revised recipe and fresh qualification.

## 7. Evaluation and interpretation

Primary selection and reporting metric: **macro-F1 across the fixed ten labels**. Also report accuracy, per-class precision/recall/F1/support, raw confusion counts and a row-normalized confusion matrix. Use top-one argmax with a stable lowest-index tie. Metrics aggregate clips; report source-recording counts and warn that multiple excerpts from one source are correlated. Do not infer independence from clip count alone.

Define precision `TP/(TP+FP)`, recall `TP/(TP+FN)` and F1 `2TP/(2TP+FP+FN)`. A supported class with no correct predictions receives F1 zero. Undefined precision/recall display null plus a reason; do not invent a perfect score for an unpredicted class. Every class must be represented in default validation/test. Missing BYOD class support fails the declared closed-set evaluation rather than silently changing the macro denominator.

Export full ten-class logits and probabilities, true and predicted labels, class order and per-record group/fold identities. Compare initial versus selected head on the same test clips: corrected errors, new errors, unchanged correct and unchanged incorrect. Include a fixed example, a representative confusion pair and highest-confidence mistakes selected by a disclosed deterministic rule. If a category has no errors, say so; do not substitute a fabricated failure.

The worked example should demonstrate why 90% accuracy on a hypothetical 90/10 split can hide zero recall for the minority class. State that this toy arithmetic is explanatory, not a result from the balanced ESC-10 sample. Softmax scores are not calibrated reliability estimates; this unit claims neither calibration nor out-of-distribution detection.

## 8. Controlled additive-noise activity

After locking selection, choose two validation clips per class, from distinct `src_file` groups where available, by frozen filename order. Freeze the resulting 20 IDs before model evaluation. This reuses validation for a descriptive activity; it supplies no additional held-out test evidence and cannot change the selected checkpoint.

For each clip `x`, create one seeded zero-mean Gaussian noise vector `z`, normalize it to unit RMS, and reuse it at all noise levels. Seed from a fixed seed plus a cryptographic digest of the clip ID, never Python's randomized `hash()`.

Let `r = sqrt(mean(x**2))`. For SNR `s` in 20, 10 and 0 dB, set `n_s = z * r * 10**(-s/20)`. A clean condition has no added noise. RMS uses the entire five-second clip, including quiet regions; do not describe it as speech-active SNR. If `r <= 1e-8`, fail with a silence explanation rather than fabricate an SNR.

Prevent clipping with one common per-clip gain across **all** four conditions: `g = min(1, 0.99 / max(peak(x), peak(x+n_20), peak(x+n_10), peak(x+n_0)))`. Score `g*x` and `g*(x+n_s)`. The control is therefore the gain-matched clean signal, not an unmatched original amplitude. Log gain, actual SNR, peak, noise/input hashes and zero clipped samples. Never normalize each condition independently.

Keep model, labels, preprocessing, duration, noise realization and class order fixed. Recompute features from each modified waveform; reusing clean cached embeddings would invalidate the experiment. Report macro-F1/accuracy by condition and paired changed predictions versus gain-matched clean. Provide matching playback and spectrograms. Say **synthetic additive noise** rather than implying real street, microphone or room-noise robustness. Improvement at a noisy condition is a valid observation, not a reason to rerun or alter the noise seed.

## 9. Guided notebook sequence

Every substantive experiment follows orient → Input → Model/System → Output → predict → run → notice → interpret → change one thing → compare → conclude with evidence and limitations.

1. Orientation, sound classification versus transcription, closed-set limits and runtime requirements.
2. Anonymous pinned acquisition, audio/header validation and source-disjoint split table.
3. Listen to fixed examples, view waveforms/spectrograms and inspect model padding.
4. Explain AudioSet multi-label outputs versus the new ten-label task.
5. Majority and seeded-head baselines; worked confusion/metric example.
6. Predict adaptation outcomes; extract frozen features, train and select the checkpoint.
7. Evaluate test clips, per-class outcomes, confusion pairs and paired errors.
8. Export and reload the selected head in a fresh process; verify raw predictions and metrics.
9. Predict noise sensitivity, run the controlled activity and explain non-monotonic outcomes.
10. Export the evidence bundle and complete an optional conclusion with limitations.
11. Optional labelled BYOD full workflow; optional unlabelled clip inference.

Use compact tables and short summaries in the learner flow. Keep hashes, detailed traces and full per-clip outputs in files or collapsible sections. Never require learners to read an embedded source payload to interpret results.

## 10. Artifact and evidence contract

Export only the selected classifier parameters and compatible adaptation metadata in the existing `.pt` format. Include a separate JSON manifest with artifact SHA-256, base snapshot identity, extractor identity, ordered class names, softmax activation, training recipe and selection result. Retain the base weight manifest; never bundle base weights.

In a new process, verify digest and lineage, use `weights_only=True`, validate exact expected tensor names/shapes and finiteness, then reconstruct the selected head. Refuse wrong vocabulary order, base revision, activation, tensor schema, nonfinite values or corrupted bytes before applying weights. Do not fall back to unrestricted pickle loading. Round-trip compatibility with the existing repository loader must be tested; any stricter wrapper checks must be documented.

Run fresh waveform→features→head inference on fixed validation probes and all test clips. Require logits `allclose(atol=1e-5, rtol=1e-4)`, probabilities `atol=1e-6, rtol=1e-4`, identical top-one predictions and identical confusion counts; recomputed accuracy/macro-F1 tolerance `1e-12`. Preserve maximum deviations. Fixed tolerances cannot be relaxed to conceal a failed run.

Bundle explicit allowlisted files: data/split and attribution manifests, configuration, full per-clip scores, confusion matrices, metrics, training/selection/update evidence, noise records, reload checks, runtime/source/dependency identities, adapter and its manifest, run summary and checksums. Reload exported tables and verify archive hashes. Public examples may have separately attributed display/audio files; exclude private BYOD waveforms and spectrograms from shareable archives by default. Record all implicit data movement clearly.

Stage receipts bind source, model, preprocessing, data and outputs. Reruns invalidate descendants. Missing stages, changed files or failed integrity checks prevent a completed-current-run claim. A successful run is evidence for maintainer review, not automatic release approval.

## 11. BYOD and execution constraints

Labelled BYOD: directory with `manifest.json`, `class_names` and records containing `id`, relative WAV `audio`, class `label`, `group_id`, and explicit `train`/`validation`/`test` role. Accept a user-defined vocabulary of 2–10 classes; each class must have at least 4/2/2 clips in train/validation/test and two independent recording groups per class per role. Validate exact vocabulary ownership, group disjointness and decoded/resampled duplicates before extracting features.

Proposed ceilings: at most 400 clips; WAV PCM16 or IEEE float32; 8–48 kHz; one or two channels; 0.5–10 seconds; 8 MiB per file; 1 MiB manifest; 4,000 seconds total duration; finite normalized samples. Inspect format/frame counts before allocating, use contained relative paths, and refuse archives, unsupported encodings, traversal, truncated files, empty classes, overlapping groups, all-silent clips and amplitude violations. Record downmix/resampling explicitly. The sample's five-second policy is retained for the default path; BYOD padding is documented.

BYOD repeats the same local validation → baselines → head adaptation → validation selection → test → reload → controlled activity → export, with dynamically derived step counts and fresh feature caches. Keep a separate run directory. A real representative labelled dataset must complete this path before release; synthetic validation fixtures do not qualify it.

Optional unlabelled inference may show both unchanged AudioSet tags and selected custom-class probabilities in separately labelled tables. No ground truth means no accuracy/F1 claim. No unknown-class rejection is implied. Default Run all never opens an upload dialog or requests authentication.

The portable notebook embeds reviewed reference code, manifests, licenses and a hash-locked environment. No runtime Git clone, repository-source download, installed DIMER-package dependency, DIMER worker or external inference API. Use an isolated Python 3.12 worker without kernel replacement/manual restart. Public defaults need no HF token; optional access uses Colab Secrets without printing or exporting credentials. Local checks are CPU-only; model qualification runs in hosted Colab/Kaggle. Pin and verify a mutually compatible torch/torchaudio pair and filterbank behavior rather than copying mismatched dependency versions from older files.

## 12. Acceptance criteria and disclosure

- [ ] Generator and notebook implement every default stage with optional branches disabled.
- [ ] All 400 real WAVs, metadata and attribution are pinned and verified; 240/80/80 roles and source-group disjointness pass.
- [ ] CPU tests cover audio headers/bounds, resampling, duplicate/group leakage, metric denominators, stable ties, label mapping, fixed noise/SNR/common gain and negative BYOD cases.
- [ ] Tests verify checkpoint selection/restoration, actual update accounting, cache identity and frozen-backbone invariants without substituting synthetic evidence for hosted model runs.
- [ ] Notebook schema, embedded compilation, source parity, stage invalidation and corrupted-artifact refusal checks pass.
- [ ] Fresh hosted default Run all completes real feature extraction, 600 optimizer steps, selection, test, reload and noise activity without manual intervention.
- [ ] Numerical reload parity, table/archive checksums, measured timing/memory and all actual errors are retained.
- [ ] Real labelled BYOD full workflow passes; privacy and optional learner-record wording are visible.
- [ ] Exact notebook SHA, source revisions, environment and evidence are recorded before any release-grade designation.

AI Assistance Disclosure: This notebook's code and explanations were developed with generative AI assistance under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results and making release decisions. AI assistance does not constitute independent verification, provider endorsement or release approval.

Build uncertainties to resolve: exact per-file download sizes and terms; compatible hosted audio dependencies; cached-feature gradient semantics; runtime of all 400 clips; numerical reload parity; class-related source preprocessing and pretraining contamination limits. These are validation obligations, not reasons to invent sample results.

No implementation, commit, push, PR, merge or hosted run is performed by this specification task.
