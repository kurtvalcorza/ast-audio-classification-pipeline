"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "ast_audio_classification_pipeline",
    "repo_name": "ast-audio-classification-pipeline",
    "stem": "ast_audio_classification",
    "notebook_name": "ast_audio_classification_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "ASTAudioClassificationPipeline",
    "weights_key": "ast-audioset",
    "runtime_imports": ["torch", "torchaudio", "transformers"],
    "title": "AST AudioSet — DIMER audio event classification tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/ast-audio-classification-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/ast-audio-classification-pipeline/blob/main/tutorials/ast_audio_classification_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MIT%2Fast--finetuned--audioset-ffcc4d?style=flat",
            "https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-YuanGongND%2Fast-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/YuanGongND/ast",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2104.01778-b31b1b.svg", "https://arxiv.org/abs/2104.01778"),
    ],
    "capability": "multi-label audio event classification over the 527 AudioSet labels using the pinned `MIT/ast-finetuned-audioset-10-10-0.4593` weights",
    "intro": (
        "At inference the clip is resampled to 16 kHz, turned into a 128-bin Kaldi filterbank with a 10 ms hop, padded "
        "or cropped to the model's 1024-frame (10.24 s) window, and passed through the spectrogram transformer; the "
        "pipeline applies an independent sigmoid per label and returns the top-k labels with their scores. "
        "**No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting happens "
        "in this notebook — the upstream checkpoint supplies the weights, the feature extractor configuration and the "
        "label space, and the carried pipeline module adds snapshot verification, input validation, resampling, a fixed "
        "output contract and the `validate_inputs` and `evaluation_report` helpers. The default sample is a synthetic "
        "tone generated in code; its ranking is demonstration (plumbing) evidence, not a production-quality or "
        "benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, generate a synthetic tone and validate it into an input manifest, run the "
        "supported task, read multi-label sigmoid scores correctly (no threshold, not probabilities), exercise an "
        "optional BYOD WAV path, produce an evaluation report that is honestly `not-measurable` and says what labelled "
        "audio would make the task measurable, and export machine-readable outputs plus provenance."
    ),
    "exclusions": (
        "speech transcription, speaker identification, temporal localisation of events inside the window, source "
        "separation, audio generation, or any training. The label space is fixed to the 527 AudioSet classes; a sound "
        "outside that space still receives ranked labels."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. The pinned `torch==2.14.0` install and the 346 MB checkpoint are the largest downloads of the run.",
        "- **Knowledge:** basic Python and NumPy; what a mel/filterbank spectrogram is, and why an independent sigmoid per label is not a probability distribution.",
        "- **Data:** the default sample is a deterministic 3 s, 440 Hz sine tone generated in code at 16 kHz, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one PCM WAV file (8/16/32-bit), mono or stereo, between 0.025 s and 120 s; it is decoded with the standard-library `wave` module, scaled to `[-1, 1]`, averaged to mono, and resampled to 16 kHz by the pipeline. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Generate the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: a deterministic 3 s, 440 Hz sine tone at amplitude 0.5, generated "
                "in code at the model's 16 kHz rate as a float32 mono array (the same kind of input the repository's "
                "smoke run used), so it needs no download and its SHA-256 is printed for the record. A pure tone is not "
                "a recording of any real acoustic event, so it has **no ground truth** and whatever ranking the model "
                "returns is a sanity check that the input contract, feature extraction and forward pass work — not a "
                "correctness measurement. BYOD is optional and disabled by default; when enabled, upload one PCM WAV "
                "file. It is decoded with the standard-library `wave` module (no extra decoder is pinned), integer "
                "samples are scaled to `[-1, 1]`, stereo is averaged to mono, and the original sample rate is passed to "
                "the pipeline, which resamples to 16 kHz with `torchaudio.functional.resample` — resampling cannot "
                "restore content above the original Nyquist frequency."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import wave\n\n"
                "import numpy as np\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "TONE_SECONDS = 3.0\n"
                "TONE_HZ = 440.0\n"
                "TONE_AMPLITUDE = 0.5\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    clip_name = next(iter(uploaded))\n"
                "    with wave.open(io.BytesIO(uploaded[clip_name]), 'rb') as handle:\n"
                "        channels, sample_width, sample_rate, frames = handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()\n"
                "        raw = handle.readframes(frames)\n"
                "    if sample_width == 1:\n"
                "        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0\n"
                "    elif sample_width == 2:\n"
                "        samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0\n"
                "    elif sample_width == 4:\n"
                "        samples = np.frombuffer(raw, dtype='<i4').astype(np.float32) / 2147483648.0\n"
                "    else:\n"
                "        raise ValueError(f'{{clip_name}}: {{8 * sample_width}}-bit PCM is not supported here; convert the file to 16-bit PCM WAV and rerun this cell.')\n"
                "    audio = samples.reshape(-1, channels).mean(axis=1).astype(np.float32) if channels > 1 else samples\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic tone: no randomness, so no seed is needed and the digest is stable.\n"
                "    sample_rate = SAMPLE_RATE\n"
                "    t = np.arange(int(TONE_SECONDS * sample_rate)) / sample_rate\n"
                "    audio = (TONE_AMPLITUDE * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)\n"
                "    clip_name = f'synthetic_sine_{{int(TONE_HZ)}}hz_{{int(TONE_SECONDS)}}s.wav'\n"
                "    sample_kind = 'synthetic'\n\n"
                "audio_sha256 = hashlib.sha256(audio.tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': clip_name, 'samples': int(audio.shape[0]), 'sample_rate': sample_rate, 'dtype': str(audio.dtype), 'float32_sha256': audio_sha256}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the clip → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `predict` "
                "applies, through the same private check, so the two cannot diverge — 1-D float array, positive integer "
                "sample rate, finite samples, duration between `MIN_AUDIO_SECONDS` and `MAX_INPUT_SECONDS`, `top_k` in "
                "1..`NUM_LABELS`. It returns an **input manifest** naming the schema and ceilings, each clip's "
                "identifier, sample count, duration, peak amplitude, and whether it will be resampled or truncated, "
                "written to `outputs/{stem}_input_manifest.json`. The ceilings are printed first, before any model "
                "work. A clip above `MAX_INPUT_SECONDS` (120 s) is **rejected** — chunk it first — while a clip longer "
                "than the 10.24 s model window is **accepted** but **only its first 10.24 s reach the model**: the "
                "manifest flags `will_truncate` and the result carries `truncated: True`. Clips shorter than one 25 ms "
                "filterbank frame are rejected. To show what rejection looks like, the cell also validates a "
                "deliberately over-long clip and records the pipeline's own error message as a finding. Nothing else is "
                "dropped or altered."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'SAMPLE_RATE': SAMPLE_RATE, 'MIN_AUDIO_SECONDS': MIN_AUDIO_SECONDS, 'MAX_AUDIO_SECONDS': MAX_AUDIO_SECONDS, 'MAX_INPUT_SECONDS': MAX_INPUT_SECONDS, 'NUM_LABELS': NUM_LABELS}}}})\n"
                "input_manifest = validate_inputs(audio, sample_rate, top_k=5, names=[clip_name])\n"
                "# Demonstrate rejection on a clip that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(np.zeros(int((MAX_INPUT_SECONDS + 1) * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'over-long-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "if input_manifest['inputs'][0]['will_truncate']:\n"
                "    print(f\"NOTE: {{clip_name}} is longer than the {{MAX_AUDIO_SECONDS}} s model window; only its first {{MAX_AUDIO_SECONDS}} s reach the model and the result is flagged truncated.\")\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Classify and read the scores correctly\n\n"
                "`predict` returns a `predictions` list of `{{label, index, score}}` entries **ordered by descending "
                "score** — rank position is the label ordering, and the exported files preserve it — plus `activation` "
                "(`sigmoid`), `truncated`, `duration_seconds`, `window_seconds`, `resampled`, `input_sample_rate`, and "
                "the model identity. Each `score` is an **independent sigmoid** of that label's logit: this is "
                "multi-label classification, so the scores do not sum to one, several labels can be high at once, and a "
                "score is **not a calibrated probability** (the head was trained with binary cross-entropy on weak, "
                "incomplete AudioSet labels). **No decision threshold is shipped**: `top_k` (default 5, ceiling 527) is "
                "a presentation choice, not an acceptance rule, and no label is asserted present or absent. The "
                "operator owns the threshold and should set it per label from precision-recall curves on their own "
                "labelled clips. Inference is deterministic on a fixed device and dtype (no sampling, `model.eval()`, "
                "`torch.inference_mode`); CUDA kernel selection and the resampler can shift scores in the third or "
                "fourth decimal place across hardware. As recorded in the model card, the repository's smoke run on "
                "this same tone (CUDA, float32, verified snapshot) ranked `Sine wave` first at score 0.84; that is one "
                "observation for a synthetic tone and sanity evidence only — a materially different top label on your "
                "runtime is a signal to check the install, not a measurement of anything."
            ),
            "code": (
                "result = pipe.predict(audio, sample_rate=sample_rate, top_k=5)\n"
                "print({{'activation': result['activation'], 'truncated': result['truncated'], 'resampled': result['resampled'], 'duration_seconds': round(result['duration_seconds'], 3), 'window_seconds': result['window_seconds'], 'input_sample_rate': result['input_sample_rate'], 'device': pipe.device}})\n"
                "for rank, item in enumerate(result['predictions'], start=1):\n"
                "    print(f\"{{rank:>2}}. index {{item['index']:>3}}  score {{item['score']:.4f}}  {{item['label']}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. Its "
                "verdict here is **always `not-measurable`**: this repository ships no metric helper, and a synthetic "
                "tone has no ground truth, so there is nothing honest to report as a number. The report names the score "
                "semantics (independent sigmoids, not probabilities, no threshold) and states what a real evaluation "
                "needs: clips labelled against the same 527-label ontology, the full score vector "
                "(`predict(..., top_k=527)`), and mean average precision plus per-label precision and recall at a "
                "threshold chosen on your own labelled clips. The `0.4593` in the checkpoint name is the "
                "upstream-reported AudioSet mAP for this configuration and is not measured here. The report is written "
                "to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is computed: the pipeline ships no metric helper and the sample has no ground truth; the ranking above is sanity evidence only.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full result (rank-ordered sigmoid scores, activation, truncation "
                "and resampling flags), the input manifest, the evaluation report, the clip identity and digest, the "
                "notebook's source (repository, revision, embedded module digest, generator), the model identifier, the "
                "immutable model revision, the model licence, and the runtime identity (Python, `torch`, `torchaudio`, "
                "`transformers`, device). The rank-ordered top-k table is also written as CSV with explicit `clip`, "
                "`rank`, `index`, `label` and `score` columns so label ordering survives downstream use. No credentials "
                "are recorded."
            ),
            "code": (
                "import csv\n\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': clip_name, 'samples': int(audio.shape[0]), 'sample_rate': sample_rate, 'float32_sha256': audio_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'torchaudio': torchaudio.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_top_k.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['clip', 'rank', 'index', 'label', 'score'])\n"
                "    for rank, item in enumerate(result['predictions'], start=1):\n"
                "        writer.writerow([clip_name, rank, item['index'], item['label'], f\"{{item['score']:.6f}}\"])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The ranked labels are independent sigmoid scores over the fixed 527-label AudioSet ontology; they are not "
        "probabilities of presence, they do not sum to one, and the pipeline ships no threshold, so nothing in this "
        "notebook asserts that an event is present or absent. On the synthetic tone the ranking is sanity evidence by "
        "construction and the evaluation report says `not-measurable`; a ranking shown for a BYOD clip is a single-clip "
        "observation for that recording and must not be generalized to a domain, microphone, or acoustic environment. "
        "Only the first 10.24 s of a clip reach the model, resampling from other rates loses content above the original "
        "Nyquist frequency, and sounds outside the ontology still receive some ranked label. The pipeline provides no "
        "transcription, speaker identity, temporal localisation, source separation, or training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated input, execute the public pipeline "
        "path, and emit the shown machine-readable outputs in the tested runtime — without the repository being "
        "reachable. It does **not** establish benchmark superiority, deployment calibration, safety for "
        "high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** enable `USE_BYOD` with a short real recording (a door closing, a dog barking) and "
        "compare how the sigmoid scores spread across related ontology labels; change `TONE_HZ` and watch which "
        "tone-like labels (`Sine wave`, `Dial tone`, `Beep, bleep`) move; upload a clip longer than 10.24 s and confirm "
        "that the input manifest flags `will_truncate` and the result carries `truncated: True`, then chunk it yourself "
        "and score each window separately.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/ast-audio-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/ast-audio-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/ast-audio-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/YuanGongND/ast\n"
        "- AST: Audio Spectrogram Transformer (Gong, Chung, Glass, 2021): https://arxiv.org/abs/2104.01778\n"
        "- AudioSet ontology: https://research.google.com/audioset/"
    ),
}
