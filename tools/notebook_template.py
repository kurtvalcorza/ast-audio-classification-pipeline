"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "ast_audio_classification_pipeline",
    "repo_name": "ast-audio-classification-pipeline",
    "stem": "ast_audio_classification",
    "notebook_name": "ast_audio_classification_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned AST snapshot, demonstrates the unchanged 527-label AudioSet head, generates and validates the deterministic "
        "24-clip acoustic-ecology dataset, creates a seeded disjoint split, re-heads the classifier onto three classes, freezes "
        "the transformer backbone, measures the pre-adaptation baseline, runs the bounded classifier-head fine-tune, evaluates "
        "the held-out clips, predicts an unseen generated clip, exports the classifier-head adapter, reloads it over a fresh "
        "base-model instance, verifies numeric parity, and writes machine-readable outputs. The default path needs no repository "
        "clone, DIMER worker or service, credential, upload dialog, or configuration edit (NOTEBOOK_SPEC 2.0 §5, RUN7, FT2)."
    ),
    "byod": (
        "Two optional branches are disabled by default. `USE_BYOD = True` accepts one 16 kHz PCM WAV for the unchanged AudioSet "
        "inference demonstration. `USE_BYOD_DATASET = True` accepts one ZIP whose top-level directories are exactly `geophony/`, "
        "`biophony/`, and `anthrophony/`, with at least two 16 kHz mono or multichannel PCM WAV files per class; each clip must be "
        "0.025–10.24 s. The ZIP branch applies explicit archive limits, decodes to mono float32, then enters the same validation, "
        "seeded split, local adaptation, evaluation, export, and reload path as the generated dataset. Uploads stay in this runtime."
    ),
    "pipeline_class": "ASTAudioClassificationPipeline",
    "weights_key": "ast-audioset",
    "runtime_imports": ["torch", "torchaudio", "transformers"],
    "modules": ["metrics.py", "samples.py", "pipeline.py"],
    "entry_module": "pipeline.py",
    "title": "AST Audio Spectrogram Transformer — DIMER Acoustic Ecology E2E Tutorial (Standalone)",
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
    "capability": "pretrained AudioSet classification & supervised in-kernel acoustic ecology adaptation with `MIT/ast-finetuned-audioset-10-10-0.4593`",
    "intro": (
        "The Audio Spectrogram Transformer (AST) applies a Vision Transformer (ViT) architecture directly to audio "
        "spectrograms. At inference, input audio is resampled to 16 kHz, converted into a 128-bin Kaldi filterbank with "
        "a 10 ms hop, padded or cropped to 1024 frames (10.24 s), and processed by the transformer backbone. "
        "**In-kernel fine-tuning:** this tutorial demonstrates both pretrained 527-class AudioSet event inference and "
        "end-to-end supervised adaptation to a custom acoustic ecology classification task (`ADAPT_CLASSES = ('geophony', 'biophony', 'anthrophony')`). "
        "The backbone is frozen and the classifier head is dynamically re-headed (`Linear(768, 3)`), training only ~3.8k parameters "
        "with bounded AdamW in ~10–15 s on CPU with zero external worker repositories. "
        "Quantitative evaluation against a majority-class baseline and portable adapter export (`outputs/ast-audio-adapter-v1.pt`) "
        "complete the end-to-end adaptation workflow."
    ),
    "learning_objectives": (
        "install the pinned runtime, verify the immutable upstream AST checkpoint against its SHA-256 manifest, demonstrate "
        "pretrained AudioSet inference on a synthetic tone, generate and validate an in-code acoustic ecology dataset conforming to "
        "the owner-namespaced `io.github.kurtvalcorza.dataset.audio.waveform-classification.v1` representation, perform a stratified "
        "75/25 split, dynamically re-head the classifier and freeze the backbone, "
        "execute an in-process AdamW fine-tuning loop, quantitatively evaluate accuracy and macro-F1 against a majority baseline, "
        "export a portable adapter artifact, and verify safe reloading across an isolated boundary."
    ),
    "exclusions": (
        "speech transcription, speaker identification, temporal localisation of events inside the window, source "
        "separation, or unconstrained multi-hour training. Adaptation is bounded to classifier head adaptation."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). Runs on CPU or CUDA automatically; float32 on both.",
        "- **Knowledge:** basic Python and PyTorch; familiarity with acoustic spectrograms and classification baselines.",
        "- **Data:** the default path is 100% self-contained. Optional single-clip BYOD accepts one PCM WAV. Optional adaptation BYOD accepts one ZIP with exactly `geophony/`, `biophony/`, and `anthrophony/` top-level directories and at least two 16 kHz PCM WAV files per class; clips must be 0.025–10.24 s. The ZIP is capped at 100 files, 64 MiB compressed, and 256 MiB expanded. Do not upload confidential or restricted audio to a hosted runtime unless authorized.",
    ],
    "cells": [
        # ---------------------------------------------------------------- 4. Confirm runtime
        {
            "md": (
                "## 4. Confirm the qualified runtime environment\n\n"
                "The carried package verifies runtime dependency versions (`torch`, `torchaudio`, `transformers`) and prints "
                "the execution device alongside input ceilings. Look for confirmed versions and device."
            ),
            "code": (
                "import platform\n"
                "import numpy as np\n"
                "import torch\n"
                "import torchaudio\n"
                "import transformers\n\n"
                "print({{\n"
                "    'python': platform.python_version(),\n"
                "    'torch': torch.__version__,\n"
                "    'torchaudio': torchaudio.__version__,\n"
                "    'transformers': transformers.__version__,\n"
                "    'device': pipe.device,\n"
                "    'ceilings': {{\n"
                "        'SAMPLE_RATE': SAMPLE_RATE,\n"
                "        'MIN_AUDIO_SECONDS': MIN_AUDIO_SECONDS,\n"
                "        'MAX_AUDIO_SECONDS': MAX_AUDIO_SECONDS,\n"
                "        'MAX_INPUT_SECONDS': MAX_INPUT_SECONDS,\n"
                "        'NUM_LABELS': NUM_LABELS,\n"
                "    }},\n"
                "}})"
            ),
        },
        # ---------------------------------------------------------------- 5. Pretrained demonstration
        {
            "md": (
                "## 5. Demonstrate pretrained AudioSet capability\n\n"
                "Before adapting to custom classes, this cell exercises the base model on a deterministic 440 Hz sinusoidal "
                "tone (or optional uploaded BYOD WAV). `predict` runs the clip through the pretrained 527-class head with independent "
                "sigmoid scores. Each score is **not a calibrated probability**, the scores do not sum to one, and no decision "
                "threshold is shipped. Look for top predicted AudioSet classes."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import wave\n"
                "import numpy as np\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "TONE_SECONDS = 3.0\n"
                "TONE_HZ = 440.0\n"
                "TONE_AMPLITUDE = 0.5\n\n"
                "def decode_pcm_wav(payload):\n"
                "    with wave.open(io.BytesIO(payload), 'rb') as handle:\n"
                "        channels = handle.getnchannels()\n"
                "        sample_width = handle.getsampwidth()\n"
                "        decoded_rate = handle.getframerate()\n"
                "        frames = handle.getnframes()\n"
                "        compression = handle.getcomptype()\n"
                "        raw = handle.readframes(frames)\n"
                "    if compression != 'NONE':\n"
                "        raise ValueError(f'compressed WAV is unsupported: {{compression}}')\n"
                "    if sample_width == 1:\n"
                "        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0\n"
                "    elif sample_width == 2:\n"
                "        samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0\n"
                "    elif sample_width == 4:\n"
                "        samples = np.frombuffer(raw, dtype='<i4').astype(np.float32) / 2147483648.0\n"
                "    else:\n"
                "        raise ValueError(f'{{8 * sample_width}}-bit PCM WAV is unsupported; convert to 16-bit PCM')\n"
                "    if channels < 1 or samples.size % channels:\n"
                "        raise ValueError('invalid WAV channel layout')\n"
                "    mono = samples.reshape(-1, channels).mean(axis=1) if channels > 1 else samples\n"
                "    return mono.astype(np.float32), decoded_rate\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError('upload exactly one PCM WAV file')\n"
                "    clip_name = next(iter(uploaded))\n"
                "    audio, sample_rate = decode_pcm_wav(uploaded[clip_name])\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    sample_rate = SAMPLE_RATE\n"
                "    audio = tutorial_tone(duration=TONE_SECONDS, sample_rate=sample_rate, frequency=TONE_HZ, amplitude=TONE_AMPLITUDE)\n"
                "    clip_name = f'synthetic_sine_{{int(TONE_HZ)}}hz_{{int(TONE_SECONDS)}}s.wav'\n"
                "    sample_kind = 'synthetic'\n\n"
                "audio_sha256 = hashlib.sha256(audio.tobytes()).hexdigest()\n"
                "result = pipe.predict(audio, sample_rate=sample_rate, top_k=5)\n"
                "print({{\n"
                "    'sample_kind': sample_kind,\n"
                "    'name': clip_name,\n"
                "    'samples': int(audio.shape[0]),\n"
                "    'sample_rate': sample_rate,\n"
                "    'sha256': audio_sha256[:16] + '...',\n"
                "    'top_label': result['predictions'][0]['label'],\n"
                "    'top_score': round(result['predictions'][0]['score'], 4),\n"
                "    'activation': result['activation'],\n"
                "    'truncated': result['truncated'],\n"
                "}})\n"
                "for rank, item in enumerate(result['predictions'], start=1):\n"
                "    print(f\"  {{rank:>2}}. index {{item['index']:>3}}  score {{item['score']:.4f}}  {{item['label']}}\")"
            ),
        },
        # ---------------------------------------------------------------- 6. Custom dataset & validation
        {
            "md": (
                "## 6. Generate and validate the custom acoustic ecology dataset\n\n"
                "The adaptation task targets the canonical Krause/Pijanowski tripartite acoustic ecology classification: "
                "`ADAPT_CLASSES = ('geophony', 'biophony', 'anthrophony')`. "
                "The dataset generator produces 24 16 kHz mono 3.0 s clips conforming to the owner-namespaced "
                "`io.github.kurtvalcorza.dataset.audio.waveform-classification.v1` representation; no stable `core.*` audio "
                "representation exists in the current DIMER fleet specification. "
                "`validate_dataset` checks audio integrity, duration limits, and class labels. The input manifest is written to "
                "`outputs/{stem}_input_manifest.json` along with a rejected over-long ceiling probe finding."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "import zipfile\n"
                "from pathlib import PurePosixPath\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "USE_BYOD_DATASET = False  # @param {{type:\"boolean\"}}\n\n"
                "if USE_BYOD_DATASET:\n"
                "    from google.colab import files\n"
                "    uploaded_dataset = files.upload()\n"
                "    if len(uploaded_dataset) != 1:\n"
                "        raise ValueError('upload exactly one ZIP dataset')\n"
                "    dataset_name, dataset_bytes = next(iter(uploaded_dataset.items()))\n"
                "    if not dataset_name.lower().endswith('.zip') or len(dataset_bytes) > 64 * 1024**2:\n"
                "        raise ValueError('dataset must be one ZIP no larger than 64 MiB compressed')\n"
                "    dataset_records = []\n"
                "    with zipfile.ZipFile(io.BytesIO(dataset_bytes)) as archive:\n"
                "        archive_members = archive.infolist()\n"
                "        members = [member for member in archive_members if not member.is_dir()]\n"
                "        if not members or len(members) > 100:\n"
                "            raise ValueError('dataset ZIP must contain 1–100 files')\n"
                "        if sum(member.file_size for member in members) > 256 * 1024**2:\n"
                "            raise ValueError('dataset ZIP expands beyond 256 MiB')\n"
                "        top_levels = set()\n"
                "        for member in archive_members:\n"
                "            path = PurePosixPath(member.filename)\n"
                "            expected_parts = 1 if member.is_dir() else 2\n"
                "            if path.is_absolute() or '..' in path.parts or len(path.parts) != expected_parts:\n"
                "                raise ValueError(f'unsafe or invalid member path: {{member.filename!r}}')\n"
                "            class_name = path.parts[0]\n"
                "            if class_name not in ADAPT_CLASSES:\n"
                "                raise ValueError(f'unknown class directory {{class_name!r}}; expected {{ADAPT_CLASSES}}')\n"
                "            top_levels.add(class_name)\n"
                "            if member.is_dir():\n"
                "                continue\n"
                "            if member.flag_bits & 0x1 or path.suffix.lower() != '.wav':\n"
                "                raise ValueError(f'member must be an unencrypted PCM WAV: {{member.filename!r}}')\n"
                "            waveform, record_rate = decode_pcm_wav(archive.read(member))\n"
                "            dataset_records.append({{\n"
                "                'id': member.filename,\n"
                "                'waveform': waveform,\n"
                "                'sample_rate': record_rate,\n"
                "                'label': ADAPT_CLASSES.index(class_name),\n"
                "                'class_name': class_name,\n"
                "            }})\n"
                "        if top_levels != set(ADAPT_CLASSES):\n"
                "            raise ValueError(f'dataset ZIP must contain exactly these class directories: {{ADAPT_CLASSES}}')\n"
                "    dataset_kind = 'BYOD'\n"
                "else:\n"
                "    dataset_records = synthetic_audio_dataset(n_samples=24, seed=42)\n"
                "    dataset_kind = 'synthetic'\n\n"
                "dataset_summary = validate_dataset(dataset_records, ADAPT_CLASSES)\n"
                "if any(count < 2 for count in dataset_summary['class_counts'].values()):\n"
                "    raise ValueError('dataset requires at least two records per class for a disjoint split')\n\n"
                "input_manifest = validate_inputs(audio, sample_rate, top_k=5, names=[clip_name])\n"
                "input_manifest['dataset_representation'] = dataset_summary['representation']\n"
                "input_manifest['dataset_records'] = dataset_summary['n_records']\n"
                "input_manifest['class_counts'] = dataset_summary['class_counts']\n\n"
                "# Demonstrate validation rejection on an over-long input\n"
                "try:\n"
                "    validate_inputs(np.zeros(int((MAX_INPUT_SECONDS + 1) * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'over-long-probe', 'verdict': 'rejected', 'message': str(exc)}})\n\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 7. Train / val partition
        {
            "md": (
                "## 7. Deterministic train/validation partition\n\n"
                "The 24 records are partitioned into a 75% train set (18 clips, 6 per class) and a 25% held-out validation set "
                "(6 clips, 2 per class) using `split_dataset`. Stratification preserves exact class balance across splits."
            ),
            "code": (
                "train_records, val_records = split_dataset(dataset_records, val_fraction=0.25, seed=42)\n"
                "train_summary = validate_dataset(train_records, ADAPT_CLASSES)\n"
                "val_summary = validate_dataset(val_records, ADAPT_CLASSES)\n\n"
                "print({{\n"
                "    'split': 'stratified_75_25',\n"
                "    'train_samples': len(train_records),\n"
                "    'val_samples': len(val_records),\n"
                "    'train_counts': train_summary['class_counts'],\n"
                "    'val_counts': val_summary['class_counts'],\n"
                "}})"
            ),
        },
        # ---------------------------------------------------------------- 8. Re-heading & pre-adaptation baseline
        {
            "md": (
                "## 8. Dynamic re-heading and pre-adaptation baseline\n\n"
                "`pipe.rehead(ADAPT_CLASSES)` dynamically swaps the 527-class AudioSet classifier head for a new 3-class linear head "
                "(`Linear(768, 3)`), and `pipe.freeze_backbone()` freezes the 85.5M backbone parameters so that only the classifier head "
                "(~3.8k parameters) is updated. Evaluating on the validation split before fine-tuning establishes the untrained baseline."
            ),
            "code": (
                "pipe.rehead(ADAPT_CLASSES, seed=42)\n"
                "frozen_params = pipe.freeze_backbone()\n\n"
                "pre_eval = pipe.evaluate(val_records)\n"
                "print({{\n"
                "    'stage': 'reheaded_pre_adaptation',\n"
                "    'classes': pipe.labels,\n"
                "    'frozen_backbone_parameters': frozen_params,\n"
                "    'pre_adaptation_accuracy': pre_eval['accuracy'],\n"
                "    'majority_baseline_accuracy': pre_eval['baseline']['majority_class_accuracy'],\n"
                "}})"
            ),
        },
        # ---------------------------------------------------------------- 9. In-process fine-tuning loop
        {
            "md": (
                "## 9. In-process bounded fine-tuning loop\n\n"
                "`pipe.finetune(...)` caches the frozen backbone features once, then executes bounded supervised adaptation "
                "using `torch.optim.AdamW` over 5 epochs (batch size 4, 25 optimizer steps total). The measured loss and "
                "held-out metrics are printed; the tutorial does "
                "not promise monotonic loss or a quality threshold on this synthetic sample."
            ),
            "code": (
                "history = pipe.finetune(\n"
                "    train_records=train_records,\n"
                "    val_records=val_records,\n"
                "    epochs=5,\n"
                "    batch_size=4,\n"
                "    learning_rate=1e-3,\n"
                "    seed=42,\n"
                ")\n\n"
                "for epoch_data in history:\n"
                "    print(f\"Epoch {{epoch_data['epoch']}}/5: train_loss={{epoch_data['train_loss']:.4f}}  val_acc={{epoch_data.get('val_accuracy', 0.0):.4f}}\")"
            ),
        },
        # ---------------------------------------------------------------- 10. Post-adaptation evaluation
        {
            "md": (
                "## 10. Post-adaptation evaluation on held-out split\n\n"
                "The adapted model is evaluated on the 6 held-out validation clips. `pipe.evaluate(val_records)` scores multiclass "
                "accuracy, macro-F1, per-class metrics, and compares against the majority-class baseline. "
                "The machine-readable report is written to `outputs/{stem}_evaluation_report.json` with verdict `sample-sanity`."
            ),
            "code": (
                "eval_report = pipe.evaluate(val_records)\n\n"
                "report_payload = {{\n"
                "    'task': 'multiclass acoustic ecology classification',\n"
                "    'score_semantics': 'softmax probability distribution over target classes',\n"
                "    'sample_kind': f'{{dataset_kind}}_heldout',\n"
                "    'n_clips': len(val_records),\n"
                "    'classes': list(ADAPT_CLASSES),\n"
                "    'metrics': eval_report,\n"
                "    'baseline': eval_report['baseline'],\n"
                "    'verdict': 'sample-sanity',\n"
                "    'reason': f\"{{len(val_records)}} held-out clip(s) evaluated against majority baseline; not a benchmark claim\",\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "}}\n\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report_payload, handle, indent=2, ensure_ascii=False)\n\n"
                "print(json.dumps(report_payload, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 11. Inference on unseen test clip
        {
            "md": (
                "## 11. Inference on unseen acoustic clip\n\n"
                "Demonstrates the adapted pipeline on a newly synthesized unseen test clip (biophony chirp whistle). "
                "The predictions are exported to `outputs/{stem}_top_k.csv` with rank-ordered softmax probabilities."
            ),
            "code": (
                "import csv\n\n"
                "test_clip = generate_audio_clip(index=99, class_idx=1, duration=3.0, sample_rate=SAMPLE_RATE)\n"
                "test_result = pipe.predict(test_clip, sample_rate=SAMPLE_RATE, top_k=3)\n\n"
                "print({{\n"
                "    'test_clip': 'unseen_synthetic_biophony_99',\n"
                "    'predicted_label': test_result['predictions'][0]['label'],\n"
                "    'predicted_score': round(test_result['predictions'][0]['score'], 4),\n"
                "    'activation': test_result['activation'],\n"
                "}})\n\n"
                "with open('outputs/{stem}_top_k.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['clip', 'rank', 'index', 'label', 'score'])\n"
                "    for rank, item in enumerate(test_result['predictions'], start=1):\n"
                "        writer.writerow(['unseen_test_clip_99', rank, item['index'], item['label'], f\"{{item['score']:.6f}}\"])\n"
                "        print(f\"  {{rank:>2}}. index {{item['index']:>2}}  score {{item['score']:.4f}}  {{item['label']}}\")"
            ),
        },
        # ---------------------------------------------------------------- 12. Portable adapter export
        {
            "md": (
                "## 12. Export portable adapter artifact\n\n"
                "`pipe.save_artifact('outputs/ast-audio-adapter-v1.pt')` persists the adapted weights, class vocabulary, "
                "adaptation configuration, and exact base-model identity to a reloadable classifier-head adapter. The frozen "
                "85M-parameter backbone is deliberately not duplicated in this file and must be reconstructed from the pinned base revision."
            ),
            "code": (
                "saved_artifact_path = pipe.save_artifact('outputs/ast-audio-adapter-v1.pt')\n"
                "artifact_size = saved_artifact_path.stat().st_size\n"
                "print({{\n"
                "    'saved_artifact': str(saved_artifact_path),\n"
                "    'bytes': artifact_size,\n"
                "    'format': ARTIFACT_FORMAT,\n"
                "}})"
            ),
        },
        # ---------------------------------------------------------------- 13. Clean reload & numeric verification
        {
            "md": (
                "## 13. Clean reload and numeric verification\n\n"
                "Verifies artifact portability across a clean boundary: instantiates a fresh pipeline and loads the adapter "
                "with safe `weights_only=True` deserialization. An assertion enforces exact numerical score equivalence."
            ),
            "code": (
                "reloaded_pipe = ASTAudioClassificationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n"
                "reloaded_pipe.load_artifact('outputs/ast-audio-adapter-v1.pt')\n\n"
                "reloaded_result = reloaded_pipe.predict(test_clip, sample_rate=SAMPLE_RATE, top_k=3)\n"
                "np.testing.assert_allclose(\n"
                "    [p['score'] for p in reloaded_result['predictions']],\n"
                "    [p['score'] for p in test_result['predictions']],\n"
                "    rtol=1e-5,\n"
                "    atol=1e-6,\n"
                ")\n"
                "print({{\n"
                "    'reload_verification': 'PASSED',\n"
                "    'original_top_score': round(test_result['predictions'][0]['score'], 6),\n"
                "    'reloaded_top_score': round(reloaded_result['predictions'][0]['score'], 6),\n"
                "    'labels': reloaded_pipe.labels,\n"
                "}})"
            ),
        },
        # ---------------------------------------------------------------- 14. Export outputs & provenance
        {
            "md": (
                "## 14. Export outputs and provenance bundle\n\n"
                "Persists the complete machine-readable bundle (`manifest`, `report`, `result`, `top_k`, `adapter`) "
                "recording runtime metadata, base model revision, and license for traceability."
            ),
            "code": (
                "payload = {{\n"
                "    'evaluation_report': report_payload,\n"
                "    'input_manifest': input_manifest,\n"
                "    'test_prediction': test_result,\n"
                "    'training_history': history,\n"
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
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The adapted classifier predicts single-label softmax probability distributions over `ADAPT_CLASSES` (`geophony`, `biophony`, `anthrophony`). "
        "The baseline comparison measures performance relative to a trivial majority-class predictor on the held-out sample. "
        "Training is bounded to the classifier head with frozen backbone parameters, providing fast and reliable in-kernel adaptation "
        "without requiring GPU compute or external dependencies.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated dataset contract, execute supervised head adaptation, "
        "evaluate metrics against a majority baseline, and emit the shown machine-readable artifacts — without the repository being reachable. "
        "It does **not** establish benchmark superiority, production fitness, or performance on real field recordings.\n\n"
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
