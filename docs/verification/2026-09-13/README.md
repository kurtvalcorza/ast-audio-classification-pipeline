# AST — Colab T4 execution evidence

Execution date (UTC): 2026-09-13. Outcome: **PASS — 8/8 unchanged code cells**, with GPU device `cuda:0`.

| Provenance | Value |
|---|---|
| Tested repository commit | `749fbf6b9a87bbf67394aad2e338252d29d5910b` |
| Source notebook Git blob | `59edb7e523cd419cd64e49624e793d9de205b033` |
| Source notebook SHA-256 | `6dfac5f8acb99a958b14091d6b42f07a40963f894c569f534dab49434c809db4` |
| Embedded source revision | `666248b65f5d8e63fb3fca763447d22da1f70bd7` |
| Model | `MIT/ast-finetuned-audioset-10-10-0.4593@f826b80d28226b62986cc218e5cec390b1096902` |
| GPU / driver | `Tesla T4, 15360 MiB, 580.82.07` |
| Runtime | `{"device": "cuda:0", "python": "3.12.3", "torch": "2.14.0+cu130", "torchaudio": "2.11.0+cu130", "transformers": "4.57.6"}` |
| Sum of code-cell wall times | 134.118 s |
| Total with environment setup and bookkeeping | 138.366 s |

## Execution method

Colab CLI 0.6.0 ran a driver from WSL `claude-science` on one Tesla T4 VM. The driver created a separate Python 3.12.3 virtual environment for this notebook and launched a new interpreter. Each original code cell was executed sequentially with `exec(compile(...))`; source cells and default form parameters were unchanged. The executor source is retained as [executor-source.txt](executor-source.txt), an evidence artifact rather than repository tooling. The VM had no repository checkout. The weights directory and isolated Hugging Face cache were empty before this notebook ran; all snapshot entries were downloaded and SHA-256 verified by the embedded pipeline.

The hosted Colab kernel used Python 3.13.15. An earlier direct `.ipynb` attempt was aborted during installation after that mismatch was confirmed. The successful result here uses the repository-supported Python 3.12 interpreter. A completed native hosted-kernel run is not claimed.

CLI transport prerequisite: PyPI `jupyter-kernel-client==1.0.2` lacked `KernelClient`; the CLI environment used Google’s fork at `f18e982c3265df5e923aa9def101ab3fd737e139` (distribution 0.8.0). This affects the CLI host, not the notebook runtime pins.

## Observations

- Sine wave: 0.841864.
- Dial tone: 0.032473.
- Chirp tone: 0.010536.
- Beep, bleep: 0.009725.
- Busy signal: 0.007299.

The generated input was a 3 s, 440 Hz float32 sine at 16 kHz. The result was neither resampled nor truncated, and the five scores were valid, sorted independent sigmoid scores. The evaluation verdict is `not-measurable`; no labelled AudioSet evaluation or mAP was computed.

## Verification and retained files

The read-back checks confirmed the exact source hash and Git blob, unchanged code cells, eight error-free executed cells, runtime pins, CUDA inference, accepted inputs, and the recorded rejection probe. Model-specific sanity checks and exported artifacts were checked after download. See [execution-record.json](execution-record.json) for the individual checks and per-cell timings.

- [Executed notebook](ast_audio_classification_colab_output.ipynb)
- [Execution log](execution.log)
- [Installed distributions](packages-after.json)
- [Artifact SHA-256 manifest](artifacts.sha256)
- [ast_audio_classification_evaluation_report.json](outputs/ast_audio_classification_evaluation_report.json)
- [ast_audio_classification_input_manifest.json](outputs/ast_audio_classification_input_manifest.json)
- [ast_audio_classification_result.json](outputs/ast_audio_classification_result.json)
- [ast_audio_classification_top_k.csv](outputs/ast_audio_classification_top_k.csv)

The CLI stopped the shared runtime after all four notebook tests; a subsequent `colab sessions` call returned no active sessions. Both cleanup outputs are retained in the execution record. Repository release status remains **Candidate** pending evidence review; this record does not promote it.
