# AST E2E Kaggle T4 verification

This directory retains the serial-suite evidence for the standalone AST acoustic-ecology E2E notebook.

## Immutable identity

- Target commit: `b939aa3cd30cbe334c43f87b96eb96ed8e16180e`
- Notebook: `tutorials/ast_audio_classification_colab.ipynb`
- Git blob: `3a99508abd3b3308174a5f8346e685fca771bbf4`
- Embedded source revision: `c54ddcf129968b835ef2ffda2bf5da8478a1a435`
- Kernel: `kurtvalcorza/dimer-nb2-ast-audio-classification`, version 2
- Kaggle image: `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

## Result

- Outcome: **PASS**
- Runtime: Python 3.12.13, Tesla T4 15,360 MiB
- Installed notebook pins: PyTorch `2.14.0+cu130`, torchaudio `2.11.0+cu130`, Transformers `4.57.6`
- Fresh cache: yes
- Wall time: 204.3 seconds
- Cells: 16/16 successful after one expected restart following dependency replacement
- Model snapshot: all four manifest entries downloaded and digest-verified at revision `f826b80d28226b62986cc218e5cec390b1096902`
- Adapted evaluation: accuracy `1.0`, macro-F1 `1.0`, majority baseline `0.3333`, six generated held-out clips
- Unseen generated clip: `biophony`, score `0.987455`
- Artifact: `org.valcorza.ast-audio.adapter.v1`, 19,103 bytes
- Reload: passed; original and reloaded top score `0.987455`

The metrics are synthetic sample-sanity evidence, not field-recording or benchmark evidence.

## Preserved output hashes

| File | Bytes | SHA-256 |
|---|---:|---|
| `ast-audio-adapter-v1.pt` | 19,103 | `15d8ccaca37dc015e4aa211d18215a5a007329e55aec96a009e25fd4af29c911` |
| `ast_audio_classification_evaluation_report.json` | 1,551 | `ff241157258cb25667afb0b88d01400e418b3f251da6835dfd057376d1ac1eb8` |
| `ast_audio_classification_input_manifest.json` | 1,300 | `1446278e9a83b3d6875afc7f2b99ffa9f4818acae4ac7d3c81b6c6dc19aec249` |
| `ast_audio_classification_result.json` | 5,452 | `ebd8fd383b4e42c95ee053a777b9a66e585f39772d8477dc47cf82df2e21f9aa` |
| `ast_audio_classification_top_k.csv` | 161 | `d930eef1c30aa6dd278bfec0cf243bcaa77e367b860a8ee9bc8d97baba09cdbb` |

`suite/` contains the generated executor, kernel metadata, first-pass restart record, successful executed notebook, `run_summary.json`, and preserved outputs. `LEDGER.md` is the serial-suite audit row.
