# Weight provenance and DIMER hosting

- Upstream: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Immutable revision: `f826b80d28226b62986cc218e5cec390b1096902`
- Weight format: SafeTensors (`model.safetensors`, 346,404,948 bytes)
- Upstream weight license: BSD-3-Clause (`license: bsd-3-clause` in the pinned upstream README)
- Local layout: `weights/ast-audioset/` holds `README.md`, `config.json`, `model.safetensors`, `preprocessor_config.json` and `dimer-base-manifest.json`. The manifest lists every file with its byte size and SHA-256; `verify_snapshot()` in `src/ast_audio_classification_pipeline/pipeline.py` checks all four before any load. `.safetensors` files are git-ignored; the Git repository does not vendor the checkpoint.
- DIMER hosting: BSD-3-Clause permits use, modification, redistribution and commercial use provided the copyright notice, the conditions list and the disclaimer are retained and the MIT name is not used to endorse derived products without permission. DIMER may mirror the pinned checkpoint in its model store under those terms.
- Loader trust boundary: Transformers `ASTForAudioClassification` + `ASTFeatureExtractor` with `trust_remote_code=False`; the loader reads only the verified local directory (`local_files_only=True`) and falls back to the Hub at the pinned revision only when `allow_download=True` is passed explicitly.
