# Release status

Current status: **Candidate** — the current standalone `E2E` tutorial has source validation, offline unit coverage, and a local CPU exercise of the real pinned AST checkpoint, but no clean supported-runtime execution record for this revision.

The 2026-09-16 local CPU exercise ran all 16 generated code cells with the exact pins already installed. It used the deterministic 24-clip generated dataset, 18/6 stratified split, frozen backbone, seeded three-class head, five epochs, batch size 4, and learning rate `1e-3`. It observed held-out accuracy and macro-F1 of `1.0`, versus a `0.3333` majority baseline; predicted the unseen generated biophony clip correctly; wrote an approximately 19 KB classifier-head artifact; and reproduced its scores exactly after reload. These are narrow sample-sanity observations on synthetic data, not benchmark or field-recording evidence.

The retained 2026-09-13 Colab run covered an earlier inference-only notebook blob. It remains useful historical base-inference evidence but cannot qualify the changed E2E notebook. Promotion requires a fresh, unchanged top-to-bottom run of the exact current notebook in a supported clean runtime and review of the retained outputs described in [docs/release-verification.md](docs/release-verification.md).
