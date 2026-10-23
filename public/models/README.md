# ROI model assets

`roi_tiny_no_concat.onnx` is downloaded here by `pnpm model:fetch` (or by the
Vite dev/build plugin) from the release asset declared in
[`model-manifest.json`](../../model-manifest.json).

The `.onnx` file is intentionally git-ignored. Do not commit it. The manifest,
SHA-256, and release workflow live in the repository; the binary stays in the
GitHub release.
