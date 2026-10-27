# ROI model assets

`roi_net.safetensors` is downloaded here by `pnpm model:fetch` (or by the Vite
dev/build plugin) from the release asset declared in
[`model-manifest.json`](../../model-manifest.json).

It is produced from the blob_lab checkpoint by `scripts/exportRoiRunntime.py`;
the web app loads it through the vendored ruNNtime core.

The file is intentionally git-ignored. Do not commit it. The manifest, SHA-256,
and release workflow live in the repository; the binary stays in the GitHub
release.
