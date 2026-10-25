# Model weights hosting

The production ROI model is **not committed to git**. It is published as a
GitHub release asset and fetched into the ignored `public/models/` directory by
the manifest and scripts.

## Files

- [`model-manifest.json`](../model-manifest.json) — release URL, destination,
  size, and SHA-256. This is the source of truth.
- `scripts/modelAsset.ts` — download, checksum, and verification logic.
- `scripts/fetchModel.ts` / `pnpm model:fetch` — download or reuse the local
  asset.
- `scripts/verifyModel.ts` / `pnpm model:verify` — verify the ignored local asset
  without network access.
- `vite.config.ts` — `model-weights` plugin runs the same fetch before dev and
  build.

## One-time release setup

Create a GitHub release (for example tag `models-v2`) and attach
`roi_tiny_no_concat.onnx`:

```bash
gh release create models-v2 public/models/roi_tiny_no_concat.onnx \
  --title "ROI model v2 (full-view close-ups)" \
  --notes "roi_tiny_no_concat.onnx sha256 1207adee2f7fee9bc8caa67483bfaa75902abc112ff57e441fdf1422e1a2641e"
```

The manifest URL is:

```text
https://github.com/deluksic/webcam-calibrator/releases/download/models-v2/roi_tiny_no_concat.onnx
```

The current manifest points at `models-v2`. The release asset must be uploaded
under that tag before a fresh checkout or CI run can fetch the weights; an
existing local `public/models/roi_tiny_no_concat.onnx` that matches the
manifest hash is reused without a download.

For a fork, update `url` in `model-manifest.json` (or set `MODEL_ASSET_URL`).
Release assets are only friction-free without authentication for a public
repository.

## Local development

`pnpm dev` runs the Vite `model-weights` plugin. On startup it checks
`public/models/roi_tiny_no_concat.onnx` against the manifest hash and downloads
it only when missing or stale.

Manual equivalents:

```bash
pnpm model:fetch    # download if missing/stale
pnpm model:fetch -- --force
pnpm model:verify   # offline check of the existing file
```

To use a local or alternate source without changing the manifest:

```bash
MODEL_ASSET_URL=/absolute/path/to/roi_tiny_no_concat.onnx pnpm model:fetch -- --force
```

## CI / GitHub Pages

The Pages workflow runs `pnpm model:fetch` before `pnpm build`. The Vite build
plugin also verifies it. The downloaded ONNX is copied from `public/models/`
into `dist/models/`, so the deployed app loads it same-origin with no runtime
CORS or GitHub dependency.

## Updating the model

1. Upload a new asset under a new immutable tag, e.g. `models-v2`.
2. Update `model-manifest.json`: `releaseTag`, `url`, `bytes`, `sha256`.
3. Run `pnpm model:fetch -- --force` and `pnpm test`.
