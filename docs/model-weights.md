# Model weights hosting

The production ROI model is **not committed to git**. It is published as a
GitHub release asset and fetched into the ignored `public/models/` directory by
the manifest and scripts.

The asset is `roi_net.safetensors`, an **f16** `safetensors` blob consumed by
the vendored ruNNtime core. It is produced from the blob_lab checkpoint by
`scripts/exportRoiRunntime.py`.

## Train → deploy

```bash
# 1. train — writes python/notebooks/blob_lab_data/roi_net_best.pt
#    (notebooks live under python/notebooks)

# 2. export weights (~150 KiB, f16, BatchNorm folded in f32)
pnpm model:export

# 3. update bytes/sha256 in model-manifest.json, then verify
shasum -a 256 public/models/roi_net.safetensors
pnpm model:verify
```

`model:export` also rewrites the spike's reference tensors (`public/spike/*.bin`),
so `spike.html` keeps checking the deployed weights against PyTorch on the same
frame.

BatchNorm is folded **at export**, in f32, and only then rounded to f16, which
halves the file without changing the numbers. The exporter omits the `bn.*` keys
entirely, so the runtime fold is a no-op.

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
- `scripts/exportRoiRunntime.py` — rebuilds `roi_net.safetensors` (plus the
  spike's reference tensors) from `roi_net_best.pt`.

## One-time release setup

Create a GitHub release (for example tag `models-v3`) and attach
`roi_net.safetensors`:

```bash
gh release create models-v3 public/models/roi_net.safetensors \
  --title "ROI model v3 (ruNNtime, f16)" \
  --notes "roi_net.safetensors sha256 9cb08cd8d39a6d40c11b1cc1e9d6590cf43a4d97bce8ebff153bbbe3362585bf"
```

The manifest URL is:

```text
https://github.com/deluksic/webcam-calibrator/releases/download/models-v3/roi_net.safetensors
```

The current manifest points at `models-v3`. The release asset must be uploaded
under that tag before a fresh checkout or CI run can fetch the weights; an
existing local `public/models/roi_net.safetensors` that matches the
manifest hash is reused without a download.

For a fork, update `url` in `model-manifest.json` (or set `MODEL_ASSET_URL`).
Release assets are only friction-free without authentication for a public
repository.

## Local development

`pnpm dev` runs the Vite `model-weights` plugin. On startup it checks
`public/models/roi_net.safetensors` against the manifest hash and downloads
it only when missing or stale.

Manual equivalents:

```bash
pnpm model:fetch    # download if missing/stale
pnpm model:fetch -- --force
pnpm model:verify   # offline check of the existing file
```

To use a local or alternate source without changing the manifest:

```bash
MODEL_ASSET_URL=/absolute/path/to/roi_net.safetensors pnpm model:fetch -- --force
```

## CI / GitHub Pages

The Pages workflow runs `pnpm model:fetch` before `pnpm build`. The Vite build
plugin also verifies it. The downloaded asset is copied from `public/models/`
into `dist/models/`, so the deployed app loads it same-origin with no runtime
CORS or GitHub dependency.

## Updating the model

1. Upload a new asset under a new immutable tag, e.g. `models-v4`.
2. Update `model-manifest.json`: `releaseTag`, `url`, `bytes`, `sha256`.
3. Run `pnpm model:fetch -- --force` and `pnpm test`.

## Regenerating the asset

```bash
python/.venv/bin/python scripts/exportRoiRunntime.py   # writes public/models/roi_net.safetensors
pnpm model:fetch -- --force                            # (only if you host it remotely)
```

Then update `bytes` and `sha256` in `model-manifest.json`
(`shasum -a 256 public/models/roi_net.safetensors`).

> **The `models-v3` release has not been uploaded yet.** Local builds work
> because the asset is already in `public/models/` and matches the manifest hash,
> but a fresh clone or CI run will fail until it is uploaded (or the file is
> force-added to git).
