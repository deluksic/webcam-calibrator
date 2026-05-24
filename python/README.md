# render-model

Differentiable tag rendering and camera response model (JAX + optax). Lives separately from the TypeScript/WebGPU app in this repo.

## Setup

```bash
cd python
uv sync
```

Requires Python 3.11+ (macOS x86_64: JAX is pinned to 0.4.38 for wheel availability).

## Notebooks

| Notebook | What it does |
|----------|----------------|
| `03_render_model.py` | Camera pipeline demo (PSF, sharpen, gamma) |
| `04_optimize_render_model.py` | Joint H + camera optimization |
| `05_optimizer_comparison.py` | Optimizer comparison (numbers + loss curves) |

```bash
uv run marimo edit notebooks/04_optimize_render_model.py
```

## Retina test images (`*_2x` → `*_1x`)

Drop `image_502_2x.png` etc. into a folder, then halve and remove the 2x files:

```bash
uv run python scripts/downscale_2x_to_1x.py notebooks/test_images
uv run python scripts/downscale_2x_to_1x.py --dry-run path/to/photos
uv run python scripts/downscale_2x_to_1x.py --replace notebooks/test_images
```

Writes `image_502_1x.png` by decimating every other pixel (no Lanczos), scales `.corners.json` if present, deletes `*_2x` sources.

## Package layout

| Module | Role |
|--------|------|
| `render_model.tag_data` | Tag36h11-style patterns |
| `render_model.homography` | DLT, apply, corner metrics |
| `render_model.renderer` | Inverse-warp + analytic AA / ESF render |
| `render_model.pipeline` | Supersample → PSF → bin → sharpen → gamma |
| `render_model.optimizer` | Joint H + camera optax loop |
| `render_model.esf_optimizer` | Legacy ESF + homography optimizer |

## CPU note

`jax` runs on CPU (JIT). On Apple Silicon you can try newer JAX; Intel Mac x86_64 needs jax/jaxlib ≤ 0.4.38.
