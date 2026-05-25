"""Camera response model: Gaussian PSF + sharpening + gamma (differentiable)."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from render_model import (
        TAG_CANONICAL_CORNERS,
        build_tag_pattern,
        homography_from_corners,
        RenderModelParams,
        configure_matplotlib_image_display,
        pixelated_grid,
        render_with_model_stages,
    )

    configure_matplotlib_image_display()
    return (
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        build_tag_pattern,
        homography_from_corners,
        jax,
        jnp,
        np,
        pixelated_grid,
        render_with_model_stages,
    )


@app.cell
def _(mo):
    mo.md("""
    # Camera response model

    Physical pipeline (all differentiable):

    1. **Render** at 4× super-resolution — Iñigo analytical AA (`render_tag_antialiased`)
    2. **Gaussian PSF** — optical blur at high-res
    3. **Bin down** — 4×4 sensor integration
    4. **Unsharp mask** — in-camera sharpening
    5. **Gamma** — tone curve

    Black / white levels are applied **in the 4× AA render** (each tag cell samples at
    the exact level, edges blend linearly between them).

    **Display:** stages are shown in a 2-column grid via ``pixelated_image`` (inline
    ``image-rendering: pixelated``). Use **Display scale** to enlarge each image
    pixel for inspection.
    """)
    return


@app.cell
def _(
    TAG_CANONICAL_CORNERS,
    build_tag_pattern,
    homography_from_corners,
    jnp,
    np,
):
    image_size = 128
    tag_pattern = build_tag_pattern(0)

    src_corners = jnp.asarray(TAG_CANONICAL_CORNERS, dtype=jnp.float32)
    dst_corners = np.array(
        [
            [24.0, 28.0],
            [104.0, 20.0],
            [108.0, 100.0],
            [20.0, 108.0],
        ],
        dtype=np.float32,
    )
    H = homography_from_corners(src_corners, jnp.asarray(dst_corners))
    return H, image_size, tag_pattern


@app.cell
def _(mo):
    psf_sigma = mo.ui.slider(0.1, 2.0, value=0.8, step=0.05, label="PSF σ (output px)")
    sharpen_amount = mo.ui.slider(0.0, 3.0, value=0.5, step=0.05, label="Sharpen amount")
    sharpen_sigma = mo.ui.slider(0.3, 2.0, value=1.0, step=0.05, label="Sharpen σ (output px)")
    gamma = mo.ui.slider(0.5, 3.0, value=2.2, step=0.05, label="Gamma")
    black_level = mo.ui.slider(0.0, 0.3, value=0.0, step=0.01, label="Black level")
    white_level = mo.ui.slider(0.7, 1.0, value=1.0, step=0.01, label="White level")
    mo.vstack([
        mo.md("### Camera parameters"),
        mo.hstack([psf_sigma, sharpen_amount]),
        mo.hstack([sharpen_sigma, gamma]),
        mo.hstack([black_level, white_level]),
    ])
    return black_level, gamma, psf_sigma, sharpen_amount, sharpen_sigma, white_level


@app.cell(hide_code=True)
def _(
    H,
    RenderModelParams,
    black_level,
    gamma,
    image_size,
    jnp,
    psf_sigma,
    render_with_model_stages,
    sharpen_amount,
    sharpen_sigma,
    tag_pattern,
    white_level,
):
    params = RenderModelParams(
        psf_sigma=jnp.float32(psf_sigma.value),
        sharpen_amount=jnp.float32(sharpen_amount.value),
        sharpen_sigma=jnp.float32(sharpen_sigma.value),
        gamma=jnp.float32(gamma.value),
        black_level=jnp.float32(black_level.value),
        white_level=jnp.float32(white_level.value),
    )
    stages = render_with_model_stages(H, tag_pattern, image_size, image_size, params)
    return params, stages


@app.cell(hide_code=True)
def _(mo, np, pixelated_grid, stages):
    panels = [
        ("Iñigo AA (4×)", np.asarray(stages.hi_res)),
        ("After PSF (4×)", np.asarray(stages.hi_blurred)),
        ("Binned", np.asarray(stages.binned)),
        ("Sharpened", np.asarray(stages.sharpened)),
        ("Gamma", np.asarray(stages.final)),
    ]
    mo.vstack([
        mo.md("### Pipeline stages"),
        pixelated_grid(panels, columns=2),
    ])
    return


@app.cell(hide_code=True)
def _(H, jax, jnp, mo, params, render_with_model_stages, tag_pattern):
    def loss_fn(p):
        out = render_with_model_stages(H, tag_pattern, 128, 128, p).final
        return jnp.sum(out**2)

    grads = jax.grad(loss_fn)(params)
    grad_table = {
        "psf_sigma": float(grads.psf_sigma),
        "sharpen_amount": float(grads.sharpen_amount),
        "sharpen_sigma": float(grads.sharpen_sigma),
        "gamma": float(grads.gamma),
        "black_level": float(grads.black_level),
        "white_level": float(grads.white_level),
    }
    mo.vstack([
        mo.md("### Gradient check (d loss / d params, loss = sum(image²))"),
        mo.md(
            "\n".join(f"- **{name}**: `{value:.6e}`" for name, value in grad_table.items())
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
