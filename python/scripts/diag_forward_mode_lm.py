"""Compare JAX forward-mode AD (jacfwd) vs finite-difference Jacobians for LM."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from render_model import (
    JOINT_PARAM_NAMES,
    OptimizeLMConfig,
    RenderModelParams,
    TAG_CANONICAL_CORNERS,
    loss_mask_from_corners,
    build_tag_pattern,
    corners_from_homography,
    homography_from_corners,
    model_params_to_vector,
    optimize_render_model_lm,
)
from render_model.camera_params import vector_to_model_params
from render_model.homography import homography_to_params, params_to_homography
from render_model.optimizer import (
    _pack_opt_params,
    _unpack_opt_params,
    make_render_model_residuals,
)

TEST_IMAGES = Path(__file__).resolve().parent.parent / "notebooks" / "test_images"


def tag_id_from_filename(name: str) -> int:
    stem = Path(name).stem
    if stem.endswith("_1x"):
        stem = stem[: -len("_1x")]
    return int(stem.removeprefix("image_"))


def load_problem(image_path: Path):
    corners_path = image_path.with_name(f"{image_path.stem}.corners.json")
    photo = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    height, width = photo.shape
    init_corners_px = np.asarray(json.loads(corners_path.read_text()), dtype=np.float32)
    src_corners = jnp.asarray(TAG_CANONICAL_CORNERS, dtype=np.float32)
    H_init = homography_from_corners(src_corners, jnp.asarray(init_corners_px))
    tag_pattern = build_tag_pattern(tag_id_from_filename(image_path.name))
    init_corners = corners_from_homography(H_init, src_corners)
    loss_mask = loss_mask_from_corners(init_corners, height, width)
    target = jnp.asarray(photo, dtype=np.float32)
    camera_init = RenderModelParams(
        psf_sigma=jnp.float32(0.5),
        sharpen_amount=jnp.float32(1.0),
        sharpen_sigma=jnp.float32(1.0),
        gamma=jnp.float32(2.0),
        black_level=jnp.float32(0.2),
        white_level=jnp.float32(0.8),
        light_grad_u=jnp.float32(0.0),
        light_grad_v=jnp.float32(0.0),
    )
    packed = _pack_opt_params(
        homography_to_params(H_init),
        model_params_to_vector(camera_init),
        optimize_homography=True,
        optimize_camera=True,
    )
    return dict(
        height=height,
        width=width,
        tag_pattern=tag_pattern,
        loss_mask=loss_mask,
        target=target,
        packed=packed,
    )


def finite_diff_jacobian(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    p: jnp.ndarray,
    eps: float,
    *,
    central: bool = False,
) -> jnp.ndarray:
    """Forward (or central) finite-difference Jacobian — WebGPU-friendly pattern."""
    n = p.shape[0]
    eye = jnp.eye(n, dtype=p.dtype)

    if central:

        def col(i: int) -> jnp.ndarray:
            step = eps * eye[i]
            return (residual_fn(p + step) - residual_fn(p - step)) / (2.0 * eps)

    else:
        r0 = residual_fn(p)

        def col(i: int) -> jnp.ndarray:
            return (residual_fn(p + eps * eye[i]) - r0) / eps

    return jax.vmap(col)(jnp.arange(n)).T


def jacobian_metrics_with_residual(
    r: np.ndarray, J_ad: np.ndarray, J_fd: np.ndarray
) -> dict[str, float]:
    diff = J_fd - J_ad
    ad_norm = np.linalg.norm(J_ad)
    rel_fro = float(np.linalg.norm(diff) / max(ad_norm, 1e-12))
    col_cos = []
    col_rel = []
    for i in range(J_ad.shape[1]):
        a = J_ad[:, i]
        b = J_fd[:, i]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 and nb < 1e-12:
            col_cos.append(1.0)
            col_rel.append(0.0)
        elif na < 1e-12 or nb < 1e-12:
            col_cos.append(0.0)
            col_rel.append(float("inf"))
        else:
            col_cos.append(float(np.dot(a, b) / (na * nb)))
            col_rel.append(float(np.linalg.norm(b - a) / na))
    g_ad = J_ad.T @ r
    g_fd = J_fd.T @ r
    g_rel = float(np.linalg.norm(g_fd - g_ad) / max(np.linalg.norm(g_ad), 1e-12))
    return {
        "rel_frobenius": rel_fro,
        "min_col_cosine": float(min(col_cos)),
        "mean_col_cosine": float(np.mean(col_cos)),
        "max_col_rel_err": float(max(col_rel)),
        "grad_rel_err": g_rel,
    }


def compile_lm(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n_steps: int,
    *,
    jacobian_fn: Callable[[jnp.ndarray], jnp.ndarray],
    damping_factor: float = 10.0,
    min_damping: float = 1e-8,
    max_damping: float = 1e8,
):
    @jax.jit
    def run(packed: jnp.ndarray, damping: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def body(
            carry: tuple[jnp.ndarray, jnp.ndarray], _: None
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
            p, lam = carry
            r = residual_fn(p)
            J = jacobian_fn(p)
            g = J.T @ r
            hess = J.T @ J + lam * jnp.eye(p.shape[0], dtype=p.dtype)
            delta = jnp.linalg.solve(hess, -g)
            p_try = p + delta
            r_try = residual_fn(p_try)
            loss = jnp.sum(r * r)
            loss_try = jnp.sum(r_try * r_try)
            accept = loss_try < loss
            p = jnp.where(accept, p_try, p)
            lam = jnp.where(accept, lam / damping_factor, lam * damping_factor)
            lam = jnp.clip(lam, min_damping, max_damping)
            return (p, lam), jnp.where(accept, loss_try, loss)

        (packed, _), losses = jax.lax.scan(body, (packed, damping), None, length=n_steps)
        return packed, losses

    return run


def run_lm_with_jacobian(
    problem: dict,
    *,
    jacobian_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n_steps: int = 30,
    initial_damping: float = 1e-2,
) -> tuple[jnp.ndarray, list[float]]:
    residual_fn = make_render_model_residuals(
        problem["target"],
        problem["tag_pattern"],
        problem["height"],
        problem["width"],
        loss_mask=problem["loss_mask"],
    )
    packed = problem["packed"]
    init_loss = float(jnp.sum(residual_fn(packed) ** 2))
    run = compile_lm(
        residual_fn,
        n_steps,
        jacobian_fn=jacobian_fn,
    )
    packed, step_losses = run(
        packed, jnp.asarray(initial_damping, dtype=jnp.float32)
    )
    losses = [init_loss, *[float(x) for x in step_losses]]
    return packed, losses


def packed_to_named(packed: jnp.ndarray) -> dict[str, float]:
    h, cam = _unpack_opt_params(
        packed, optimize_homography=True, optimize_camera=True
    )
    assert h is not None and cam is not None
    camera = vector_to_model_params(cam)
    out: dict[str, float] = {}
    for i, name in enumerate(JOINT_PARAM_NAMES[:8]):
        out[name] = float(h[i])
    from render_model import camera_params_physical_vector

    for name, val in zip(JOINT_PARAM_NAMES[8:], camera_params_physical_vector(camera)):
        out[name] = float(val)
    return out


def main() -> int:
    eps_values = [1e-2, 1e-3, 1e-4, 1e-5]
    n_steps = 30

    images = sorted(
        p
        for p in TEST_IMAGES.glob("*.png")
        if p.with_name(f"{p.stem}.corners.json").is_file()
    )
    if not images:
        print(f"No test images in {TEST_IMAGES}", file=sys.stderr)
        return 1

    print("=== Jacobian comparison at init (jacfwd vs finite diff) ===\n")
    best_eps = 1e-4
    best_mean_cos = -1.0

    for image_path in images:
        problem = load_problem(image_path)
        residual_fn = make_render_model_residuals(
            problem["target"],
            problem["tag_pattern"],
            problem["height"],
            problem["width"],
            loss_mask=problem["loss_mask"],
        )
        p = problem["packed"]
        r = np.asarray(residual_fn(p))
        J_ad = np.asarray(jax.jacfwd(residual_fn)(p))

        print(f"{image_path.name}  ({problem['width']}×{problem['height']}, {r.size} residuals)")
        for eps in eps_values:
            for central, label in ((False, "forward"), (True, "central")):
                J_fd = np.asarray(finite_diff_jacobian(residual_fn, p, eps, central=central))
                m = jacobian_metrics_with_residual(r, J_ad, J_fd)
                print(
                    f"  eps={eps:.0e} {label:7s}  "
                    f"rel_F={m['rel_frobenius']:.2e}  "
                    f"col_cos={m['mean_col_cosine']:.4f}  "
                    f"grad_rel={m['grad_rel_err']:.2e}"
                )
                if not central and m["mean_col_cosine"] > best_mean_cos:
                    best_mean_cos = m["mean_col_cosine"]
                    best_eps = eps
        print()

    print(f"Best forward-diff eps by mean column cosine: {best_eps:.0e}\n")

    print("=== LM convergence: jacfwd vs forward finite diff ===\n")
    print(f"{'image':<22} {'ad final':>10} {'fd final':>10} {'Δloss':>10} {'max|Δparam|':>12}")

    for image_path in images:
        problem = load_problem(image_path)
        residual_fn = make_render_model_residuals(
            problem["target"],
            problem["tag_pattern"],
            problem["height"],
            problem["width"],
            loss_mask=problem["loss_mask"],
        )

        fd_jac_fn = lambda p, rf=residual_fn, e=best_eps: finite_diff_jacobian(
            rf, p, e, central=False
        )

        t0 = time.perf_counter()
        packed_ad, losses_ad = run_lm_with_jacobian(
            problem, jacobian_fn=lambda p: jax.jacfwd(residual_fn)(p), n_steps=n_steps
        )
        t_ad = time.perf_counter() - t0

        t0 = time.perf_counter()
        packed_fd, losses_fd = run_lm_with_jacobian(
            problem, jacobian_fn=fd_jac_fn, n_steps=n_steps
        )
        t_fd = time.perf_counter() - t0

        final_ad = losses_ad[-1]
        final_fd = losses_fd[-1]
        params_ad = packed_to_named(packed_ad)
        params_fd = packed_to_named(packed_fd)
        max_dparam = max(abs(params_ad[k] - params_fd[k]) for k in params_ad)

        print(
            f"{image_path.name:<22} {final_ad:10.6f} {final_fd:10.6f} "
            f"{final_fd - final_ad:+10.2e} {max_dparam:12.4e}  "
            f"({t_ad:.2f}s ad, {t_fd:.2f}s fd)"
        )

    # Reference: library optimize_render_model_lm
    print("\n=== Cross-check vs optimize_render_model_lm ===")
    for image_path in images[:1]:
        problem = load_problem(image_path)
        H_init = params_to_homography(
            _unpack_opt_params(problem["packed"], optimize_homography=True, optimize_camera=True)[0]
        )
        camera_init = vector_to_model_params(
            _unpack_opt_params(problem["packed"], optimize_homography=True, optimize_camera=True)[1]
        )
        _, _, losses_ref = optimize_render_model_lm(
            H_init,
            problem["target"],
            problem["tag_pattern"],
            problem["height"],
            problem["width"],
            camera_init=camera_init,
            config=OptimizeLMConfig(n_steps=n_steps, loss_mask=problem["loss_mask"]),
        )
        print(f"{image_path.name}: ref final loss = {losses_ref[-1]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
