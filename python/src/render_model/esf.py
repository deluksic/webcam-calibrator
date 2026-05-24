"""Edge Spread Function (ESF) profile estimation and differentiable lookup.

Image coordinates use edge origin: (0, 0) is the top-left image corner;
ESF distance ``d`` is in screen pixels along the edge normal (black → white).

Relationship to PSF:
  ESF(d) = ∫_{-∞}^{d} LSF(t) dt,   LSF = d/d(d) ESF  (1D PSF cross-section)
We render with ESF directly at sub-pixel distances — no post-process convolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# Match TypeScript gradient-profile path (edgeLineFitPipeline.ts).
PROFILE_BUCKET_COUNT = 64
PROFILE_NEIGHBORHOOD_HALF = 2.5
TAG_DECODE_PEAK_GAP_FRAC = 0.375


@dataclass(frozen=True)
class ESFProfile:
    """Per-tag ESF sampled along edge normal in screen pixels."""

    # ESF(d) in [0, 1]: white fraction at signed distance d from a black→white step.
    esf_samples: jnp.ndarray
    # LSF(d) = d/d(d) ESF, normalized; convolved with Iñigo box ramp at render time.
    lsf_weights: jnp.ndarray
    i_black: jnp.ndarray
    i_white: jnp.ndarray
    half_width: float = PROFILE_NEIGHBORHOOD_HALF

    @property
    def n_bins(self) -> int:
        return int(self.esf_samples.shape[0])

    def scale_intensity(self, normalized: jnp.ndarray) -> jnp.ndarray:
        """Map tag [0, 1] render to observed intensity range."""
        return self.i_black + normalized * (self.i_white - self.i_black)


def _esf_distances(n_bins: int, half_width: float) -> np.ndarray:
    return np.linspace(-half_width, half_width, n_bins, dtype=np.float64)


def _build_lsf_weights(esf: np.ndarray, *, half_width: float = PROFILE_NEIGHBORHOOD_HALF) -> np.ndarray:
    """Discrete LSF from ESF samples; sums to 1 (delta-like for a step ESF)."""
    dist = _esf_distances(len(esf), half_width)
    lsf = np.gradient(esf, dist)
    lsf = np.maximum(lsf, 0.0)
    total = float(np.sum(lsf))
    if total < 1e-8:
        mid = len(lsf) // 2
        lsf = np.zeros_like(lsf)
        lsf[mid] = 1.0
    else:
        lsf /= total
    return lsf.astype(np.float32)


def esf_profile_from_samples(
    esf_samples: np.ndarray | jnp.ndarray,
    i_black: float,
    i_white: float,
    *,
    half_width: float = PROFILE_NEIGHBORHOOD_HALF,
) -> ESFProfile:
    esf_np = np.clip(np.asarray(esf_samples, dtype=np.float64), 0.0, 1.0)
    lsf = _build_lsf_weights(esf_np, half_width=half_width)
    return ESFProfile(
        esf_samples=jnp.asarray(esf_np, dtype=jnp.float32),
        lsf_weights=jnp.asarray(lsf, dtype=jnp.float32),
        i_black=jnp.asarray(i_black, dtype=jnp.float32),
        i_white=jnp.asarray(i_white, dtype=jnp.float32),
        half_width=half_width,
    )


def esf_lookup(d: jnp.ndarray, profile: ESFProfile) -> jnp.ndarray:
    """ESF(d): white fraction at signed pixel distance d from a step edge."""
    half = profile.half_width
    t = (d + half) / (2.0 * half + 1e-8)
    t = jnp.clip(t, 0.0, 1.0)
    n = profile.n_bins
    idx = t * (n - 1)
    i0 = jnp.floor(idx).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, n - 1)
    w = idx - i0.astype(jnp.float32)
    s0 = profile.esf_samples[i0]
    s1 = profile.esf_samples[i1]
    return (1.0 - w) * s0 + w * s1


def ramp_frac_lookup(
    f: jnp.ndarray,
    w: jnp.ndarray,
    profile: ESFProfile,
    *,
    n: float = 2.0,
) -> jnp.ndarray:
    """Iñigo ``min(f·n, 1)`` blurred by ESF (LSF ⊗ box ramp).

    ``w`` is the axis filter footprint in scaled tag coordinates (Iñigo ``w``).
    Pixel distance ``d`` maps to period offset ``d·w``; convolving the LSF with
    the box ramp recovers ``min(f·n, 1)`` exactly for a step ESF.
    """
    half = profile.half_width
    dist = jnp.linspace(-half, half, profile.n_bins, dtype=jnp.float32)
    f = jnp.clip(f, 0.0, 1.0 - 1e-6)
    w = jnp.maximum(w, 1e-6)
    # Broadcast: f (…), dist (K,) -> (…, K)
    f_shifted = f[..., None] + dist[None, ...] * w[..., None]
    box = jnp.minimum(f_shifted * n, 1.0)
    weights = profile.lsf_weights[None, ...]
    return jnp.sum(box * weights, axis=-1)


def esf_params_to_profile(params: jnp.ndarray, base: ESFProfile) -> ESFProfile:
    """Unpack learnable ESF params: [i_black, i_white, esf_delta...]."""
    i_black = params[0]
    i_white = jnp.maximum(params[1], i_black + 1e-4)
    delta = params[2:]
    esf = jnp.clip(base.esf_samples + delta, 0.0, 1.0)
    # LSF fixed from base at init; recompute lsf_weights outside JIT if esf delta is large.
    return ESFProfile(
        esf_samples=esf,
        lsf_weights=base.lsf_weights,
        i_black=i_black,
        i_white=i_white,
        half_width=base.half_width,
    )


def esf_profile_to_params(profile: ESFProfile) -> jnp.ndarray:
    return jnp.concatenate([
        jnp.array([profile.i_black, profile.i_white], dtype=jnp.float32),
        jnp.zeros_like(profile.esf_samples),
    ])


def _align_edge_profile(avg: np.ndarray) -> np.ndarray:
    """Flip profile so intensity rises from black (d<0) to white (d>0)."""
    n = len(avg)
    left = np.mean(avg[: max(1, n // 8)])
    right = np.mean(avg[-max(1, n // 8) :])
    if left > right:
        avg = avg[::-1]
    return avg


def _monotonic_esf(esf: np.ndarray) -> np.ndarray:
    """Force non-decreasing ESF (black→white along normal)."""
    return np.maximum.accumulate(esf)


def _center_edge_profile(esf: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Shift profile so the 50% crossing sits at d=0."""
    if esf[0] >= 0.5:
        return esf
    if esf[-1] <= 0.5:
        return esf
    i1 = int(np.searchsorted(esf, 0.5))
    i0 = max(i1 - 1, 0)
    if esf[i1] == esf[i0]:
        return esf
    t = (0.5 - esf[i0]) / (esf[i1] - esf[i0])
    d50 = dist[i0] + t * (dist[i1] - dist[i0])
    return np.interp(dist, dist + d50, esf, left=esf[0], right=esf[-1])


def _sample_edge_profiles(
    image: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    normal: np.ndarray,
    *,
    n_bins: int = PROFILE_BUCKET_COUNT,
    half_width: float = PROFILE_NEIGHBORHOOD_HALF,
    n_along: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample gray along edge normal; returns (bin_counts, bin_sums)."""
    h, w = image.shape
    counts = np.zeros(n_bins, dtype=np.float64)
    sums = np.zeros(n_bins, dtype=np.float64)
    dist = np.linspace(-half_width, half_width, n_bins)
    for t in np.linspace(0.05, 0.95, n_along):
        center = p0 + t * (p1 - p0)
        for i, d in enumerate(dist):
            pt = center + normal * d
            x, y = pt[0], pt[1]
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
            if 0 <= x0 < w and 0 <= y0 < h:
                fx, fy = x - x0, y - y0
                v = (
                    (1 - fx) * (1 - fy) * image[y0, x0]
                    + fx * (1 - fy) * image[y0, x1]
                    + (1 - fx) * fy * image[y1, x0]
                    + fx * fy * image[y1, x1]
                )
                counts[i] += 1.0
                sums[i] += v
    return counts, sums


def estimate_esf_from_target(
    target: np.ndarray,
    image_corners: np.ndarray,
    *,
    n_bins: int = PROFILE_BUCKET_COUNT,
    half_width: float = PROFILE_NEIGHBORHOOD_HALF,
) -> ESFProfile:
    """Estimate per-tag ESF from image edge profiles (NumPy, init-time only).

    ``image_corners`` is TL, TR, BR, BL in edge-origin image coordinates.
    """
    img = np.asarray(target, dtype=np.float64)
    corners = np.asarray(image_corners, dtype=np.float64)
    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]
    all_counts = np.zeros(n_bins, dtype=np.float64)
    all_sums = np.zeros(n_bins, dtype=np.float64)
    n_edges = 0

    for p0, p1 in edges:
        tangent = p1 - p0
        length = np.linalg.norm(tangent)
        if length < 1e-6:
            continue
        tangent /= length
        # CCW tag boundary: interior is left of edge; normal points outward (black→white).
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        c, s = _sample_edge_profiles(
            img, p0, p1, normal, n_bins=n_bins, half_width=half_width
        )
        avg_edge = np.divide(s, np.maximum(c, 1.0))
        avg_edge = _align_edge_profile(avg_edge)
        all_counts += c
        all_sums += avg_edge * c
        n_edges += 1

    avg = np.divide(all_sums, np.maximum(all_counts, 1.0))
    valid = all_counts > 0
    if not np.any(valid):
        return esf_profile_from_samples(
            np.linspace(0.0, 1.0, n_bins, dtype=np.float32),
            0.0,
            1.0,
            half_width=half_width,
        )

    gray_min = float(np.min(avg[valid]))
    gray_max = float(np.max(avg[valid]))
    diff = gray_max - gray_min
    if diff < 1e-4:
        return esf_profile_from_samples(
            np.linspace(0.0, 1.0, n_bins, dtype=np.float32),
            gray_min,
            gray_max + 1e-4,
            half_width=half_width,
        )

    i_black = gray_min
    i_white = gray_max if gray_max > gray_min + 1e-4 else gray_min + 1e-4
    esf = np.clip((avg - i_black) / (i_white - i_black + 1e-8), 0.0, 1.0)
    esf = _monotonic_esf(esf)
    dist = _esf_distances(n_bins, half_width)
    esf = _center_edge_profile(esf, dist)
    return esf_profile_from_samples(esf.astype(np.float32), i_black, i_white, half_width=half_width)
