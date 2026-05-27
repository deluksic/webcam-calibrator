"""AprilTag-style binary patterns (tag36h11, matches src/lib/tag36h11.ts)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Tag module grid: 8×8 cells = 6×6 data + 1-cell black border.
TAG_GRID_SIZE = 8

# Canonical tag corners in tag edge coordinates (top-left origin, extent 0…8).
TAG_CANONICAL_CORNERS = np.array(
    [
        [0.0, 0.0],
        [8.0, 0.0],
        [8.0, 8.0],
        [0.0, 8.0],
    ],
    dtype=np.float64,
)

# AprilTag tag36h11 bit → 1-indexed coords in the 10×10 grid (same as tag36h11.ts).
_BIT_X = (
    1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4, 1, 1, 1, 1, 1, 2, 2, 2, 3,
)
_BIT_Y = (
    1, 1, 1, 1, 1, 2, 2, 2, 3, 1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4,
)


def _repo_tag36h11_json() -> Path:
    """Monorepo path: ``src/lib/tag36h11.json`` relative to repo root."""
    return Path(__file__).resolve().parents[3] / "src" / "lib" / "tag36h11.json"


@lru_cache(maxsize=1)
def load_tag36h11_codes() -> tuple[int, ...]:
    path = _repo_tag36h11_json()
    if not path.is_file():
        raise FileNotFoundError(
            f"tag36h11 dictionary not found at {path}; run from the webcam-calibrator repo"
        )
    raw = json.loads(path.read_text())
    return tuple(int(x) for x in raw)


def tag36h11_code(tag_id: int) -> int:
    codes = load_tag36h11_codes()
    if tag_id < 0 or tag_id >= len(codes):
        raise ValueError(f"tag36h11 id {tag_id} out of range [0, {len(codes)})")
    return codes[tag_id]


def code_to_interior_pattern(code: int) -> np.ndarray:
    """6×6 interior (1=white, 0=black) from a 36-bit tag36h11 code."""
    interior = np.zeros((6, 6), dtype=np.float32)
    for bit in range(36):
        col = _BIT_X[bit] - 1
        row = _BIT_Y[bit] - 1
        interior[row, col] = 1.0 if (code >> (35 - bit)) & 1 else 0.0
    return interior


def build_tag_pattern(tag_id: int = 0, custom_codes: list[str] | None = None) -> jnp.ndarray:
    """Build an 8×8 float32 pattern in [0, 1] (1=white, 0=black)."""
    if tag_id < 0:
        if not custom_codes:
            raise ValueError(f"custom tag id {tag_id} requires custom_codes")
        idx = -tag_id - 1
        if idx < 0 or idx >= len(custom_codes):
            raise ValueError(
                f"custom tag id {tag_id} index {idx} out of range [0, {len(custom_codes)})"
            )
        code = int(custom_codes[idx])
    else:
        code = tag36h11_code(tag_id)
    interior = code_to_interior_pattern(code)
    grid = np.ones((TAG_GRID_SIZE, TAG_GRID_SIZE), dtype=np.float32)
    grid[0, :] = 0.0
    grid[-1, :] = 0.0
    grid[:, 0] = 0.0
    grid[:, -1] = 0.0
    grid[1:-1, 1:-1] = interior
    return jnp.asarray(grid)
