"""Generate procedural spot/noise/line negatives for the ROI detector.

These are pure negatives: no hex target and no composite. They broaden the
negative distribution beyond office/wall photographs, especially for camera
noise, reflections, dust, paper marks, and high-contrast spot clutter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "natural_images" / "negatives"


def fractal_noise(rng: np.random.Generator, height: int, width: int, octaves: int = 5) -> np.ndarray:
    total = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    weight = 0.0
    for octave in range(octaves):
        low_h = max(2, height // (2 ** (octaves - octave - 1)))
        low_w = max(2, width // (2 ** (octaves - octave - 1)))
        seed = rng.random((low_h, low_w)).astype(np.float32)
        layer = cv2.resize(seed, (width, height), interpolation=cv2.INTER_CUBIC)
        total += layer * amplitude
        weight += amplitude
        amplitude *= 0.55
    total /= max(1e-6, weight)
    return (total - total.min()) / max(1e-6, total.max() - total.min())


def illuminate(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = rng.uniform(0.2, 0.8) * width
    cy = rng.uniform(0.2, 0.8) * height
    radius = rng.uniform(0.35, 0.9) * max(height, width)
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    falloff = np.clip(1.0 - 0.35 * distance**2, 0.35, 1.0)
    return image * falloff


def add_soft_spots(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    count: int,
    radius: tuple[float, float],
    polarity: float,
) -> np.ndarray:
    height, width = image.shape
    overlay = np.zeros((height, width), dtype=np.float32)
    for _ in range(count):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        r = max(1, int(rng.uniform(*radius)))
        strength = float(rng.uniform(0.08, 0.5)) * polarity
        cv2.circle(overlay, (cx, cy), r, strength, -1, cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (0, 0), float(rng.uniform(0.6, 3.0)))
    return np.clip(image + overlay, 0.0, 1.0)


def add_scratches(image: np.ndarray, rng: np.random.Generator, count: int) -> np.ndarray:
    height, width = image.shape
    overlay = np.zeros((height, width), dtype=np.float32)
    for _ in range(count):
        x0, y0 = int(rng.integers(0, width)), int(rng.integers(0, height))
        angle = rng.uniform(0, np.pi)
        length = rng.uniform(30, max(width, height))
        x1 = int(np.clip(x0 + np.cos(angle) * length, 0, width - 1))
        y1 = int(np.clip(y0 + np.sin(angle) * length, 0, height - 1))
        color = float(rng.uniform(-0.18, 0.18))
        cv2.line(overlay, (x0, y0), (x1, y1), color, int(rng.integers(1, 3)), cv2.LINE_AA)
    return np.clip(image + cv2.GaussianBlur(overlay, (0, 0), 0.8), 0.0, 1.0)


def add_grid(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape
    overlay = np.zeros((height, width), dtype=np.float32)
    spacing = int(rng.integers(22, 70))
    thickness = int(rng.integers(1, 3))
    color = float(rng.uniform(0.04, 0.16))
    for x in range(int(rng.integers(0, spacing)), width, spacing):
        cv2.line(overlay, (x, 0), (x, height - 1), color, thickness, cv2.LINE_AA)
    for y in range(int(rng.integers(0, spacing)), height, spacing):
        cv2.line(overlay, (0, y), (width - 1, y), color, thickness, cv2.LINE_AA)
    return np.clip(image + overlay, 0.0, 1.0)


def make_negative(kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height = int(rng.choice([480, 540, 640, 720]))
    width = int(height * rng.choice([4 / 3, 16 / 9, 1.0]))
    base = float(rng.uniform(0.18, 0.82))
    texture = fractal_noise(rng, height, width, octaves=int(rng.integers(3, 6)))
    image = base + (texture - 0.5) * float(rng.uniform(0.15, 0.55))
    image = illuminate(image, rng)

    if kind == "spots":
        image = add_soft_spots(
            image,
            rng,
            count=int(rng.integers(8, 55)),
            radius=(1.0, 24.0),
            polarity=float(rng.choice([-1.0, 1.0])),
        )
        image = add_soft_spots(
            image,
            rng,
            count=int(rng.integers(2, 12)),
            radius=(2.0, 6.0),
            polarity=float(rng.choice([-1.0, 1.0])),
        )
    elif kind == "noise":
        grain = rng.normal(0.0, rng.uniform(0.03, 0.12), size=image.shape).astype(np.float32)
        image = image + grain
        image = add_scratches(image, rng, int(rng.integers(4, 18)))
    elif kind == "grid":
        image = add_grid(image, rng)
        image = add_soft_spots(image, rng, count=int(rng.integers(0, 8)), radius=(1.0, 8.0), polarity=1.0)
    else:  # marks: paper/whiteboard-ish strokes and specks
        image = image * float(rng.uniform(0.9, 1.25)) + float(rng.uniform(0.02, 0.12))
        image = add_scratches(image, rng, int(rng.integers(8, 24)))
        image = add_soft_spots(image, rng, count=int(rng.integers(15, 70)), radius=(0.7, 3.0), polarity=-1.0)

    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot", type=int, default=12)
    parser.add_argument("--noise", type=int, default=12)
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--marks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    index = 0
    for kind, count in (("spots", args.spot), ("noise", args.noise), ("grid", args.grid), ("marks", args.marks)):
        for item in range(count):
            index += 1
            seed = args.seed + index * 97
            image = make_negative(kind, seed)
            name = f"synthetic_{kind}_{item + 1:02d}.jpg"
            cv2.imwrite(str(OUT_DIR / name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            manifest.append({"file": name, "kind": kind, "seed": seed})
            print(name, image.shape)
    (OUT_DIR / "synthetic_negatives.json").write_text(json.dumps(manifest, indent=2))
    sources = OUT_DIR / "SOURCES.md"
    existing = sources.read_text() if sources.exists() else ""
    marker = "## Synthetic negatives (procedural)"
    if marker not in existing:
        existing = existing.rstrip() + "\n\n" + marker + "\n"
    existing += (
        "\nGenerated by `python/scripts/generate_synthetic_negatives.py` with seed "
        f"{args.seed}. These are procedural noise/spot/line negatives, not photographs.\n"
    )
    sources.write_text(existing)
    print(f"saved {len(manifest)} synthetic negatives to {OUT_DIR}")


if __name__ == "__main__":
    main()
