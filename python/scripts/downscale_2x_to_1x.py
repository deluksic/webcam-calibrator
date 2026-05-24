#!/usr/bin/env python3
"""Decimate Retina ``*_2x`` images to ``*_1x`` (every other pixel), then delete 2x files.

Assumes 2× captures are pixel-doubled (2×2 identical blocks), not optically scaled.

Example::

    uv run downscale-2x
    uv run downscale-2x path/to/photos
    uv run downscale-2x --dry-run notebooks/test_images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _decimate_2x(im: Image.Image) -> Image.Image:
    """Top-left sample of each 2×2 block — no resampling filter."""
    w, h = im.size
    if w % 2 or h % 2:
        raise ValueError(f"expected even width/height, got {w}×{h}")
    arr = np.asarray(im)
    return Image.fromarray(arr[0::2, 0::2].copy(), mode=im.mode)


def _stem_2x(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS and path.stem.endswith("_2x")


def _out_path(path: Path) -> Path:
    return path.with_name(f"{path.stem[:-3]}_1x{path.suffix}")


def _corners_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}.corners.json")


def _scale_corners(corners_path: Path, out_corners: Path) -> None:
    if not corners_path.is_file():
        return
    data = json.loads(corners_path.read_text())
    scaled = [[float(x) * 0.5, float(y) * 0.5] for x, y in data]
    out_corners.write_text(json.dumps(scaled, indent=2))


def downscale_one(src: Path, *, dry_run: bool, replace: bool) -> Path:
    dst = _out_path(src)
    if dst.exists() and not replace:
        raise FileExistsError(f"{dst} exists (pass --replace to overwrite)")

    with Image.open(src) as im:
        w, h = im.size
        out = _decimate_2x(im)

    if dry_run:
        print(f"[dry-run] {src.name} ({w}×{h}) -> {dst.name} ({w // 2}×{h // 2}), delete {src.name}")
        corners_src = _corners_path(src)
        if corners_src.is_file():
            print(f"[dry-run] {corners_src.name} -> {_corners_path(dst).name} (×0.5), delete {corners_src.name}")
        return dst

    out.save(dst)
    _scale_corners(_corners_path(src), _corners_path(dst))
    src.unlink()
    corners_src = _corners_path(src)
    if corners_src.is_file():
        corners_src.unlink()
    print(f"{src.name} -> {dst.name}, removed {src.name}")
    return dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert *_2x images to half-res *_1x and delete the 2x originals.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "notebooks" / "test_images",
        help="Folder containing *_2x.png (default: notebooks/test_images)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing or deleting files",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite existing *_1x outputs",
    )
    args = parser.parse_args(argv)

    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 1

    sources = sorted(p for p in directory.iterdir() if p.is_file() and _stem_2x(p))
    if not sources:
        print(f"no *_2x images in {directory}")
        return 0

    for src in sources:
        downscale_one(src, dry_run=args.dry_run, replace=args.replace)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
