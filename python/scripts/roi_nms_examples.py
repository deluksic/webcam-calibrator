"""Generate clean side-by-side examples for the ROI NMS experiment."""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_histogram_nms import (  # noqa: E402
    OUT_DIR,
    ROOT,
    connected_component_centers,
    histogram_cuts,
    load_gray,
    load_mask,
    nms_contrast_centers,
    segment,
)
from blob_lab.heatmap_dataset import HeatmapDatasetConfig, generate_heatmap_dataset  # noqa: E402
from blob_lab.peaks import Peak, match_peaks  # noqa: E402
from blob_lab.target_layout import build_hex_target_layout  # noqa: E402

EXAMPLES = OUT_DIR / "examples"
EXAMPLES.mkdir(parents=True, exist_ok=True)
PANEL_HEIGHT = 380


def crop_and_scale(gray, roi):
    ys, xs = np.where(roi)
    pad = 10
    x0, x1 = max(0, xs.min() - pad), min(gray.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(gray.shape[0], ys.max() + pad)
    crop = gray[y0:y1, x0:x1]
    scale = PANEL_HEIGHT / max(1, crop.shape[0])
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return resized, x0, y0, scale


def panel(gray, roi, points, color, title, gt=None):
    base, x0, y0, scale = crop_and_scale(gray, roi)
    rgb = np.repeat(np.clip(base, 0, 1)[:, :, None], 3, axis=2)
    if gt is not None:
        for x, y in gt:
            px = int(round((x - x0) * scale))
            py = int(round((y - y0) * scale))
            cv2.drawMarker(rgb, (px, py), (1.0, 0.9, 0.1), cv2.MARKER_CROSS, 5, 1)
    radius = max(2, int(round(2.0 * scale)))
    for x, y in points:
        px = int(round((x - x0) * scale))
        py = int(round((y - y0) * scale))
        cv2.circle(rgb, (px, py), radius, color, 1)
    cv2.putText(rgb, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1, cv2.LINE_AA)
    return rgb


def make_case(name, gray, roi, truth=None):
    cuts = histogram_cuts(gray, roi)
    black, white = segment(gray, roi, cuts)
    cc = [(x, y) for x, y, _ in connected_component_centers(black)] + [
        (x, y) for x, y, _ in connected_component_centers(white)
    ]
    nms = [(x, y) for x, y, _ in nms_contrast_centers(gray, black, -1, 2)] + [
        (x, y) for x, y, _ in nms_contrast_centers(gray, white, +1, 2)
    ]
    panels = [panel(gray, roi, [], (1, 1, 1), "grayscale", gt=truth)]
    if truth is not None:
        panels.append(panel(gray, roi, cc, (1, 0.2, 0.2), f"CC  n={len(cc)}", gt=truth))
        panels.append(panel(gray, roi, nms, (0.1, 1, 0.2), f"NMS  n={len(nms)}", gt=truth))
    else:
        panels.append(panel(gray, roi, cc, (1, 0.2, 0.2), f"CC  n={len(cc)}"))
        panels.append(panel(gray, roi, nms, (0.1, 1, 0.2), f"NMS+fit  n={len(nms)}"))
    montage = np.hstack(panels)
    out = EXAMPLES / f"{name}.png"
    bgr = cv2.cvtColor(np.clip(montage * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out), bgr)

    if truth is not None:
        def f1(points):
            return match_peaks(
                [Peak(float(x), float(y), 1.0) for x, y in points],
                [(float(x), float(y)) for x, y in truth],
                tolerance=1.5,
            ).f1
        print(name, "cc", len(cc), round(f1(cc), 3), "nms", len(nms), round(f1(nms), 3))
    else:
        print(name, "cc", len(cc), "nms", len(nms))


# synthetic small markers
for w, h, idx, seed in ((640, 480, 2, 7), (1280, 960, 1, 11)):
    layout = build_hex_target_layout(width_px=w, height_px=h)
    cfg = HeatmapDatasetConfig(
        output_width=w,
        output_height=h,
        samples=max(3, idx + 1),
        natural_fraction=0.0,
        randomize_colors=False,
        blur_sigma=(0.8, 1.6),
        noise_sigma=(0.0, 0.01),
        seed=seed,
    )
    ds = generate_heatmap_dataset(cfg, layout, count=max(3, idx + 1))
    gray = np.asarray(ds.images[idx, :, :, 0], dtype=np.float32)
    roi = np.asarray(ds.masks[idx, :, :, 0]) >= 0.5
    truth = [(float(x), float(y)) for x, y in ds.metadata[idx]["points"]]
    make_case(f"synthetic_{w}x{h}", gray, roi, truth)

# real photos
real = ROOT / "python" / "notebooks" / "real_positives"
for name, image, mask in (
    ("real_positive_01", "real_positive_01.jpg", "real_positive_01_mask.png"),
    ("real_positive_02", "real_positive_02.png", "real_positive_02_mask.png"),
):
    gray = load_gray(real / image)
    roi = load_mask(real / mask, gray.shape)
    make_case(name, gray, roi)

# demo images
for name, image, mask in (
    ("demo_target", ROOT / "public" / "demo" / "roi_target.png", None),
    ("demo_real_01", ROOT / "public" / "demo" / "roi_real_01.jpg", None),
):
    from roi_histogram_nms import roi_mask_from_onnx

    gray = load_gray(image)
    roi = load_mask(mask, gray.shape) if mask is not None else roi_mask_from_onnx(image)
    make_case(name, gray, roi)
