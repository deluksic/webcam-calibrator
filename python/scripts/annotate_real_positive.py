"""Click-to-annotate polygon masks for real-positive frames.

Usage:
    python scripts/annotate_real_positive.py --image notebooks/real_positives/real_positive_01.jpg
    python scripts/annotate_real_positive.py --dir notebooks/real_positives

Controls:
    left click   add polygon point
    right click  undo last point
    s            save <stem>_mask.png next to the image
    r            reset polygon
    n            next image (directory mode)
    q / Esc      quit

The saved mask is the same size as the image, white target on black.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

WINDOW = "real-positive mask annotator"
MAX_WIDTH = 1100
MAX_HEIGHT = 800
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class Annotator:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path
        self.image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise SystemExit(f"Could not read {image_path}")
        self.height, self.width = self.image.shape[:2]
        self.scale = min(
            1.0,
            MAX_WIDTH / max(1, self.width),
            MAX_HEIGHT / max(1, self.height),
        )
        self.canvas = cv2.resize(
            self.image,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_AREA,
        )
        self.points: list[tuple[int, int]] = []

    def to_image(self, x: int, y: int) -> tuple[int, int]:
        return (
            int(round(x / self.scale)),
            int(round(y / self.scale)),
        )

    def draw(self) -> np.ndarray:
        display = self.canvas.copy()
        scaled = [
            (int(round(x * self.scale)), int(round(y * self.scale)))
            for x, y in self.points
        ]
        if len(scaled) >= 2:
            cv2.polylines(
                display,
                [np.asarray(scaled, dtype=np.int32)],
                len(scaled) >= 3,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        for point in scaled:
            cv2.circle(display, point, 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            display,
            "left: add  right: undo  s: save  r: reset  n: next  q: quit",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return display

    def save(self) -> Path | None:
        if len(self.points) < 3:
            print("Need at least 3 points.")
            return None
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(
            mask,
            [np.asarray(self.points, dtype=np.int32)],
            255,
        )
        output = self.image_path.with_name(f"{self.image_path.stem}_mask.png")
        cv2.imwrite(str(output), mask)
        print(f"saved {output}")
        return output


def run_one(path: Path) -> str:
    annotator = Annotator(path)
    state = {"saved": False}

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            annotator.points.append(annotator.to_image(x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and annotator.points:
            annotator.points.pop()
        cv2.imshow(WINDOW, annotator.draw())

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    cv2.imshow(WINDOW, annotator.draw())
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            return "quit"
        if key == ord("r"):
            annotator.points.clear()
            cv2.imshow(WINDOW, annotator.draw())
        elif key == ord("s"):
            if annotator.save() is not None:
                state["saved"] = True
        elif key == ord("n"):
            return "next"
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.image:
        run_one(args.image)
        cv2.destroyAllWindows()
        return

    images = [
        path
        for path in sorted(args.dir.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
        and not path.stem.endswith("_mask")
        and (args.overwrite or not path.with_name(f"{path.stem}_mask.png").exists())
    ]
    if not images:
        print("No unannotated images found.")
        return
    for path in images:
        print(f"annotating {path}")
        if run_one(path) == "quit":
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
