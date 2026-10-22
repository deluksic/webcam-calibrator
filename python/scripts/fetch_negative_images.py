"""Fetch open-license wall textures and abstract art as negative ROI data.

The images live outside the office-background directory so they are used only
as presence-negative frames. Sources and licenses are appended to
``notebooks/natural_images/negatives/SOURCES.md``.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "natural_images" / "negatives"
USER_AGENT = "webcam-calibrator-roi-lab/0.1 (offline research; contact local project)"

GROUPS = {
    "wall": (
        "Category:Wall textures",
        "Category:Brick walls",
        "Category:Plaster walls",
        "Category:Concrete walls",
        "Category:Stone walls",
        "Category:Wooden walls",
        "Category:Wallpaper",
        "Category:Stucco",
    ),
    "art": (
        "Category:Abstract art",
        "Category:Abstract paintings",
        "Category:Geometric abstraction",
        "Category:Modern art",
        "Category:Abstract photographs",
    ),
}
BLOCKED = (
    "logo",
    "seal",
    "map",
    "diagram",
    "poster",
    "text",
    "sign",
    "sculpture",
    "statue",
    "museum",
    "gallery",
    "exhibition",
    "book",
    "cover",
    "portrait",
    "self-portrait",
    "nude",
    "violence",
    "war",
)


def _api(params: dict[str, str]) -> dict:
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _query_category(category: str, limit: int) -> list[dict]:
    data = _api(
        {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": str(limit),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size",
            "iiurlwidth": "800",
            "format": "json",
        }
    )
    return list(data.get("query", {}).get("pages", {}).values())


def _plain(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _metadata(page: dict) -> dict:
    info = (page.get("imageinfo") or [{}])[0]
    ext = info.get("extmetadata") or {}
    return {
        "title": page.get("title", ""),
        "url": info.get("thumburl") or info.get("url") or "",
        "description_url": info.get("descriptionurl", ""),
        "artist": _plain((ext.get("Artist") or {}).get("value")),
        "license": _plain((ext.get("LicenseShortName") or {}).get("value")),
        "width": int(info.get("width") or 0),
        "height": int(info.get("height") or 0),
    }


def _acceptable(meta: dict) -> bool:
    title = meta["title"].lower()
    if not meta["url"] or not title.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return False
    if meta["width"] < 400 or meta["height"] < 300:
        return False
    if any(word in title for word in BLOCKED):
        return False
    license_text = meta["license"].lower()
    if "non-free" in license_text or "fair use" in license_text:
        return False
    return "cc" in license_text or "public domain" in license_text


def _download(url: str) -> np.ndarray | None:
    request = urllib.request.Request(url.split("?")[0], headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read()
    except Exception as error:  # noqa: BLE001
        print(f"  download failed: {error}")
        return None
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall", type=int, default=24)
    parser.add_argument("--art", type=int, default=24)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources_path = OUT_DIR / "SOURCES.md"
    existing = sources_path.read_text() if sources_path.exists() else ""
    existing_titles = set(re.findall(r"- Title: (.+)", existing))

    saved: list[dict] = []
    for group, categories in GROUPS.items():
        target = args.wall if group == "wall" else args.art
        count = 0
        for category in categories:
            if count >= target:
                break
            try:
                pages = _query_category(category, 50)
            except Exception as error:  # noqa: BLE001
                print(f"{category}: {error}")
                continue
            for page in pages:
                if count >= target:
                    break
                meta = _metadata(page)
                if not _acceptable(meta) or meta["title"] in existing_titles:
                    continue
                name = f"{group}_{count + 1:02d}.jpg"
                path = OUT_DIR / name
                if path.exists():
                    continue
                print(f"{name} <- {meta['title']}")
                image = _download(meta["url"])
                if image is None or image.shape[0] < 200 or image.shape[1] < 300:
                    continue
                height, width = image.shape[:2]
                if width > 960:
                    scale = 960.0 / width
                    image = cv2.resize(
                        image,
                        (960, max(1, int(round(height * scale)))),
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                saved.append({**meta, "file": name})
                existing_titles.add(meta["title"])
                count += 1
                time.sleep(0.1)

    if not saved:
        print("nothing saved")
        return
    marker = "## Wall textures / abstract art negatives (Wikimedia Commons)"
    if marker not in existing:
        existing = existing.rstrip() + "\n\n" + marker + "\n"
    lines = [existing.rstrip(), ""]
    for entry in saved:
        lines.extend(
            [
                f"### {entry['file']}",
                f"- Title: {entry['title']}",
                f"- Artist: {entry['artist'] or 'Unknown'}",
                f"- License: {entry['license'] or 'See source'}",
                f"- Source: {entry['description_url'] or entry['url']}",
                "",
            ]
        )
    sources_path.write_text("\n".join(lines))
    print(f"saved {len(saved)} negatives to {OUT_DIR}")


if __name__ == "__main__":
    main()
