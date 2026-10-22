"""Fetch additional office photographs from Wikimedia Commons.

The images are used only as synthetic backgrounds for the ROI lab.  Each
downloaded image is recorded with its Commons title, author, and license in
``notebooks/natural_images/SOURCES.md`` (appended under a clearly marked
section).
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
NATURAL_DIR = ROOT / "notebooks" / "natural_images"
SOURCES = NATURAL_DIR / "SOURCES.md"
USER_AGENT = "webcam-calibrator-roi-lab/0.1 (offline research; contact local project)"

CATEGORIES = (
    "Category:Open plan offices",
    "Category:Office workers",
    "Category:Office Interior",
    "Category:Computer rooms",
)
QUERIES = ()
ALLOWED = (
    "office",
    "desk",
    "cubicle",
    "workspace",
    "workplace",
    "workstation",
    "computer",
    "interior",
    "room",
    "bureau",
    "coworking",
)
BLOCKED = (
    "mural",
    "cathedral",
    "church",
    "museum",
    "logo",
    "seal",
    "map",
    "portrait",
    "painting",
    "drawing",
    "poster",
    "exterior",
    "facade",
    "building",
    "street",
    "sign",
    "tiff",
    "svg",
    "pdf",
    "nypl",
    "panoramio",
    "officer",
    "secretary",
    "proofreader",
    "dispatch",
    "air wing",
    "circa",
    "1945",
    "auditorium",
    "mansion",
    "antique",
    "bedroom",
    "dorm",
    "classroom",
    "telephone desk",
    "lap desk",
    "armoire",
    "desk bottom",
    "conan",
    "washington",
    "furniture",
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
            "iiprop": "url|extmetadata",
            "iiurlwidth": "640",
            "format": "json",
        }
    )
    return list(data.get("query", {}).get("pages", {}).values())


def _query_search(query: str, limit: int) -> list[dict]:
    data = _api(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": str(limit),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": "640",
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
        "license_url": _plain((ext.get("LicenseUrl") or {}).get("value")),
    }


def _acceptable(meta: dict) -> bool:
    title = meta["title"].lower()
    if not meta["url"]:
        return False
    if not title.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return False
    if not any(word in title for word in ALLOWED):
        return False
    if any(word in title for word in BLOCKED):
        return False
    return True


def _download(url: str) -> np.ndarray | None:
    clean_url = url.split("?")[0]
    request = urllib.request.Request(clean_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read()
    except Exception as error:  # noqa: BLE001 - network failures are expected
        print(f"  download failed: {error}")
        return None
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=13)
    parser.add_argument("--out-dir", type=Path, default=NATURAL_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    existing_text = SOURCES.read_text() if SOURCES.exists() else ""
    existing_titles = set(re.findall(r"- Title: (.+)", existing_text))
    candidates: dict[str, dict] = {}
    for category in CATEGORIES:
        try:
            for page in _query_category(category, 50):
                meta = _metadata(page)
                if _acceptable(meta):
                    candidates[meta["title"]] = meta
        except Exception as error:  # noqa: BLE001
            print(f"category {category}: {error}")
    for query in QUERIES:
        try:
            for page in _query_search(query, 50):
                meta = _metadata(page)
                if _acceptable(meta):
                    candidates[meta["title"]] = meta
        except Exception as error:  # noqa: BLE001
            print(f"query {query}: {error}")

    print(f"candidates: {len(candidates)}")
    index = args.start_index
    saved: list[dict] = []
    for meta in candidates.values():
        if len(saved) >= args.count:
            break
        if not _acceptable(meta):
            continue
        if meta["title"] in existing_titles:
            continue
        name = f"office_{index:02d}.jpg"
        path = args.out_dir / name
        if path.exists():
            index += 1
            continue
        print(f"{name} <- {meta['title']}")
        if args.dry_run:
            saved.append({**meta, "file": name})
            index += 1
            continue
        image = _download(meta["url"])
        if image is None or image.shape[0] < 120 or image.shape[1] < 160:
            continue
        # Normalize the background size; the dataset cover-resizes anyway.
        height, width = image.shape[:2]
        if width > 960:
            scale = 960.0 / width
            image = cv2.resize(image, (960, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        saved.append({**meta, "file": name})
        index += 1
        time.sleep(0.2)

    if not saved:
        print("nothing saved")
        return
    if not args.dry_run:
        existing = SOURCES.read_text() if SOURCES.exists() else ""
        marker = "## Additional office images (Wikimedia Commons, auto-fetched)"
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
        SOURCES.write_text("\n".join(lines))
        print(f"saved {len(saved)} images; updated {SOURCES}")


if __name__ == "__main__":
    main()
