#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote

import requests
from tqdm import tqdm

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_QUERIES = [
    'incategory:"Portrait photographs of men" face -statue -sculpture -painting -drawing -engraving',
    'incategory:"Portrait photographs of women" face -statue -sculpture -painting -drawing -engraving',
    'incategory:"Portrait photographs of children" face -statue -sculpture -painting -drawing -engraving',
    'headshot photograph human face -statue -sculpture -painting -drawing -engraving',
    'portrait photograph human face -statue -sculpture -painting -drawing -engraving',
]
DEFAULT_CATEGORIES = [
    "Portrait photographs",
    "Portrait photographs of men",
    "Portrait photographs of women",
    "Portrait photographs of children",
    "Headshots",
    "Close-ups of human faces",
    "Human faces",
]
DEFAULT_ALLOWED_LICENSE_PREFIXES = (
    "cc by",
    "cc-by",
    "cc0",
    "public domain",
    "pd",
)
NON_PHOTO_TERMS = {
    "bronze",
    "bust",
    "caricature",
    "drawing",
    "engraving",
    "line engraving",
    "monument",
    "painting",
    "portrait of princess",
    "scabbard",
    "sculpture",
    "statue",
}


def get_json_with_retry(session: requests.Session, params: dict, timeout: int = 60) -> dict:
    for attempt in range(4):
        response = session.get(COMMONS_API, params=params, timeout=timeout)
        if response.status_code == 429:
            time.sleep(3 + attempt * 4)
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()
    return response.json()


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub("<[^<]+?>", "", value)
    return html.unescape(value).strip()


def safe_name(title: str, pageid: int) -> str:
    title = title.replace("File:", "")
    stem = Path(unquote(title)).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return f"commons_{pageid}_{stem[:60]}.jpg"


def search_commons(query: str, limit: int, page_size: int) -> list[dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "a.s.ist-neuro-latent-viewer/0.1"})
    rows = []
    cont = {}

    while len(rows) < limit:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": query,
            "gsrlimit": min(page_size, limit - len(rows)),
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": 640,
            **cont,
        }
        payload = get_json_with_retry(session, params)
        pages = payload.get("query", {}).get("pages", {})
        rows.extend(pages.values())
        if "continue" not in payload:
            break
        cont = payload["continue"]
        time.sleep(0.2)
    return rows


def category_members(category: str, limit: int, page_size: int) -> list[dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "a.s.ist-neuro-latent-viewer/0.1"})
    rows = []
    cont = {}
    category_title = category if category.startswith("Category:") else f"Category:{category}"

    while len(rows) < limit:
        params = {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": category_title,
            "gcmtype": "file",
            "gcmnamespace": 6,
            "gcmlimit": min(page_size, limit - len(rows)),
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": 640,
            **cont,
        }
        payload = get_json_with_retry(session, params)
        pages = payload.get("query", {}).get("pages", {})
        rows.extend(pages.values())
        if "continue" not in payload:
            break
        cont = payload["continue"]
        time.sleep(0.2)
    return rows


def metadata_from_page(page: dict) -> dict | None:
    imageinfo = page.get("imageinfo") or []
    if not imageinfo:
        return None
    info = imageinfo[0]
    mime = info.get("mime", "")
    if mime not in {"image/jpeg", "image/png"}:
        return None

    ext = info.get("extmetadata", {})
    license_short = clean_text(ext.get("LicenseShortName", {}).get("value")).lower()
    if license_short and not is_allowed_license(license_short):
        return None

    return {
        "title": page.get("title", ""),
        "pageid": page.get("pageid"),
        "url": info.get("thumburl") or info.get("url"),
        "sourceUrl": info.get("descriptionurl"),
        "mime": mime,
        "license": clean_text(ext.get("LicenseShortName", {}).get("value")),
        "artist": clean_text(ext.get("Artist", {}).get("value")),
        "credit": clean_text(ext.get("Credit", {}).get("value")),
        "description": clean_text(ext.get("ImageDescription", {}).get("value")),
    }


def is_allowed_license(license_short: str) -> bool:
    normalized = re.sub(r"\s+", " ", license_short.strip().lower())
    return any(normalized.startswith(prefix) for prefix in DEFAULT_ALLOWED_LICENSE_PREFIXES)


def looks_like_non_photo(meta: dict) -> bool:
    haystack = " ".join(
        [
            meta.get("title", ""),
            meta.get("description", ""),
            meta.get("credit", ""),
        ]
    ).lower()
    return any(term in haystack for term in NON_PHOTO_TERMS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small CC/PD portrait image set from Wikimedia Commons.")
    parser.add_argument("--query", action="append", help="Commons search query. Repeat to merge multiple searches.")
    parser.add_argument("--category", action="append", help="Commons category name. Repeat to merge multiple categories.")
    parser.add_argument("--no-default-categories", action="store_true")
    parser.add_argument("--no-default-queries", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/free_faces/wikimedia"))
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--page-size", type=int, default=40)
    parser.add_argument("--allow-non-photo", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.json"
    rows = []
    session = requests.Session()
    session.headers.update({"User-Agent": "a.s.ist-neuro-latent-viewer/0.1"})

    categories = args.category or ([] if args.no_default_categories else DEFAULT_CATEGORIES)
    queries = args.query or ([] if args.no_default_queries else DEFAULT_QUERIES)
    seen_pageids = set()
    sources = [("category", category) for category in categories] + [("query", query) for query in queries]
    for source_type, value in sources:
        if source_type == "category":
            pages = category_members(value, args.limit * 4, args.page_size)
            desc = f"Downloading Wikimedia portraits: Category:{value[:33]}"
        else:
            pages = search_commons(value, args.limit * 4, args.page_size)
            desc = f"Downloading Wikimedia portraits: {value[:42]}"
        for page in tqdm(pages, desc=desc):
            if len(rows) >= args.limit:
                break
            pageid = page.get("pageid")
            if pageid in seen_pageids:
                continue
            seen_pageids.add(pageid)
            meta = metadata_from_page(page)
            if not meta or not meta.get("url"):
                continue
            if not args.allow_non_photo and looks_like_non_photo(meta):
                continue
            dest = args.output_dir / safe_name(meta["title"], int(meta["pageid"]))
            if not dest.exists():
                image = session.get(meta["url"], timeout=60)
                if image.status_code >= 400:
                    continue
                dest.write_bytes(image.content)
                time.sleep(0.1)
            meta["file"] = dest.name
            rows.append(meta)
        if len(rows) >= args.limit:
            break

    metadata_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} images to {args.output_dir}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
