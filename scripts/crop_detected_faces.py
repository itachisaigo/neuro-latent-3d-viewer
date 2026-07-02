#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageOps
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(input_dirs: list[Path]) -> list[Path]:
    paths = []
    seen = set()
    for input_dir in input_dirs:
        for path in sorted(input_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                continue
            key = path.name
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def load_metadata(input_dirs: list[Path]) -> dict[str, dict]:
    rows = {}
    for input_dir in input_dirs:
        metadata_path = input_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        for row in json.loads(metadata_path.read_text(encoding="utf-8")):
            if row.get("file"):
                rows[row["file"]] = row
    return rows


def rect_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


def dedupe_rects(rects: list[tuple[int, int, int, int]], iou_threshold: float = 0.35) -> list[tuple[int, int, int, int]]:
    rects = sorted(rects, key=lambda row: row[2] * row[3], reverse=True)
    kept = []
    for rect in rects:
        if all(rect_iou(rect, existing) < iou_threshold for existing in kept):
            kept.append(rect)
    return kept


def detect_faces(
    image_bgr,
    min_size: int,
    min_neighbors: int,
    frontal_only: bool,
) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade_dir = Path(cv2.data.haarcascades)
    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
    ]
    if not frontal_only:
        cascade_names.append("haarcascade_profileface.xml")
    rects = []
    for name in cascade_names:
        cascade = cv2.CascadeClassifier(str(cascade_dir / name))
        found = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        rects.extend(tuple(int(v) for v in rect) for rect in found)
    return dedupe_rects(rects)


def crop_square(image: Image.Image, rect: tuple[int, int, int, int], expand: float, output_size: int) -> Image.Image:
    x, y, w, h = rect
    cx = x + w / 2
    cy = y + h / 2
    side = max(w, h) * expand
    left = max(0, int(round(cx - side / 2)))
    top = max(0, int(round(cy - side / 2)))
    right = min(image.width, int(round(cx + side / 2)))
    bottom = min(image.height, int(round(cy + side / 2)))
    crop = image.crop((left, top, right, bottom))
    crop.thumbnail((output_size, output_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (output_size, output_size), (248, 250, 252))
    canvas.paste(crop.convert("RGB"), ((output_size - crop.width) // 2, (output_size - crop.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and crop face-only images from a folder of portrait photos.")
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-size", type=int, default=384)
    parser.add_argument("--min-face-size", type=int, default=48)
    parser.add_argument("--min-face-fraction", type=float, default=0.11)
    parser.add_argument("--min-neighbors", type=int, default=6)
    parser.add_argument("--max-faces-per-image", type=int, default=2)
    parser.add_argument("--frontal-only", action="store_true")
    parser.add_argument("--expand", type=float, default=1.75)
    args = parser.parse_args()

    image_paths = find_images(args.input_dir)
    metadata_by_file = load_metadata(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for image_path in tqdm(image_paths, desc="Detecting faces"):
        if len(rows) >= args.limit:
            break
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        rects = detect_faces(image_bgr, args.min_face_size, args.min_neighbors, args.frontal_only)
        image_min_side = min(image_bgr.shape[0], image_bgr.shape[1])
        rects = [
            rect
            for rect in rects
            if rect[2] >= image_min_side * args.min_face_fraction and rect[3] >= image_min_side * args.min_face_fraction
        ][: args.max_faces_per_image]
        if not rects:
            continue
        pil_image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        for face_index, rect in enumerate(rects, start=1):
            if len(rows) >= args.limit:
                break
            out_name = f"face_{len(rows) + 1:04d}.jpg"
            face = crop_square(pil_image, rect, args.expand, args.output_size)
            face.save(args.output_dir / out_name, "JPEG", quality=88, optimize=True, progressive=True)
            source_meta = metadata_by_file.get(image_path.name, {})
            rows.append(
                {
                    "file": out_name,
                    "sourceFile": image_path.name,
                    "title": source_meta.get("title") or image_path.stem,
                    "description": source_meta.get("description", ""),
                    "sourceUrl": source_meta.get("sourceUrl", ""),
                    "license": source_meta.get("license", ""),
                    "artist": source_meta.get("artist", ""),
                    "faceBox": list(rect),
                    "sourceWidth": pil_image.width,
                    "sourceHeight": pil_image.height,
                    "faceIndex": face_index,
                }
            )

    (args.output_dir / "metadata.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} cropped faces to {args.output_dir}")


if __name__ == "__main__":
    main()
