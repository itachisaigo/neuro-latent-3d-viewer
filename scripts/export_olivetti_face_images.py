#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Olivetti faces as local JPEG stimuli with metadata.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/free_faces/olivetti"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()

    data = fetch_olivetti_faces(download_if_missing=True, shuffle=False)
    images = data.images[: args.limit]
    targets = data.target[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (image, target) in enumerate(zip(images, targets, strict=False), start=1):
        array = np.clip(image * 255, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(array, mode="L").convert("RGB")
        pil_image = pil_image.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
        file_name = f"olivetti_{index:04d}.jpg"
        pil_image.save(args.output_dir / file_name, "JPEG", quality=90, optimize=True, progressive=True)
        rows.append(
            {
                "file": file_name,
                "title": f"Olivetti face {index:04d}",
                "description": "Olivetti faces research dataset image.",
                "sourceUrl": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html",
                "license": "Research dataset",
                "artist": "AT&T Laboratories Cambridge",
                "subjectId": int(target),
            }
        )

    (args.output_dir / "metadata.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} Olivetti face images to {args.output_dir}")


if __name__ == "__main__":
    main()
