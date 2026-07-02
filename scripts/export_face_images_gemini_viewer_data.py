#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from api_embedding_clients import gemini_embed_images
from viewer_export_utils import build_layouts, l2_normalize, make_thumbnail, update_manifest, write_lightweight_json


def find_images(images_dir: Path, limit: int) -> list[Path]:
    paths = sorted(
        [
            *images_dir.glob("*.jpg"),
            *images_dir.glob("*.jpeg"),
            *images_dir.glob("*.png"),
            *images_dir.glob("*.webp"),
            *images_dir.glob("*.bmp"),
        ]
    )
    return paths[:limit]


def load_metadata(metadata_json: Path | None) -> dict:
    if not metadata_json:
        return {}
    rows = json.loads(metadata_json.read_text(encoding="utf-8"))
    return {row.get("file"): row for row in rows if row.get("file")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a lightweight 3D face-image latent viewer dataset using Gemini API.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("viewer/public/viewer-data/face-gemini"))
    parser.add_argument("--manifest", type=Path, default=Path("viewer/public/viewer-data/datasets.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/api_embeddings"))
    parser.add_argument("--dataset-id", default="face-gemini")
    parser.add_argument("--short-label", default="Face images")
    parser.add_argument("--label", default="Face images + Gemini")
    parser.add_argument("--source-label", default="Free face image folder plus Gemini API embeddings")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--thumbnail-size", type=int, default=128)
    parser.add_argument("--model", default="gemini-embedding-2")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--image-max-side", type=int, default=512)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--pca-precomponents", type=int, default=50)
    parser.add_argument("--no-manifest-update", action="store_true")
    args = parser.parse_args()

    image_paths = find_images(args.images_dir, args.limit)
    if len(image_paths) < 4:
        raise ValueError(f"Need at least 4 images, found {len(image_paths)}")
    metadata_by_file = load_metadata(args.metadata_json or (args.images_dir / "metadata.json"))

    model_slug = args.model.replace("/", "_").replace(":", "_")
    cache_path = args.cache_dir / f"{args.dataset_id}_{model_slug}_{len(image_paths)}.npy"
    features = gemini_embed_images(
        image_paths,
        cache_path=cache_path,
        model=args.model,
        batch_size=args.batch_size,
        image_max_side=args.image_max_side,
    )
    model_label = "Gemini" if args.model.startswith("gemini") else args.model
    embeddings = build_layouts(
        l2_normalize(features),
        key_prefix="face",
        label_prefix=model_label,
        random_state=args.random_state,
        pca_precomponents=args.pca_precomponents,
    )

    thumbs_dir = args.output_dir / "thumbs"
    items = []
    for index, image_path in enumerate(tqdm(image_paths, desc="Writing thumbnails")):
        thumb_name = f"{index + 1:04d}.jpg"
        make_thumbnail(image_path, thumbs_dir / thumb_name, args.thumbnail_size)
        meta = metadata_by_file.get(image_path.name, {})
        items.append(
            {
                "index": index + 1,
                "id": f"{args.dataset_id}_{index + 1:04d}",
                "label": image_path.stem,
                "image": image_path.name,
                "thumb": f"thumbs/{thumb_name}",
                "mediaType": "image",
                "category": "face",
                "caption": meta.get("description") or meta.get("title", ""),
                "sourceUrl": meta.get("sourceUrl", ""),
                "license": meta.get("license", ""),
                "artist": meta.get("artist", ""),
            }
        )

    payload = {
        "schema": "neuro-latent-viewer-dataset-v1",
        "id": args.dataset_id,
        "label": args.label,
        "source": args.source_label,
        "itemCount": len(items),
        "imageCount": len(items),
        "itemLabel": "face images",
        "signalLabel": "Gemini image latent",
        "thumbnailSize": args.thumbnail_size,
        "defaultEmbedding": "face_ppca",
        "embeddings": embeddings,
        "items": items,
    }
    out_path = args.output_dir / "index.json"
    write_lightweight_json(out_path, payload)
    print(f"Wrote lightweight viewer JSON: {out_path} ({out_path.stat().st_size / 1024:.1f} KiB)")

    if not args.no_manifest_update:
        update_manifest(
            args.manifest,
            [
                {
                    "id": args.dataset_id,
                    "label": args.label,
                    "shortLabel": args.short_label,
                    "href": f"{args.output_dir.name}/index.json",
                    "stimulus": f"{len(items):,} face images",
                    "signal": "Gemini image latent",
                    "subject": "Face image set",
                    "description": "Lightweight 3D layouts for free face images embedded with Gemini.",
                }
            ],
        )
        print(f"Updated manifest: {args.manifest}")


if __name__ == "__main__":
    main()
