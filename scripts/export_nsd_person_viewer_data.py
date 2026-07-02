#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from api_embedding_clients import gemini_embed_images
from viewer_export_utils import build_layouts, l2_normalize, make_thumbnail, update_manifest, write_lightweight_json


def image_id_from_name(path: Path) -> int | None:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        return None
    return int(digits[-12:])


def load_person_image_ids(coco_instances: Path) -> set[int]:
    data = json.loads(coco_instances.read_text(encoding="utf-8"))
    person_ids = {cat["id"] for cat in data["categories"] if cat["name"] == "person"}
    image_ids = {ann["image_id"] for ann in data["annotations"] if ann["category_id"] in person_ids}
    return image_ids


def load_nsd_person_rows(stim_info_csv: Path, coco_image_ids: set[int]) -> list[dict]:
    rows = []
    with stim_info_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            coco_id = row.get("cocoId") or row.get("coco_id") or row.get("cocoID")
            if not coco_id:
                continue
            try:
                coco_id_int = int(float(coco_id))
            except ValueError:
                continue
            if coco_id_int in coco_image_ids:
                rows.append(row)
    return rows


def export_images_from_nsd_hdf5(nsd_hdf5: Path, rows: list[dict], output_dir: Path, limit: int) -> list[Path]:
    import h5py

    image_dir = output_dir / "stimuli"
    image_dir.mkdir(parents=True, exist_ok=True)
    selected = rows[:limit]
    paths = []

    with h5py.File(nsd_hdf5, "r") as h5:
        key = "imgBrick" if "imgBrick" in h5 else next(iter(h5.keys()))
        images = h5[key]
        for row in tqdm(selected, desc="Exporting NSD person images"):
            nsd_id = row.get("nsdId") or row.get("nsd_id") or row.get("nsdID")
            if not nsd_id:
                continue
            index = int(float(nsd_id)) - 1
            image = Image.fromarray(np.asarray(images[index], dtype=np.uint8))
            dest = image_dir / f"nsd_{index + 1:06d}.jpg"
            if not dest.exists():
                image.save(dest, "JPEG", quality=88, optimize=True)
            paths.append(dest)
    return paths


def collect_images_from_dir(images_dir: Path, coco_instances: Path | None, limit: int) -> list[Path]:
    paths = sorted(
        [
            *images_dir.glob("*.jpg"),
            *images_dir.glob("*.jpeg"),
            *images_dir.glob("*.png"),
            *images_dir.glob("*.bmp"),
        ]
    )
    if coco_instances:
        person_ids = load_person_image_ids(coco_instances)
        paths = [path for path in paths if (image_id_from_name(path) in person_ids)]
    return paths[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a lightweight NSD person/face-like image latent viewer dataset.")
    parser.add_argument("--images-dir", type=Path, help="Directory of NSD/COCO images. If provided, these files are used directly.")
    parser.add_argument("--nsd-hdf5", type=Path, help="NSD stimuli HDF5, e.g. nsd_stimuli.hdf5.")
    parser.add_argument("--stim-info-csv", type=Path, help="NSD stimulus metadata with nsdId and cocoId columns.")
    parser.add_argument("--coco-instances", type=Path, help="COCO instances JSON used to select person images.")
    parser.add_argument("--output-dir", type=Path, default=Path("viewer/public/viewer-data/nsd-person-gemini"))
    parser.add_argument("--manifest", type=Path, default=Path("viewer/public/viewer-data/datasets.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/api_embeddings"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--thumbnail-size", type=int, default=128)
    parser.add_argument("--model", default="gemini-embedding-2")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--image-max-side", type=int, default=512)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--pca-precomponents", type=int, default=50)
    parser.add_argument("--no-manifest-update", action="store_true")
    args = parser.parse_args()

    if args.images_dir:
        image_paths = collect_images_from_dir(args.images_dir, args.coco_instances, args.limit)
    elif args.nsd_hdf5 and args.stim_info_csv and args.coco_instances:
        rows = load_nsd_person_rows(args.stim_info_csv, load_person_image_ids(args.coco_instances))
        image_paths = export_images_from_nsd_hdf5(args.nsd_hdf5, rows, args.output_dir, args.limit)
    else:
        raise SystemExit("Provide either --images-dir, or --nsd-hdf5 + --stim-info-csv + --coco-instances.")

    if len(image_paths) < 4:
        raise ValueError(f"Need at least 4 images, found {len(image_paths)}")

    model_slug = args.model.replace("/", "_").replace(":", "_")
    model_label = "Gemini" if args.model.startswith("gemini") else args.model
    cache_path = args.cache_dir / f"nsd_person_{model_slug}_{len(image_paths)}.npy"
    features = gemini_embed_images(
        image_paths,
        cache_path=cache_path,
        model=args.model,
        batch_size=args.batch_size,
        image_max_side=args.image_max_side,
    )
    embeddings = build_layouts(
        l2_normalize(features),
        key_prefix="ai",
        label_prefix=model_label,
        random_state=args.random_state,
        pca_precomponents=args.pca_precomponents,
    )

    thumbs_dir = args.output_dir / "thumbs"
    items = []
    for index, image_path in enumerate(tqdm(image_paths, desc="Writing thumbnails")):
        thumb_name = f"{index + 1:04d}.jpg"
        make_thumbnail(image_path, thumbs_dir / thumb_name, args.thumbnail_size)
        items.append(
            {
                "index": index + 1,
                "id": f"nsd_person_{index + 1:04d}",
                "label": image_path.name,
                "image": image_path.name,
                "thumb": f"thumbs/{thumb_name}",
                "mediaType": "image",
                "category": "person/face candidate",
            }
        )

    payload = {
        "schema": "neuro-latent-viewer-dataset-v1",
        "id": "nsd-person-gemini",
        "label": f"NSD person candidates + {args.model}",
        "source": "NSD/COCO person-filtered stimuli plus Gemini API embeddings",
        "itemCount": len(items),
        "imageCount": len(items),
        "itemLabel": "person images",
        "signalLabel": "Gemini API image latent",
        "thumbnailSize": args.thumbnail_size,
        "defaultEmbedding": "ai_ppca",
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
                    "id": "nsd-person-gemini",
                    "label": f"NSD person candidates + {args.model}",
                    "shortLabel": "NSD person",
                    "href": "nsd-person-gemini/index.json",
                    "stimulus": f"{len(items):,} person/face candidate images",
                    "signal": "Gemini image latent",
                    "subject": "Human fMRI stimulus set",
                    "description": "Lightweight 3D layouts for NSD/COCO person images embedded with Gemini.",
                }
            ],
        )
        print(f"Updated manifest: {args.manifest}")


if __name__ == "__main__":
    main()
