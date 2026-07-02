#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))

import numpy as np
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from plot_response_pca_images import find_images, load_response_matrix


def make_thumbnail(source: Path, dest: Path, size: int) -> None:
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (248, 250, 252))
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.paste(image, (x, y))
        canvas.save(dest, "JPEG", quality=82, optimize=True, progressive=True)


def standardize_layout(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    coords = coords - np.nanmean(coords, axis=0, keepdims=True)
    scale = np.nanstd(coords, axis=0, keepdims=True)
    scale[scale == 0] = 1
    coords = coords / scale
    return np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)


def rounded_coords(coords: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 6), round(float(y), 6), round(float(z), 6)] for x, y, z in coords]


def build_embeddings(response: np.ndarray, pca_precomponents: int, random_state: int) -> dict:
    pca_components = min(pca_precomponents, response.shape[0] - 1, response.shape[1])
    print(f"Computing PCA pre-reduction: {pca_components} components")
    pca_reduced = PCA(n_components=pca_components, random_state=random_state).fit_transform(response)

    print("Computing PCA 3D")
    pca3d = standardize_layout(pca_reduced[:, :3])

    print("Computing t-SNE 3D")
    tsne3d = standardize_layout(
        TSNE(
            n_components=3,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            metric="euclidean",
            random_state=random_state,
            max_iter=1000,
        ).fit_transform(pca_reduced)
    )

    print("Computing UMAP 3D")
    umap3d = standardize_layout(
        UMAP(
            n_components=3,
            n_neighbors=30,
            min_dist=0.08,
            metric="euclidean",
            random_state=random_state,
        ).fit_transform(pca_reduced)
    )

    return {
        "pca": {
            "label": "PCA",
            "axes": ["PC1", "PC2", "PC3"],
            "coordinates": rounded_coords(pca3d),
        },
        "tsne": {
            "label": "t-SNE",
            "axes": ["t-SNE 1", "t-SNE 2", "t-SNE 3"],
            "coordinates": rounded_coords(tsne3d),
        },
        "umap": {
            "label": "UMAP",
            "axes": ["UMAP 1", "UMAP 2", "UMAP 3"],
            "coordinates": rounded_coords(umap3d),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--stimuli-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("viewer/public/viewer-data"))
    parser.add_argument("--image-limit", type=int, default=1000)
    parser.add_argument("--reliability-threshold", type=float, default=0.4)
    parser.add_argument("--thumbnail-size", type=int, default=128)
    parser.add_argument("--pca-precomponents", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    image_paths = find_images(args.stimuli_dir)
    if len(image_paths) < args.image_limit:
        raise FileNotFoundError(f"Need {args.image_limit} images, found {len(image_paths)}")

    response = load_response_matrix(args.processed_dir, args.image_limit, args.reliability_threshold)
    embeddings = build_embeddings(response, args.pca_precomponents, args.random_state)

    output_dir = args.output_dir
    thumbs_dir = output_dir / "thumbs"
    output_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for index, image_path in enumerate(image_paths[: args.image_limit]):
        thumb_name = f"{index + 1:04d}.jpg"
        make_thumbnail(image_path, thumbs_dir / thumb_name, args.thumbnail_size)
        x, y, z = embeddings["pca"]["coordinates"][index]
        items.append(
            {
                "index": index + 1,
                "image": image_path.name,
                "thumb": f"thumbs/{thumb_name}",
                "pc1": x,
                "pc2": y,
                "pc3": z,
            }
        )

    payload = {
        "schema": "triple-n-pca-3d-v2",
        "source": "Triple-N Processed response_best, reliability_best >= threshold",
        "imageCount": len(items),
        "unitCount": int(response.shape[1]),
        "sessionCount": len(list(args.processed_dir.glob("Processed_ses*.mat"))),
        "reliabilityThreshold": args.reliability_threshold,
        "thumbnailSize": args.thumbnail_size,
        "defaultEmbedding": "pca",
        "embeddings": embeddings,
        "items": items,
    }
    out_path = output_dir / "pca3d.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {len(items)} thumbnails to {thumbs_dir}")


if __name__ == "__main__":
    main()
