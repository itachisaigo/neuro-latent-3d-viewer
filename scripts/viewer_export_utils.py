from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))

import numpy as np
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


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


def l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return features / norm


def build_layouts(
    features: np.ndarray,
    key_prefix: str,
    label_prefix: str,
    random_state: int = 0,
    pca_precomponents: int = 50,
) -> dict:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got {features.shape}")
    if features.shape[0] < 4:
        raise ValueError("Need at least 4 stimuli to build 3D layouts")

    n_pre = min(pca_precomponents, features.shape[0] - 1, features.shape[1])
    reduced = PCA(n_components=n_pre, random_state=random_state).fit_transform(features)
    ppca3d = standardize_layout(reduced[:, :3])

    perplexity = max(2, min(30, (features.shape[0] - 1) // 3))
    tsne3d = standardize_layout(
        TSNE(
            n_components=3,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            metric="euclidean",
            random_state=random_state,
            max_iter=1000,
        ).fit_transform(reduced)
    )

    n_neighbors = max(2, min(30, features.shape[0] - 1))
    umap3d = standardize_layout(
        UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=0.08,
            metric="euclidean",
            random_state=random_state,
        ).fit_transform(reduced)
    )

    return {
        f"{key_prefix}_ppca": {
            "label": f"{label_prefix} PPCA",
            "axes": ["PPCA 1", "PPCA 2", "PPCA 3"],
            "coordinates": rounded_coords(ppca3d),
        },
        f"{key_prefix}_tsne": {
            "label": f"{label_prefix} t-SNE",
            "axes": ["t-SNE 1", "t-SNE 2", "t-SNE 3"],
            "coordinates": rounded_coords(tsne3d),
        },
        f"{key_prefix}_umap": {
            "label": f"{label_prefix} UMAP",
            "axes": ["UMAP 1", "UMAP 2", "UMAP 3"],
            "coordinates": rounded_coords(umap3d),
        },
    }


def write_lightweight_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def update_manifest(manifest_path: Path, entries: Iterable[dict], default_dataset: str | None = None) -> None:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema": "neuro-latent-viewer-manifest-v1",
            "defaultDataset": default_dataset,
            "datasets": [],
        }

    by_id = {entry["id"]: entry for entry in manifest.get("datasets", [])}
    for entry in entries:
        by_id[entry["id"]] = entry
    manifest["datasets"] = list(by_id.values())
    if default_dataset:
        manifest["defaultDataset"] = default_dataset

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
