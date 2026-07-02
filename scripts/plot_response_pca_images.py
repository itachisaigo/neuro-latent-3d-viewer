#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm


def load_mat(path: Path) -> dict:
    try:
        return sio.loadmat(path, simplify_cells=True)
    except NotImplementedError as exc:
        raise ValueError(f"{path} is MATLAB v7.3/HDF5; convert or load with h5py first") from exc


def natural_key(path: Path) -> list[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def find_images(stimuli_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for sub in ["StimuliNNN", "images", ""]:
        base = stimuli_dir / sub if sub else stimuli_dir
        if base.exists():
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                candidates.extend(base.rglob(pattern))
    unique = sorted(set(candidates), key=natural_key)
    if len(unique) < 1000:
        zip_path = stimuli_dir / "StimuliNNN.zip"
        if zip_path.exists():
            raise FileNotFoundError(
                f"Found {len(unique)} images. Extract stimuli first: unzip -n {zip_path} -d {stimuli_dir}"
            )
    return unique[:1000]


def load_response_matrix(processed_dir: Path, image_limit: int, reliability_threshold: float) -> np.ndarray:
    files = sorted(processed_dir.glob("Processed_ses*.mat"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No Processed_ses*.mat files found in {processed_dir}")

    chunks: list[np.ndarray] = []
    unit_counts: list[int] = []
    for file in tqdm(files, desc="loading processed sessions"):
        mat = load_mat(file)
        if "response_best" not in mat:
            continue
        response = np.asarray(mat["response_best"], dtype=float)
        if response.ndim != 2 or response.shape[1] < image_limit:
            continue

        keep = np.ones(response.shape[0], dtype=bool)
        if "reliability_best" in mat:
            reliability = np.asarray(mat["reliability_best"], dtype=float).reshape(-1)
            if reliability.shape[0] == response.shape[0]:
                keep &= reliability >= reliability_threshold

        selected = response[keep, :image_limit]
        if selected.size == 0:
            continue

        mean = np.nanmean(selected, axis=1, keepdims=True)
        std = np.nanstd(selected, axis=1, keepdims=True)
        std[std == 0] = 1
        selected = (selected - mean) / std
        selected = np.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)
        chunks.append(selected)
        unit_counts.append(selected.shape[0])

    if not chunks:
        raise ValueError("No usable response_best matrices after filtering")

    features_by_image = np.concatenate(chunks, axis=0).T
    print(f"Loaded {features_by_image.shape[1]} units across {len(unit_counts)} sessions.")
    return features_by_image


def farthest_point_subset(points: np.ndarray, n: int) -> np.ndarray:
    if points.shape[0] <= n:
        return np.arange(points.shape[0])

    center = np.median(points, axis=0)
    first = int(np.argmin(np.linalg.norm(points - center, axis=1)))
    selected = [first]
    min_dist = np.linalg.norm(points - points[first], axis=1)

    while len(selected) < n:
        best = int(np.argmax(min_dist))
        selected.append(best)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[best], axis=1))

    return np.asarray(selected, dtype=int)


def choose_thumbnail_indices(coords: np.ndarray, n: int, grid_factor: float) -> np.ndarray:
    if coords.shape[0] <= n:
        return np.arange(coords.shape[0])

    mins = coords.min(axis=0)
    spans = coords.max(axis=0) - mins
    spans[spans == 0] = 1
    aspect = max(spans[0] / spans[1], 0.1)
    target_cells = max(n, int(np.ceil(n * max(grid_factor, 1.0))))
    cols = max(1, int(np.ceil(np.sqrt(target_cells * aspect))))
    rows = max(1, int(np.ceil(target_cells / cols)))

    scaled = (coords - mins) / spans
    cell_cols = np.clip((scaled[:, 0] * cols).astype(int), 0, cols - 1)
    cell_rows = np.clip((scaled[:, 1] * rows).astype(int), 0, rows - 1)

    cells: dict[tuple[int, int], list[int]] = {}
    for idx in range(coords.shape[0]):
        cell = (int(cell_cols[idx]), int(cell_rows[idx]))
        cells.setdefault(cell, []).append(idx)

    cell_keys = list(cells)
    cell_centers = np.array(
        [
            mins
            + np.array([(col + 0.5) / cols, (row + 0.5) / rows])
            * spans
            for col, row in cell_keys
        ]
    )
    chosen_cell_positions = farthest_point_subset(cell_centers, min(n, len(cell_keys)))

    chosen: list[int] = []
    for cell_pos in chosen_cell_positions:
        cell = cell_keys[int(cell_pos)]
        candidates = np.asarray(cells[cell], dtype=int)
        center = cell_centers[int(cell_pos)]
        closest = candidates[np.argmin(np.linalg.norm(coords[candidates] - center, axis=1))]
        chosen.append(int(closest))

    if len(chosen) < n:
        all_indices = np.arange(coords.shape[0])
        remaining = np.setdiff1d(all_indices, np.asarray(chosen, dtype=int), assume_unique=False)
        selected_coords = coords[np.asarray(chosen, dtype=int)]
        min_dist = np.min(
            np.linalg.norm(coords[remaining, None, :] - selected_coords[None, :, :], axis=2),
            axis=1,
        )
        while len(chosen) < n and remaining.size:
            best_pos = int(np.argmax(min_dist))
            best_idx = int(remaining[best_pos])
            chosen.append(best_idx)
            keep = np.arange(remaining.size) != best_pos
            remaining = remaining[keep]
            min_dist = min_dist[keep]
            if remaining.size:
                min_dist = np.minimum(min_dist, np.linalg.norm(coords[remaining] - coords[best_idx], axis=1))

    return np.asarray(chosen[:n], dtype=int)


def plot(
    coords: np.ndarray,
    image_paths: list[Path],
    output: Path,
    thumbnails: int,
    zoom: float,
    grid_factor: float,
    point_alpha: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 11), dpi=180)
    ax.scatter(coords[:, 0], coords[:, 1], s=7, c="#1f2937", alpha=point_alpha, linewidths=0)

    chosen = choose_thumbnail_indices(coords, thumbnails, grid_factor)
    for idx in chosen:
        img = Image.open(image_paths[idx]).convert("RGB")
        img.thumbnail((72, 72))
        ab = AnnotationBbox(
            OffsetImage(np.asarray(img), zoom=zoom),
            (coords[idx, 0], coords[idx, 1]),
            frameon=True,
            bboxprops={"edgecolor": "white", "linewidth": 0.8, "boxstyle": "round,pad=0.02"},
            pad=0.01,
        )
        ax.add_artist(ab)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Triple-N natural images in neural response PCA space")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.set_facecolor("#f8fafc")
    fig.tight_layout()
    fig.savefig(output)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--stimuli-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/response_pca_images.png"))
    parser.add_argument("--image-limit", type=int, default=1000)
    parser.add_argument("--reliability-threshold", type=float, default=0.4)
    parser.add_argument("--thumbnails", type=int, default=240)
    parser.add_argument("--zoom", type=float, default=0.44)
    parser.add_argument("--thumbnail-grid-factor", type=float, default=2.0)
    parser.add_argument("--point-alpha", type=float, default=0.16)
    args = parser.parse_args()

    image_paths = find_images(args.stimuli_dir)
    if len(image_paths) < args.image_limit:
        raise FileNotFoundError(f"Need {args.image_limit} images, found {len(image_paths)}")

    response = load_response_matrix(args.processed_dir, args.image_limit, args.reliability_threshold)
    coords = PCA(n_components=2, random_state=0).fit_transform(response)

    scale = np.nanstd(coords, axis=0)
    scale[scale == 0] = 1
    coords = coords / scale
    plot(
        coords,
        image_paths[: args.image_limit],
        args.output,
        args.thumbnails,
        args.zoom,
        args.thumbnail_grid_factor,
        args.point_alpha,
    )


if __name__ == "__main__":
    main()
