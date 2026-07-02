#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps
from tqdm import tqdm

from plot_response_pca_images import find_images, load_response_matrix
from viewer_export_utils import (
    build_layouts,
    l2_normalize,
    make_thumbnail,
    update_manifest,
    write_lightweight_json,
)

JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


def image_to_data_uri(path: Path, max_side: int, quality: int) -> str:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def parse_jina_embeddings(response_json: dict) -> np.ndarray:
    if "data" not in response_json:
        raise ValueError(f"Unexpected Jina response: {response_json.keys()}")
    rows = sorted(response_json["data"], key=lambda row: row.get("index", 0))
    return np.asarray([row["embedding"] for row in rows], dtype=np.float32)


def parse_gemini_embeddings(response_json: dict) -> np.ndarray:
    embeddings = response_json.get("embeddings")
    if embeddings is None and "responses" in response_json:
        embeddings = [row.get("embedding", row) for row in response_json["responses"]]
    if embeddings is None:
        raise ValueError(f"Unexpected Gemini response: {response_json.keys()}")

    values = []
    for row in embeddings:
        if "values" in row:
            values.append(row["values"])
        elif "embedding" in row and "values" in row["embedding"]:
            values.append(row["embedding"]["values"])
        else:
            raise ValueError(f"Unexpected Gemini embedding row: {row.keys()}")
    return np.asarray(values, dtype=np.float32)


def embed_images_with_gemini(
    image_paths: list[Path],
    cache_path: Path,
    model: str,
    api_key: str,
    batch_size: int,
    image_max_side: int,
    jpeg_quality: int,
    sleep_seconds: float,
    timeout: int,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(image_paths):
            print(f"Using cached API embeddings: {cache_path}")
            return cached
        print(f"Ignoring cache with mismatched shape: {cached.shape}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = model if model.startswith("models/") else f"models/{model}"
    endpoint_model = model_name.split("/", 1)[1]
    endpoint = f"{GEMINI_API_ROOT}/models/{endpoint_model}:batchEmbedContents"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    batch_size = min(batch_size, 6)

    chunks = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc=f"Embedding images with {model_name}"):
        batch_paths = image_paths[start : start + batch_size]
        requests_payload = []
        for path in batch_paths:
            image_data = image_to_data_uri(path, image_max_side, jpeg_quality).split(",", 1)[1]
            requests_payload.append(
                {
                    "model": model_name,
                    "content": {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_data,
                                }
                            }
                        ]
                    },
                }
            )
        response = requests.post(endpoint, headers=headers, json={"requests": requests_payload}, timeout=timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"Gemini API failed ({response.status_code}): {response.text[:800]}")
        chunks.append(parse_gemini_embeddings(response.json()))
        if sleep_seconds:
            time.sleep(sleep_seconds)

    embeddings = np.vstack(chunks)
    np.save(cache_path, embeddings)
    print(f"Wrote API embedding cache: {cache_path}")
    return embeddings


def embed_images_with_jina(
    image_paths: list[Path],
    cache_path: Path,
    model: str,
    api_key: str,
    batch_size: int,
    image_max_side: int,
    jpeg_quality: int,
    sleep_seconds: float,
    timeout: int,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(image_paths):
            print(f"Using cached API embeddings: {cache_path}")
            return cached
        print(f"Ignoring cache with mismatched shape: {cached.shape}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    chunks = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc=f"Embedding images with {model}"):
        batch_paths = image_paths[start : start + batch_size]
        inputs = [{"image": image_to_data_uri(path, image_max_side, jpeg_quality)} for path in batch_paths]
        payload = {
            "model": model,
            "normalized": True,
            "embedding_type": "float",
            "input": inputs,
        }
        response = requests.post(JINA_EMBEDDINGS_URL, headers=headers, json=payload, timeout=timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"Jina API failed ({response.status_code}): {response.text[:800]}")
        chunks.append(parse_jina_embeddings(response.json()))
        if sleep_seconds:
            time.sleep(sleep_seconds)

    embeddings = np.vstack(chunks)
    np.save(cache_path, embeddings)
    print(f"Wrote API embedding cache: {cache_path}")
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight Triple-N viewer data with brain and API image latent spaces.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--stimuli-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("viewer/public/viewer-data/triplen-api"))
    parser.add_argument("--manifest", type=Path, default=Path("viewer/public/viewer-data/datasets.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/api_embeddings"))
    parser.add_argument("--image-limit", type=int, default=1000)
    parser.add_argument("--reliability-threshold", type=float, default=0.4)
    parser.add_argument("--thumbnail-size", type=int, default=128)
    parser.add_argument("--pca-precomponents", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--provider", choices=["gemini", "jina"], default="gemini")
    parser.add_argument("--model", default="gemini-embedding-2")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--image-max-side", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=86)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--skip-api", action="store_true", help="Write only brain layouts; useful for testing the export path.")
    parser.add_argument("--no-manifest-update", action="store_true")
    args = parser.parse_args()

    image_paths = find_images(args.stimuli_dir)[: args.image_limit]
    if len(image_paths) < args.image_limit:
        raise FileNotFoundError(f"Need {args.image_limit} images, found {len(image_paths)}")

    response = load_response_matrix(args.processed_dir, args.image_limit, args.reliability_threshold)
    embeddings = build_layouts(
        response,
        key_prefix="brain",
        label_prefix="Brain",
        random_state=args.random_state,
        pca_precomponents=args.pca_precomponents,
    )

    model_slug = args.model.replace("/", "_").replace(":", "_")
    model_label = "Gemini" if args.provider == "gemini" else args.model
    if not args.skip_api:
        api_key_env = args.api_key_env or ("GEMINI_API_KEY" if args.provider == "gemini" else "JINA_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(f"{api_key_env} is not set. Export needs an API key for {args.provider}.")

        cache_path = args.cache_dir / f"triplen_{model_slug}_{args.image_limit}.npy"
        if args.provider == "gemini":
            api_features = embed_images_with_gemini(
                image_paths=image_paths,
                cache_path=cache_path,
                model=args.model,
                api_key=api_key,
                batch_size=args.batch_size,
                image_max_side=args.image_max_side,
                jpeg_quality=args.jpeg_quality,
                sleep_seconds=args.sleep_seconds,
                timeout=args.timeout,
            )
        else:
            api_features = embed_images_with_jina(
                image_paths=image_paths,
                cache_path=cache_path,
                model=args.model,
                api_key=api_key,
                batch_size=args.batch_size,
                image_max_side=args.image_max_side,
                jpeg_quality=args.jpeg_quality,
                sleep_seconds=args.sleep_seconds,
                timeout=args.timeout,
            )
        embeddings.update(
            build_layouts(
                l2_normalize(api_features),
                key_prefix="ai",
                label_prefix=model_label,
                random_state=args.random_state,
                pca_precomponents=args.pca_precomponents,
            )
        )

    output_dir = args.output_dir
    thumbs_dir = output_dir / "thumbs"
    output_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for index, image_path in enumerate(tqdm(image_paths, desc="Writing thumbnails")):
        thumb_name = f"{index + 1:04d}.jpg"
        make_thumbnail(image_path, thumbs_dir / thumb_name, args.thumbnail_size)
        items.append(
            {
                "index": index + 1,
                "id": f"triplen_{index + 1:04d}",
                "label": image_path.name,
                "image": image_path.name,
                "thumb": f"thumbs/{thumb_name}",
                "mediaType": "image",
            }
        )

    payload = {
        "schema": "neuro-latent-viewer-dataset-v1",
        "id": "triple-n-api",
        "label": f"Triple-N brain + {args.model}",
        "source": "Triple-N Processed response_best plus API image embeddings",
        "itemCount": len(items),
        "imageCount": len(items),
        "itemLabel": "images",
        "signalLabel": f"{int(response.shape[1]):,} Neuropixels units",
        "sessionCount": len(list(args.processed_dir.glob("Processed_ses*.mat"))),
        "reliabilityThreshold": args.reliability_threshold,
        "thumbnailSize": args.thumbnail_size,
        "defaultEmbedding": "brain_ppca",
        "embeddings": embeddings,
        "items": items,
    }
    out_path = output_dir / "index.json"
    write_lightweight_json(out_path, payload)
    print(f"Wrote lightweight viewer JSON: {out_path} ({out_path.stat().st_size / 1024:.1f} KiB)")

    if not args.no_manifest_update and not args.skip_api:
        update_manifest(
            args.manifest,
            [
                {
                    "id": "triple-n-api",
                    "label": f"Triple-N brain + {args.model}",
                    "shortLabel": "Triple-N + AI",
                    "href": "triplen-api/index.json",
                    "stimulus": f"{len(items):,} natural images",
                    "signal": "Neuropixels + API image latent",
                    "subject": "Macaque visual cortex",
                    "description": "Lightweight 3D layouts for Triple-N neural responses and large API image embeddings.",
                }
            ],
        )
        print(f"Updated manifest: {args.manifest}")


if __name__ == "__main__":
    main()
