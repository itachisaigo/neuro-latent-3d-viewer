from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps
from tqdm import tqdm

GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


def _gemini_model_names(model: str) -> tuple[str, str]:
    model_name = model if model.startswith("models/") else f"models/{model}"
    endpoint_model = model_name.split("/", 1)[1]
    return model_name, endpoint_model


def _parse_gemini_embeddings(response_json: dict) -> np.ndarray:
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


def _image_base64(path: Path, max_side: int, jpeg_quality: int) -> str:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def gemini_embed_parts(
    parts: list[dict],
    cache_path: Path,
    model: str = "gemini-embedding-2",
    api_key: str | None = None,
    api_key_env: str = "GEMINI_API_KEY",
    batch_size: int = 6,
    sleep_seconds: float = 0.2,
    timeout: int = 180,
    description: str = "Embedding with Gemini",
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(parts):
            print(f"Using cached API embeddings: {cache_path}")
            return cached
        print(f"Ignoring cache with mismatched shape: {cached.shape}")

    api_key = api_key or os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(f"{api_key_env} is not set.")

    model_name, endpoint_model = _gemini_model_names(model)
    endpoint = f"{GEMINI_API_ROOT}/models/{endpoint_model}:batchEmbedContents"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    batch_size = min(batch_size, 6)

    chunks = []
    for start in tqdm(range(0, len(parts), batch_size), desc=description):
        batch_parts = parts[start : start + batch_size]
        requests_payload = [
            {
                "model": model_name,
                "content": {"parts": [part]},
            }
            for part in batch_parts
        ]
        response = requests.post(endpoint, headers=headers, json={"requests": requests_payload}, timeout=timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"Gemini API failed ({response.status_code}): {response.text[:800]}")
        chunks.append(_parse_gemini_embeddings(response.json()))
        if sleep_seconds:
            time.sleep(sleep_seconds)

    embeddings = np.vstack(chunks)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"Wrote API embedding cache: {cache_path}")
    return embeddings


def gemini_embed_images(
    image_paths: list[Path],
    cache_path: Path,
    model: str = "gemini-embedding-2",
    api_key: str | None = None,
    batch_size: int = 6,
    image_max_side: int = 512,
    jpeg_quality: int = 86,
    sleep_seconds: float = 0.2,
    timeout: int = 180,
) -> np.ndarray:
    parts = [
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": _image_base64(path, image_max_side, jpeg_quality),
            }
        }
        for path in image_paths
    ]
    return gemini_embed_parts(
        parts,
        cache_path=cache_path,
        model=model,
        api_key=api_key,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
        timeout=timeout,
        description=f"Embedding images with {model}",
    )


def gemini_embed_audio(
    audio_paths: list[Path],
    cache_path: Path,
    model: str = "gemini-embedding-2",
    api_key: str | None = None,
    batch_size: int = 6,
    sleep_seconds: float = 0.2,
    timeout: int = 180,
) -> np.ndarray:
    parts = []
    for path in audio_paths:
        suffix = path.suffix.lower()
        mime_type = "audio/mpeg" if suffix in {".mp3", ".mpeg"} else "audio/wav"
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": _file_base64(path),
                }
            }
        )
    return gemini_embed_parts(
        parts,
        cache_path=cache_path,
        model=model,
        api_key=api_key,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
        timeout=timeout,
        description=f"Embedding audio with {model}",
    )


def gemini_embed_texts(
    texts: list[str],
    cache_path: Path,
    model: str = "gemini-embedding-2",
    api_key: str | None = None,
    batch_size: int = 6,
    sleep_seconds: float = 0.2,
    timeout: int = 180,
) -> np.ndarray:
    parts = [{"text": text} for text in texts]
    return gemini_embed_parts(
        parts,
        cache_path=cache_path,
        model=model,
        api_key=api_key,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
        timeout=timeout,
        description=f"Embedding texts with {model}",
    )
