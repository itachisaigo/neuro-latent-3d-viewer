#!/usr/bin/env python3
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

from api_embedding_clients import gemini_embed_audio
from viewer_export_utils import build_layouts, l2_normalize, update_manifest, write_lightweight_json


def find_audio_files(root: Path) -> list[Path]:
    candidates = []
    for pattern in ("stimuli/**/*.wav", "stimuli/**/*.mp3", "**/*_audio.wav", "**/*.wav", "**/*.mp3"):
        candidates.extend(root.glob(pattern))
    seen = set()
    unique = []
    for path in sorted(candidates):
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def cut_wav_clip(source: Path, dest: Path, start_seconds: float, duration_seconds: float) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        sample_rate = reader.getframerate()
        start_frame = int(start_seconds * sample_rate)
        frame_count = int(duration_seconds * sample_rate)
        reader.setpos(min(start_frame, reader.getnframes()))
        frames = reader.readframes(frame_count)
    with wave.open(str(dest), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)


def make_spectrogram(source: Path, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    sample_rate, data = wavfile.read(source)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(float)
    frequencies, times, spec = signal.spectrogram(data, fs=sample_rate, nperseg=512, noverlap=384)
    spec_db = 10 * np.log10(spec + 1e-10)

    fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=96)
    ax.pcolormesh(times, frequencies, spec_db, shading="auto", cmap="magma")
    ax.set_axis_off()
    ax.set_ylim(0, min(8000, sample_rate / 2))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(dest, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_audio_clips(audio_files: list[Path], output_dir: Path, clip_seconds: float, max_clips: int) -> list[dict]:
    clips_dir = output_dir / "audio"
    specs_dir = output_dir / "spectrograms"
    clips = []
    for audio_path in audio_files:
        if len(clips) >= max_clips:
            break
        if audio_path.suffix.lower() != ".wav":
            continue
        with wave.open(str(audio_path), "rb") as reader:
            sample_rate = reader.getframerate()
            duration = reader.getnframes() / sample_rate
        starts = np.arange(0, max(duration - clip_seconds, 0.1), clip_seconds)
        for start in starts:
            if len(clips) >= max_clips:
                break
            index = len(clips) + 1
            stem = f"{audio_path.stem}_{int(start):05d}_{int(start + clip_seconds):05d}"
            clip_path = clips_dir / f"{stem}.wav"
            spectrogram_path = specs_dir / f"{stem}.jpg"
            cut_wav_clip(audio_path, clip_path, float(start), clip_seconds)
            make_spectrogram(clip_path, spectrogram_path)
            clips.append(
                {
                    "index": index,
                    "source": audio_path,
                    "clip": clip_path,
                    "spectrogram": spectrogram_path,
                    "start": float(start),
                    "end": float(start + clip_seconds),
                }
            )
    return clips


def load_feature_matrix(path: Path | None, expected_rows: int) -> np.ndarray | None:
    if not path:
        return None
    if path.suffix == ".npy":
        features = np.load(path)
    else:
        features = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if features.shape[0] != expected_rows:
        raise ValueError(f"Feature rows {features.shape[0]} do not match clips {expected_rows}")
    return np.asarray(features, dtype=np.float32)


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight OpenNeuro Narratives audio clip viewer data.")
    parser.add_argument("--bids-root", type=Path, required=True, help="OpenNeuro ds002345/Narratives BIDS root.")
    parser.add_argument("--output-dir", type=Path, default=Path("viewer/public/viewer-data/narratives-audio-gemini"))
    parser.add_argument("--manifest", type=Path, default=Path("viewer/public/viewer-data/datasets.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/api_embeddings"))
    parser.add_argument("--clip-seconds", type=float, default=6.0)
    parser.add_argument("--max-clips", type=int, default=240)
    parser.add_argument("--model", default="gemini-embedding-2")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--fmri-features", type=Path, help="Optional clip-by-feature matrix from fMRI responses.")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--pca-precomponents", type=int, default=50)
    parser.add_argument("--no-manifest-update", action="store_true")
    args = parser.parse_args()

    audio_files = find_audio_files(args.bids_root)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found under {args.bids_root}")

    clips = make_audio_clips(audio_files, args.output_dir, args.clip_seconds, args.max_clips)
    if len(clips) < 4:
        raise ValueError(f"Need at least 4 clips, found {len(clips)}")

    clip_paths = [clip["clip"] for clip in clips]
    model_slug = args.model.replace("/", "_").replace(":", "_")
    model_label = "Gemini" if args.model.startswith("gemini") else args.model
    audio_cache = args.cache_dir / f"narratives_audio_{model_slug}_{len(clips)}.npy"
    audio_features = gemini_embed_audio(
        clip_paths,
        cache_path=audio_cache,
        model=args.model,
        batch_size=args.batch_size,
    )

    embeddings = build_layouts(
        l2_normalize(audio_features),
        key_prefix="audio",
        label_prefix=model_label,
        random_state=args.random_state,
        pca_precomponents=args.pca_precomponents,
    )

    fmri_features = load_feature_matrix(args.fmri_features, len(clips))
    if fmri_features is not None:
        embeddings.update(
            build_layouts(
                fmri_features,
                key_prefix="fmri",
                label_prefix="fMRI",
                random_state=args.random_state,
                pca_precomponents=args.pca_precomponents,
            )
        )
        default_embedding = "fmri_ppca"
        signal_label = "fMRI response + Gemini audio latent"
    else:
        default_embedding = "audio_ppca"
        signal_label = "Gemini audio latent"

    items = []
    for clip in tqdm(clips, desc="Preparing items"):
        source_name = clip["source"].stem
        label = f"{source_name} {clip['start']:.0f}-{clip['end']:.0f}s"
        items.append(
            {
                "index": clip["index"],
                "id": f"narratives_{clip['index']:04d}",
                "label": label,
                "audio": relpath(clip["clip"], args.output_dir),
                "thumb": relpath(clip["spectrogram"], args.output_dir),
                "mediaType": "audio",
                "stimulusType": "spoken story",
                "source": clip["source"].name,
                "start": round(clip["start"], 3),
                "end": round(clip["end"], 3),
            }
        )

    payload = {
        "schema": "neuro-latent-viewer-dataset-v1",
        "id": "narratives-audio-gemini",
        "label": f"Narratives audio clips + {args.model}",
        "source": "OpenNeuro ds002345/Narratives audio clips plus Gemini API embeddings",
        "itemCount": len(items),
        "audioCount": len(items),
        "itemLabel": "audio clips",
        "signalLabel": signal_label,
        "defaultEmbedding": default_embedding,
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
                    "id": "narratives-audio-gemini",
                    "label": f"Narratives audio clips + {args.model}",
                    "shortLabel": "Narratives audio",
                    "href": "narratives-audio-gemini/index.json",
                    "stimulus": f"{len(items):,} spoken-story clips",
                    "signal": signal_label,
                    "subject": "Human fMRI story-listening dataset",
                    "description": "Lightweight 3D layouts for Narratives audio clips, with click-to-play audio previews.",
                }
            ],
        )
        print(f"Updated manifest: {args.manifest}")


if __name__ == "__main__":
    main()
