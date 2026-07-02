#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

from api_embedding_clients import gemini_embed_audio
from viewer_export_utils import build_layouts, l2_normalize, update_manifest, write_lightweight_json


OPENNEURO_DATASET = "ds007808"
OPENNEURO_VERSION = "1.0.0"
OPENNEURO_S3_ROOT = f"https://s3.amazonaws.com/openneuro.org/{OPENNEURO_DATASET}"
OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"
SPEAKER_COLORS = ["#38bdf8", "#f97316", "#a3e635", "#e879f9", "#facc15", "#14b8a6"]


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    sample_width: int
    data_offset: int
    data_size: int

    @property
    def block_align(self) -> int:
        return self.channels * self.sample_width


def openneuro_url(filename: str) -> str:
    return f"{OPENNEURO_S3_ROOT}/{quote(filename)}"


def fetch_file_list(cache_path: Path, refresh: bool) -> list[dict]:
    if cache_path.exists() and not refresh:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data["data"]["snapshot"]["files"]

    query = """
    query JapanEEGFiles($datasetId: ID!, $tag: String!) {
      snapshot(datasetId: $datasetId, tag: $tag) {
        files(recursive: true) { filename size }
      }
    }
    """
    response = requests.post(
        OPENNEURO_GRAPHQL,
        json={"query": query, "variables": {"datasetId": OPENNEURO_DATASET, "tag": OPENNEURO_VERSION}},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload["data"]["snapshot"]["files"]


def parse_run_key(filename: str) -> dict | None:
    pattern = re.compile(
        r"(?P<subject>sub-[^/]+)/(?P<session>ses-[^/]+)/(?P<folder>beh|eeg)/"
        r"(?P=subject)_(?P=session)_task-(?P<task>[^_]+)_acq-(?P<acq>[^_]+)_run-(?P<run>\d+)"
    )
    match = pattern.search(filename)
    if not match:
        return None
    return match.groupdict()


def find_candidate_runs(files: list[dict], task: str, max_source_wav_mb: float) -> list[dict]:
    by_key: dict[tuple[str, str, str, str, str], dict] = {}
    for row in files:
        filename = row["filename"]
        parsed = parse_run_key(filename)
        if not parsed or parsed["task"] != task:
            continue
        key = (parsed["subject"], parsed["session"], parsed["task"], parsed["acq"], parsed["run"])
        entry = by_key.setdefault(key, {**parsed, "size": 0})
        if filename.endswith("_recording-audio_beh.wav") or filename.endswith("_recording-vocal_beh.wav"):
            entry["wav"] = filename
            entry["size"] = int(row.get("size") or 0)
        elif filename.endswith("_events.tsv"):
            entry["events"] = filename
            entry["eventsSize"] = int(row.get("size") or 0)
        elif filename.endswith("_eeg.edf"):
            entry["edf"] = filename
            entry["edfSize"] = int(row.get("size") or 0)

    max_bytes = max_source_wav_mb * 1024 * 1024
    candidates = [
        entry
        for entry in by_key.values()
        if entry.get("wav") and entry.get("events") and 0 < entry.get("size", 0) <= max_bytes
    ]
    return sorted(candidates, key=lambda item: (item.get("size", 0), item["subject"], item["session"], item["run"]))


def download_text(filename: str) -> str:
    response = requests.get(openneuro_url(filename), timeout=60)
    response.raise_for_status()
    return response.content.decode("utf-8")


def download_openneuro_file(filename: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    response = requests.get(openneuro_url(filename), stream=True, timeout=180)
    response.raise_for_status()
    total = int(response.headers.get("Content-Length") or 0)
    with tmp.open("wb") as handle:
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}") as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))
    shutil.move(tmp, dest)
    return dest


def parse_events(text: str, trial_type: str, min_duration: float, max_duration: float) -> list[dict]:
    rows = []
    for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
        if row.get("trial_type") != trial_type:
            continue
        wav_onset = row.get("wav_onset", "n/a")
        if wav_onset == "n/a":
            continue
        try:
            duration = float(row["duration"])
            start = float(wav_onset)
        except (KeyError, TypeError, ValueError):
            continue
        if duration < min_duration:
            continue
        rows.append(
            {
                "wav_onset": start,
                "duration": min(duration, max_duration),
                "trial_type": row.get("trial_type", trial_type),
                "transcript": row.get("value", ""),
                "eeg_onset": float(row["onset"]) if row.get("onset") not in {None, "n/a", ""} else None,
                "sample": int(float(row["sample"])) if row.get("sample") not in {None, "n/a", ""} else None,
            }
        )
    return rows


def ranged_get(url: str, start: int, end: int) -> bytes:
    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    response.raise_for_status()
    if response.status_code == 206:
        return response.content
    content = response.content
    expected = end - start + 1
    if len(content) >= end + 1:
        return content[start : end + 1]
    if len(content) == expected:
        return content
    raise RuntimeError(f"Unexpected range response {response.status_code}: got {len(content)} bytes, expected {expected}")


def parse_wav_info(header: bytes) -> WavInfo:
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("Expected a RIFF/WAVE file.")

    offset = 12
    channels = sample_rate = sample_width = data_offset = data_size = None
    while offset + 8 <= len(header):
        chunk_id = header[offset : offset + 4]
        chunk_size = int.from_bytes(header[offset + 4 : offset + 8], "little")
        chunk_data = offset + 8
        if chunk_id == b"fmt ":
            audio_format = int.from_bytes(header[chunk_data : chunk_data + 2], "little")
            channels = int.from_bytes(header[chunk_data + 2 : chunk_data + 4], "little")
            sample_rate = int.from_bytes(header[chunk_data + 4 : chunk_data + 8], "little")
            bits_per_sample = int.from_bytes(header[chunk_data + 14 : chunk_data + 16], "little")
            sample_width = bits_per_sample // 8
            if audio_format != 1:
                raise ValueError(f"Only PCM WAV is supported, got format {audio_format}.")
        elif chunk_id == b"data":
            data_offset = chunk_data
            data_size = chunk_size
            break
        offset = chunk_data + chunk_size + (chunk_size % 2)

    if None in {channels, sample_rate, sample_width, data_offset, data_size}:
        raise ValueError("Could not parse WAV fmt/data chunks from header.")
    return WavInfo(
        sample_rate=int(sample_rate),
        channels=int(channels),
        sample_width=int(sample_width),
        data_offset=int(data_offset),
        data_size=int(data_size),
    )


def fetch_wav_info(filename: str) -> WavInfo:
    header = ranged_get(openneuro_url(filename), 0, 65535)
    return parse_wav_info(header)


def cut_remote_wav_clip(filename: str, info: WavInfo, dest: Path, start_seconds: float, duration_seconds: float) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    start_frame = max(0, int(round(start_seconds * info.sample_rate)))
    frame_count = max(1, int(round(duration_seconds * info.sample_rate)))
    max_frames = info.data_size // info.block_align
    frame_count = min(frame_count, max(0, max_frames - start_frame))
    byte_start = info.data_offset + start_frame * info.block_align
    byte_end = byte_start + frame_count * info.block_align - 1
    frames = ranged_get(openneuro_url(filename), byte_start, byte_end)
    with wave.open(str(dest), "wb") as writer:
        writer.setnchannels(info.channels)
        writer.setsampwidth(info.sample_width)
        writer.setframerate(info.sample_rate)
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


def extract_acoustic_features(audio_paths: list[Path]) -> np.ndarray:
    features = []
    bands = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    for path in tqdm(audio_paths, desc="Extracting acoustic features"):
        sample_rate, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))

        duration = len(data) / sample_rate
        rms = float(np.sqrt(np.mean(np.square(data)) + 1e-12))
        peak = float(np.max(np.abs(data)) if len(data) else 0)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(data)))) if len(data) > 1 else 0)
        freqs, power = signal.welch(data, fs=sample_rate, nperseg=min(2048, max(256, len(data))))
        power = np.maximum(power, 1e-12)
        total_power = float(np.sum(power))
        centroid = float(np.sum(freqs * power) / total_power)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
        cumulative = np.cumsum(power)
        rolloff = float(freqs[np.searchsorted(cumulative, cumulative[-1] * 0.85)])
        band_powers = []
        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            band_powers.append(float(np.log10(np.sum(power[mask]) + 1e-12)))
        features.append([duration, rms, peak, zcr, centroid, bandwidth, rolloff, *band_powers])
    return np.asarray(features, dtype=np.float32)


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def speaker_name(index: int) -> str:
    if 0 <= index < 26:
        return f"Speaker {chr(ord('A') + index)}"
    return f"Speaker {index + 1}"


def assign_speaker_groups(items: list[dict]) -> dict[str, str]:
    sources = []
    for item in items:
        source = item.get("source") or "unknown-source"
        if source not in sources:
            sources.append(source)

    palette = {}
    for item in items:
        source = item.get("source") or "unknown-source"
        source_index = sources.index(source)
        speaker = speaker_name(source_index)
        color = SPEAKER_COLORS[source_index % len(SPEAKER_COLORS)]
        item["speaker"] = speaker
        item["speakerSource"] = source
        item["color"] = color
        palette[speaker] = color
    return palette


def build_clips(args: argparse.Namespace) -> tuple[list[dict], list[Path]]:
    file_cache = args.cache_dir / "openneuro_ds007808_files.json"
    files = fetch_file_list(file_cache, refresh=args.refresh_file_list)
    candidates = find_candidate_runs(files, args.task, args.max_source_wav_mb)
    if not candidates:
        raise FileNotFoundError(f"No JapanEEG runs found for task={args.task!r}.")

    clips_dir = args.output_dir / "audio"
    items = []
    audio_paths = []
    for run in tqdm(candidates, desc="Scanning JapanEEG runs"):
        if len(items) >= args.max_clips:
            break
        events = parse_events(
            download_text(run["events"]),
            trial_type=args.trial_type,
            min_duration=args.min_duration,
            max_duration=args.clip_seconds,
        )
        if not events:
            continue
        wav_info = fetch_wav_info(run["wav"])
        run_label = f"{run['subject']}_{run['session']}_{run['task']}_run-{run['run']}"
        for event_number, event in enumerate(events, start=1):
            if len(items) >= args.max_clips:
                break
            clip_index = len(items) + 1
            clip_stem = f"{run_label}_event-{event_number:04d}"
            clip_path = clips_dir / f"{clip_stem}.wav"
            duration = min(event["duration"], args.clip_seconds)
            cut_remote_wav_clip(run["wav"], wav_info, clip_path, event["wav_onset"], duration)
            items.append(
                {
                    "index": clip_index,
                    "id": f"japaneeg_{clip_index:04d}",
                    "label": f"{run['subject']} {run['session']} run {run['run']} event {event_number}",
                    "audio": relpath(clip_path, args.output_dir),
                    "mediaType": "audio",
                    "stimulusType": "Japanese speech listening",
                    "subject": run["subject"],
                    "session": run["session"],
                    "run": run["run"],
                    "task": run["task"],
                    "trialType": event["trial_type"],
                    "transcript": event["transcript"],
                    "start": round(event["wav_onset"], 3),
                    "end": round(event["wav_onset"] + duration, 3),
                    "duration": round(duration, 3),
                    "eegOnset": round(event["eeg_onset"], 3) if event["eeg_onset"] is not None else None,
                    "eegSample": event["sample"],
                    "source": Path(run["wav"]).name,
                    "sourceUrl": openneuro_url(run["wav"]),
                    "eegFile": run.get("edf"),
                    "eegSourceUrl": openneuro_url(run["edf"]) if run.get("edf") else None,
                }
            )
            audio_paths.append(clip_path)

    if len(items) < 4:
        raise ValueError(f"Need at least 4 clips, found {len(items)}")
    return items, audio_paths


def extract_eeg_features(items: list[dict], cache_dir: Path, cache_name: str) -> np.ndarray:
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(items):
            print(f"Using cached EEG features: {cache_path}")
            return cached
        print(f"Ignoring EEG cache with mismatched shape: {cached.shape}")

    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required for --include-eeg. Install it or omit --include-eeg.") from exc

    by_edf: dict[str, list[tuple[int, dict]]] = {}
    for index, item in enumerate(items):
        eeg_file = item.get("eegFile")
        if eeg_file and item.get("eegOnset") is not None:
            by_edf.setdefault(eeg_file, []).append((index, item))
    if not by_edf:
        raise ValueError("No EEG files were available for the selected clips.")

    features: list[np.ndarray | None] = [None] * len(items)
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 80)]
    edf_dir = cache_dir / "edf"
    for edf_file, indexed_items in by_edf.items():
        edf_path = download_openneuro_file(edf_file, edf_dir / Path(edf_file).name)
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        picks = list(range(min(128, len(raw.ch_names))))
        for index, item in tqdm(indexed_items, desc=f"EEG features {Path(edf_file).name}"):
            start = max(0, int(round(float(item["eegOnset"]) * sfreq)))
            duration = max(0.25, float(item.get("duration") or 1.0))
            stop = min(raw.n_times, start + max(2, int(round(duration * sfreq))))
            data = raw.get_data(picks=picks, start=start, stop=stop)
            if data.size == 0:
                data = np.zeros((len(picks), 2), dtype=np.float32)
            data = data - np.mean(data, axis=1, keepdims=True)
            nperseg = min(512, max(64, data.shape[1]))
            freqs, power = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=1)
            power = np.maximum(power, 1e-20)
            channel_features = []
            for low, high in bands:
                mask = (freqs >= low) & (freqs < high)
                band_power = np.log10(np.mean(power[:, mask], axis=1) + 1e-20)
                channel_features.append(band_power)
            features[index] = np.concatenate(channel_features).astype(np.float32)
        raw.close()

    filled = [feature if feature is not None else np.zeros(len(bands) * 128, dtype=np.float32) for feature in features]
    matrix = np.vstack(filled)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    print(f"Wrote EEG feature cache: {cache_path}")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight JapanEEG audio clip viewer data.")
    parser.add_argument("--output-dir", type=Path, default=Path("viewer/public/viewer-data/japaneeg-audio"))
    parser.add_argument("--manifest", type=Path, default=Path("viewer/public/viewer-data/datasets.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/japaneeg_audio"))
    parser.add_argument("--task", default="listening", choices=["listening", "listeningcovert", "speechopen"])
    parser.add_argument("--trial-type", default="listening")
    parser.add_argument("--max-clips", type=int, default=100)
    parser.add_argument("--clip-seconds", type=float, default=6.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-source-wav-mb", type=float, default=120.0)
    parser.add_argument("--include-eeg", action="store_true", help="Download matching EDFs and add event-aligned EEG bandpower layouts.")
    parser.add_argument("--embedding-source", choices=["auto", "acoustic", "gemini", "both"], default="auto")
    parser.add_argument("--model", default="gemini-embedding-2")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--pca-precomponents", type=int, default=50)
    parser.add_argument("--refresh-file-list", action="store_true")
    parser.add_argument("--no-manifest-update", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    items, audio_paths = build_clips(args)
    speaker_palette = assign_speaker_groups(items)

    acoustic_features = extract_acoustic_features(audio_paths)
    embeddings = build_layouts(
        acoustic_features,
        key_prefix="acoustic",
        label_prefix="Acoustic",
        random_state=args.random_state,
        pca_precomponents=args.pca_precomponents,
    )
    default_embedding = "acoustic_ppca"
    signal_parts = ["acoustic audio features"]

    if args.include_eeg:
        eeg_features = extract_eeg_features(items, args.cache_dir, f"japaneeg_eeg_{args.task}_{len(items)}.npy")
        eeg_embeddings = build_layouts(
            eeg_features,
            key_prefix="eeg",
            label_prefix="EEG",
            random_state=args.random_state,
            pca_precomponents=args.pca_precomponents,
        )
        embeddings = {**eeg_embeddings, **embeddings}
        default_embedding = "eeg_ppca"
        signal_parts.insert(0, "event-aligned EEG bandpower")

    wants_gemini = args.embedding_source in {"gemini", "both"} or (
        args.embedding_source == "auto" and bool(os.getenv("GEMINI_API_KEY"))
    )
    if wants_gemini:
        model_slug = args.model.replace("/", "_").replace(":", "_")
        gemini_cache = args.cache_dir / f"japaneeg_audio_{model_slug}_{len(audio_paths)}.npy"
        gemini_features = gemini_embed_audio(
            audio_paths,
            cache_path=gemini_cache,
            model=args.model,
            batch_size=args.batch_size,
        )
        model_label = "Gemini" if args.model.startswith("gemini") else args.model
        embeddings.update(
            build_layouts(
                l2_normalize(gemini_features),
                key_prefix="gemini",
                label_prefix=model_label,
                random_state=args.random_state,
                pca_precomponents=args.pca_precomponents,
            )
        )
        default_embedding = "gemini_ppca"
        signal_parts.insert(0, f"{args.model} audio latent")
    elif args.embedding_source == "gemini":
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    payload = {
        "schema": "neuro-latent-viewer-dataset-v1",
        "id": "japaneeg-audio",
        "label": "JapanEEG speech audio clips",
        "source": "JapanEEG / OpenNeuro ds007808 v1.0.0 listening events",
        "itemCount": len(items),
        "audioCount": len(items),
        "itemLabel": "audio clips",
        "signalLabel": " + ".join(signal_parts),
        "defaultEmbedding": default_embedding,
        "display": {
            "mediaSprites": False,
            "selectedMedia": "point",
            "selectionHalo": "circle",
            "pointSize": 0.13,
            "pointOpacity": 0.9,
            "pointPickRadius": 0.14,
        },
        "colorBy": {
            "field": "speaker",
            "label": "Speaker",
            "palette": speaker_palette,
        },
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
                    "id": "japaneeg-audio",
                    "label": "JapanEEG speech audio clips",
                    "shortLabel": "JapanEEG audio",
                    "href": "japaneeg-audio/index.json",
                    "stimulus": f"{len(items):,} Japanese speech clips",
                    "signal": payload["signalLabel"],
                    "subject": "Human EEG/EMG/audio speech dataset",
                    "description": "Lightweight 3D point layouts for JapanEEG listening clips, colored by source speaker/run, with click-to-play audio and transcripts.",
                }
            ],
        )
        print(f"Updated manifest: {args.manifest}")


if __name__ == "__main__":
    main()
