#!/usr/bin/env python3
"""Decode-level smoke test for the downloaded music/audio datasets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
import tempfile
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _write_temp_bytes(tmpdir: Path, name: str, payload: bytes) -> Path:
    out = tmpdir / Path(name).name
    out.write_bytes(payload)
    return out


def _audio_stats(path: Path, sr: int = 16000, duration: float = 5.0) -> dict:
    info = sf.info(path)
    y, actual_sr = librosa.load(path, sr=sr, mono=True, duration=duration)
    return {
        "path": str(path),
        "format": info.format,
        "subtype": info.subtype,
        "channels": info.channels,
        "native_sample_rate": info.samplerate,
        "native_duration_sec": float(info.duration),
        "decoded_sample_rate": actual_sr,
        "decoded_samples": int(y.shape[0]),
        "decoded_duration_sec": float(y.shape[0] / actual_sr),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "peak_abs": float(np.max(np.abs(y))) if y.size else 0.0,
    }


def probe_maestro(download_dir: Path, tmpdir: Path) -> dict:
    archive = download_dir / "maestro-v3.0.0-midi.zip"
    with zipfile.ZipFile(archive) as zf:
        midi_names = [n for n in zf.namelist() if n.lower().endswith((".mid", ".midi"))]
        midi_path = _write_temp_bytes(tmpdir, midi_names[0], zf.read(midi_names[0]))
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    note_count = sum(len(inst.notes) for inst in pm.instruments)
    return {
        "archive": str(archive),
        "sample_midi": midi_names[0],
        "duration_sec": float(pm.get_end_time()),
        "instruments": len(pm.instruments),
        "note_count": note_count,
        "first_instruments": [
            {
                "program": inst.program,
                "is_drum": bool(inst.is_drum),
                "name": inst.name,
                "notes": len(inst.notes),
            }
            for inst in pm.instruments[:5]
        ],
    }


def probe_fma(download_dir: Path, tmpdir: Path) -> dict:
    archive = download_dir / "fma_small.zip"
    with zipfile.ZipFile(archive) as zf:
        mp3_names = [n for n in zf.namelist() if n.lower().endswith(".mp3")]
        mp3_path = _write_temp_bytes(tmpdir, mp3_names[0], zf.read(mp3_names[0]))
    stats = _audio_stats(mp3_path)
    stats.update({"archive": str(archive), "sample_mp3": mp3_names[0]})
    return stats


def probe_musicnet(download_dir: Path, tmpdir: Path) -> dict:
    archive = download_dir / "musicnet.tar.gz"
    first_label = None
    first_rows: list[dict[str, str]] = []
    first_wav = None
    wav_path = None

    with tarfile.open(archive, mode="r:gz") as tf:
        for member in tf:
            if first_label is None and member.isfile() and member.name.endswith(".csv"):
                first_label = member.name
                fh = tf.extractfile(member)
                if fh is not None:
                    text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
                    reader = csv.DictReader(text)
                    for _, row in zip(range(5), reader):
                        first_rows.append(dict(row))
            if first_wav is None and member.isfile() and member.name.endswith(".wav"):
                first_wav = member.name
                fh = tf.extractfile(member)
                if fh is not None:
                    wav_path = _write_temp_bytes(tmpdir, member.name, fh.read())
            if first_label is not None and first_wav is not None:
                break

    if wav_path is None:
        raise RuntimeError("MusicNet WAV sample not found")
    stats = _audio_stats(wav_path)
    stats.update(
        {
            "archive": str(archive),
            "sample_wav": first_wav,
            "sample_label_csv": first_label,
            "sample_label_rows": first_rows,
        }
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", type=Path, default=Path("data/downloads"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase2/audio_dataset_smoke"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="audio_decode_smoke_") as tmp:
        tmpdir = Path(tmp)
        summary = {
            "maestro": probe_maestro(args.download_dir, tmpdir),
            "fma_small": probe_fma(args.download_dir, tmpdir),
            "musicnet": probe_musicnet(args.download_dir, tmpdir),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "decode_summary.json"
    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
