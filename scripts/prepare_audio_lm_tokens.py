#!/usr/bin/env python3
"""Prepare byte-token LM corpora from downloaded music/audio datasets."""

from __future__ import annotations

import argparse
import io
import json
import random
import tarfile
import tempfile
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pretty_midi


PIANO_MIN = 21
PIANO_MAX = 108
FILE_SEP = 0
TIME_SHIFT_START = 1
TIME_SHIFT_BINS = 32
NOTE_ON_START = TIME_SHIFT_START + TIME_SHIFT_BINS
NOTE_OFF_START = NOTE_ON_START + (PIANO_MAX - PIANO_MIN + 1)


def _write_tmp(tmpdir: Path, name: str, payload: bytes) -> Path:
    out = tmpdir / Path(name).name
    out.write_bytes(payload)
    return out


def split_items(items: list[str], eval_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    eval_count = max(1, int(round(len(shuffled) * eval_fraction))) if len(shuffled) > 1 else 0
    return shuffled[eval_count:], shuffled[:eval_count]


def write_bytes(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(int(v) & 0xFF for v in values))


def encode_midi_file(path: Path, *, time_quantum: float = 0.02) -> list[int]:
    pm = pretty_midi.PrettyMIDI(str(path))
    events: list[tuple[float, int, int]] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if PIANO_MIN <= note.pitch <= PIANO_MAX:
                events.append((float(note.start), 1, int(note.pitch)))
                events.append((float(note.end), 0, int(note.pitch)))
    events.sort(key=lambda item: (item[0], item[1], item[2]))

    tokens = [FILE_SEP]
    cursor = 0
    for time_sec, is_on, pitch in events:
        tick = max(0, int(round(time_sec / time_quantum)))
        delta = max(0, tick - cursor)
        while delta > 0:
            shift = min(delta, TIME_SHIFT_BINS)
            tokens.append(TIME_SHIFT_START + shift - 1)
            delta -= shift
        cursor = tick
        pitch_idx = pitch - PIANO_MIN
        if is_on:
            tokens.append(NOTE_ON_START + pitch_idx)
        else:
            tokens.append(NOTE_OFF_START + pitch_idx)
    return tokens


def prepare_maestro(download_dir: Path, out_dir: Path, max_files: int, eval_fraction: float, seed: int) -> dict:
    archive = download_dir / "maestro-v3.0.0-midi.zip"
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".mid", ".midi"))]
        names = names[:max_files]
        train_names, eval_names = split_items(names, eval_fraction, seed)

        summary = {"files": len(names), "train_files": len(train_names), "eval_files": len(eval_names)}
        for split, split_names in (("train", train_names), ("eval", eval_names)):
            values: list[int] = []
            skipped = 0
            with tempfile.TemporaryDirectory(prefix=f"maestro_{split}_") as tmp:
                tmpdir = Path(tmp)
                for name in split_names:
                    try:
                        midi_path = _write_tmp(tmpdir, name, zf.read(name))
                        values.extend(encode_midi_file(midi_path))
                    except Exception:
                        skipped += 1
            write_bytes(out_dir / "maestro" / f"{split}.bin", values)
            summary[f"{split}_tokens"] = len(values)
            summary[f"{split}_skipped"] = skipped
    return summary


def audio_to_tokens(path: Path, *, sr: int, duration: float, n_mels: int, hop_length: int) -> list[int]:
    y, actual_sr = librosa.load(path, sr=sr, mono=True, duration=duration)
    if y.size == 0:
        return [FILE_SEP]
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=actual_sr,
        n_fft=1024,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    quantized = np.clip(np.round((log_mel + 80.0) * (255.0 / 80.0)), 0, 255).astype(np.uint8)
    # Time-major flattening keeps short-range local continuity within frames and
    # exposes frame-to-frame structure at a stable stride of n_mels.
    return [FILE_SEP] + quantized.T.reshape(-1).tolist()


def prepare_fma(
    download_dir: Path,
    out_dir: Path,
    max_tracks: int,
    eval_fraction: float,
    seed: int,
    *,
    sr: int,
    duration: float,
    n_mels: int,
    hop_length: int,
) -> dict:
    archive = download_dir / "fma_small.zip"
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".mp3")]
        names = names[:max_tracks]
        train_names, eval_names = split_items(names, eval_fraction, seed)
        summary = {"files": len(names), "train_files": len(train_names), "eval_files": len(eval_names)}

        for split, split_names in (("train", train_names), ("eval", eval_names)):
            values: list[int] = []
            skipped = 0
            with tempfile.TemporaryDirectory(prefix=f"fma_{split}_") as tmp:
                tmpdir = Path(tmp)
                for name in split_names:
                    try:
                        audio_path = _write_tmp(tmpdir, name, zf.read(name))
                        values.extend(
                            audio_to_tokens(
                                audio_path,
                                sr=sr,
                                duration=duration,
                                n_mels=n_mels,
                                hop_length=hop_length,
                            )
                        )
                    except Exception:
                        skipped += 1
            write_bytes(out_dir / "fma_mel" / f"{split}.bin", values)
            summary[f"{split}_tokens"] = len(values)
            summary[f"{split}_skipped"] = skipped
    return summary


def prepare_musicnet(
    download_dir: Path,
    out_dir: Path,
    max_wavs: int,
    eval_fraction: float,
    seed: int,
    *,
    sr: int,
    duration: float,
    n_mels: int,
    hop_length: int,
) -> dict:
    archive = download_dir / "musicnet.tar.gz"
    wav_payloads: list[tuple[str, bytes]] = []
    with tarfile.open(archive, mode="r:gz") as tf:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".wav")):
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            wav_payloads.append((member.name, fh.read()))
            if len(wav_payloads) >= max_wavs:
                break

    names = [name for name, _ in wav_payloads]
    train_names, eval_names = split_items(names, eval_fraction, seed)
    payload_by_name = dict(wav_payloads)
    summary = {"files": len(names), "train_files": len(train_names), "eval_files": len(eval_names)}

    for split, split_names in (("train", train_names), ("eval", eval_names)):
        values: list[int] = []
        skipped = 0
        with tempfile.TemporaryDirectory(prefix=f"musicnet_{split}_") as tmp:
            tmpdir = Path(tmp)
            for name in split_names:
                try:
                    audio_path = _write_tmp(tmpdir, name, payload_by_name[name])
                    values.extend(
                        audio_to_tokens(
                            audio_path,
                            sr=sr,
                            duration=duration,
                            n_mels=n_mels,
                            hop_length=hop_length,
                        )
                    )
                except Exception:
                    skipped += 1
        write_bytes(out_dir / "musicnet_mel" / f"{split}.bin", values)
        summary[f"{split}_tokens"] = len(values)
        summary[f"{split}_skipped"] = skipped
    return summary


def write_mixed(out_dir: Path, datasets: list[str]) -> dict:
    summary: dict[str, int] = {}
    for split in ("train", "eval"):
        values = bytearray()
        for name in datasets:
            path = out_dir / name / f"{split}.bin"
            if path.exists():
                values.extend(path.read_bytes())
        write_bytes(out_dir / "music_audio_mixed" / f"{split}.bin", list(values))
        summary[f"{split}_tokens"] = len(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", type=Path, default=Path("data/downloads"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/audio_lm_smoke"))
    parser.add_argument("--max-maestro-files", type=int, default=96)
    parser.add_argument("--max-fma-tracks", type=int, default=64)
    parser.add_argument("--max-musicnet-wavs", type=int, default=16)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--n-mels", type=int, default=32)
    parser.add_argument("--hop-length", type=int, default=512)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "maestro": prepare_maestro(
            args.download_dir,
            args.out_dir,
            args.max_maestro_files,
            args.eval_fraction,
            args.seed,
        ),
        "fma_mel": prepare_fma(
            args.download_dir,
            args.out_dir,
            args.max_fma_tracks,
            args.eval_fraction,
            args.seed,
            sr=args.sr,
            duration=args.duration,
            n_mels=args.n_mels,
            hop_length=args.hop_length,
        ),
        "musicnet_mel": prepare_musicnet(
            args.download_dir,
            args.out_dir,
            args.max_musicnet_wavs,
            args.eval_fraction,
            args.seed,
            sr=args.sr,
            duration=args.duration,
            n_mels=args.n_mels,
            hop_length=args.hop_length,
        ),
    }
    summary["music_audio_mixed"] = write_mixed(args.out_dir, ["maestro", "fma_mel", "musicnet_mel"])
    summary["audio_params"] = {
        "sr": args.sr,
        "duration": args.duration,
        "n_mels": args.n_mels,
        "hop_length": args.hop_length,
    }
    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
