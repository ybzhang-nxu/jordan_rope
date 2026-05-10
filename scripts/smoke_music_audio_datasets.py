#!/usr/bin/env python3
"""Lightweight smoke checks for downloaded music/audio archives.

The checks intentionally avoid full extraction. They verify that each archive
can be opened, that expected file types are present, and that at least one
sample payload is readable.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
import zipfile
from collections import Counter
from pathlib import Path


def _suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".tar.gz"):
        return ".tar.gz"
    return Path(lower).suffix or "<none>"


def _read_u16(data: bytes, pos: int) -> int:
    return int.from_bytes(data[pos : pos + 2], "big")


def _read_u32(data: bytes, pos: int) -> int:
    return int.from_bytes(data[pos : pos + 4], "big")


def _read_le_u16(data: bytes, pos: int) -> int:
    return int.from_bytes(data[pos : pos + 2], "little")


def _read_le_u32(data: bytes, pos: int) -> int:
    return int.from_bytes(data[pos : pos + 4], "little")


def _read_varlen(data: bytes, pos: int, limit: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if pos >= limit:
            raise ValueError("unterminated MIDI varlen")
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value, pos
    return value, pos


def _parse_midi_note_probe(data: bytes) -> dict:
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI header")
    header_len = _read_u32(data, 4)
    fmt = _read_u16(data, 8)
    ntracks = _read_u16(data, 10)
    division = _read_u16(data, 12)
    pos = 8 + header_len
    note_on = 0
    note_off = 0
    max_tick = 0
    notes: Counter[int] = Counter()
    first_note_on: list[dict[str, int]] = []

    for track_idx in range(ntracks):
      if pos + 8 > len(data) or data[pos : pos + 4] != b"MTrk":
          break
      track_len = _read_u32(data, pos + 4)
      pos += 8
      end = min(pos + track_len, len(data))
      tick = 0
      running_status: int | None = None

      while pos < end:
          delta, pos = _read_varlen(data, pos, end)
          tick += delta
          max_tick = max(max_tick, tick)
          if pos >= end:
              break

          byte = data[pos]
          if byte >= 0x80:
              status = byte
              pos += 1
              if status < 0xF0:
                  running_status = status
          else:
              if running_status is None:
                  break
              status = running_status

          if status == 0xFF:
              if pos >= end:
                  break
              pos += 1
              length, pos = _read_varlen(data, pos, end)
              pos = min(pos + length, end)
              continue
          if status in (0xF0, 0xF7):
              length, pos = _read_varlen(data, pos, end)
              pos = min(pos + length, end)
              continue

          event_type = status & 0xF0
          channel = status & 0x0F
          data_len = 1 if event_type in (0xC0, 0xD0) else 2
          payload = data[pos : min(pos + data_len, end)]
          pos += len(payload)
          if len(payload) < data_len:
              break

          if event_type == 0x90 and payload[1] > 0:
              note = payload[0]
              velocity = payload[1]
              note_on += 1
              notes[note] += 1
              if len(first_note_on) < 12:
                  first_note_on.append(
                      {"track": track_idx, "tick": tick, "channel": channel, "note": note, "velocity": velocity}
                  )
          elif event_type == 0x80 or (event_type == 0x90 and payload[1] == 0):
              note_off += 1

      pos = end

    return {
        "format": fmt,
        "tracks": ntracks,
        "division": division,
        "note_on_events": note_on,
        "note_off_events": note_off,
        "unique_notes": len(notes),
        "max_tick": max_tick,
        "top_notes": notes.most_common(8),
        "first_note_on": first_note_on,
    }


def _mp3_probe(data: bytes) -> dict:
    id3_size = 0
    if data[:3] == b"ID3" and len(data) >= 10:
        id3_size = 10 + sum((data[6 + idx] & 0x7F) << shift for idx, shift in enumerate((21, 14, 7, 0)))
    first_frame = None
    for pos in range(id3_size, max(id3_size, len(data) - 1)):
        if data[pos] == 0xFF and data[pos + 1] & 0xE0 == 0xE0:
            first_frame = pos
            break
    return {
        "id3_present": data[:3] == b"ID3",
        "id3_size": id3_size,
        "first_frame_offset": first_frame,
        "first_frame_header_hex": data[first_frame : first_frame + 4].hex() if first_frame is not None else "",
    }


def _parse_wav_header(header: bytes) -> dict:
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE header")
    pos = 12
    fmt: dict | None = None
    data_size: int | None = None
    while pos + 8 <= len(header):
        chunk_id = header[pos : pos + 4]
        chunk_size = _read_le_u32(header, pos + 4)
        payload = pos + 8
        if chunk_id == b"fmt " and payload + 16 <= len(header):
            format_tag = _read_le_u16(header, payload)
            channels = _read_le_u16(header, payload + 2)
            sample_rate = _read_le_u32(header, payload + 4)
            byte_rate = _read_le_u32(header, payload + 8)
            block_align = _read_le_u16(header, payload + 12)
            bits_per_sample = _read_le_u16(header, payload + 14)
            format_name = {
                1: "PCM",
                3: "IEEE_FLOAT",
                65534: "EXTENSIBLE",
            }.get(format_tag, f"UNKNOWN_{format_tag}")
            fmt = {
                "format_tag": format_tag,
                "format_name": format_name,
                "channels": channels,
                "sample_rate": sample_rate,
                "byte_rate": byte_rate,
                "block_align": block_align,
                "bits_per_sample": bits_per_sample,
            }
        elif chunk_id == b"data":
            data_size = chunk_size
            if fmt is not None:
                break
        pos = payload + chunk_size + (chunk_size % 2)

    if fmt is None:
        raise ValueError("missing WAV fmt chunk")
    if data_size is not None and fmt["byte_rate"]:
        fmt["data_bytes"] = data_size
        fmt["duration_sec"] = data_size / fmt["byte_rate"]
    return fmt


def smoke_maestro(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        midi = [n for n in names if n.lower().endswith((".mid", ".midi"))]
        csvs = [n for n in names if n.lower().endswith(".csv")]
        first_rows: list[dict[str, str]] = []
        if csvs:
            with zf.open(csvs[0]) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                reader = csv.DictReader(text)
                for _, row in zip(range(3), reader):
                    first_rows.append(dict(row))
        sample_magic = ""
        if midi:
            with zf.open(midi[0]) as raw:
                midi_bytes = raw.read()
                sample_magic = midi_bytes[:4].hex()
                note_probe = _parse_midi_note_probe(midi_bytes)
    return {
        "archive": str(path),
        "entries": len(names),
        "midi_files": len(midi),
        "csv_files": len(csvs),
        "first_midi": midi[0] if midi else None,
        "first_midi_magic_hex": sample_magic,
        "first_midi_note_probe": note_probe if midi else None,
        "first_csv": csvs[0] if csvs else None,
        "first_csv_rows": first_rows,
    }


def smoke_fma_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        csvs = [n for n in names if n.lower().endswith(".csv")]
        previews: dict[str, list[str]] = {}
        for name in csvs[:3]:
            with zf.open(name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                previews[name] = [line.rstrip("\n") for _, line in zip(range(3), text)]
    return {
        "archive": str(path),
        "entries": len(names),
        "csv_files": len(csvs),
        "csv_preview": previews,
    }


def smoke_fma_small(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        audio = [n for n in names if n.lower().endswith(".mp3")]
        sample_magic = ""
        if audio:
            with zf.open(audio[0]) as raw:
                sample = raw.read(65536)
                sample_magic = sample[:16].hex()
                mp3_probe = _mp3_probe(sample)
    return {
        "archive": str(path),
        "entries": len(names),
        "mp3_files": len(audio),
        "first_mp3": audio[0] if audio else None,
        "first_mp3_magic_hex": sample_magic,
        "first_mp3_probe": mp3_probe if audio else None,
    }


def smoke_musicnet(path: Path, max_members: int) -> dict:
    suffixes: Counter[str] = Counter()
    first_members: list[str] = []
    first_regular_file: str | None = None
    first_wav: str | None = None
    first_label_csv: str | None = None
    sample_magic = ""
    wav_probe: dict | None = None
    label_rows: list[dict[str, str]] = []

    with tarfile.open(path, mode="r:gz") as tf:
        for idx, member in enumerate(tf):
            suffixes[_suffix(member.name)] += 1
            if len(first_members) < 20:
                first_members.append(member.name)
            if first_regular_file is None and member.isfile():
                first_regular_file = member.name
                fh = tf.extractfile(member)
                if fh is not None:
                    sample_magic = fh.read(16).hex()
            if first_label_csv is None and member.isfile() and member.name.endswith(".csv"):
                first_label_csv = member.name
                fh = tf.extractfile(member)
                if fh is not None:
                    text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
                    reader = csv.DictReader(text)
                    for _, row in zip(range(5), reader):
                        label_rows.append(dict(row))
            if first_wav is None and member.isfile() and member.name.endswith(".wav"):
                first_wav = member.name
                fh = tf.extractfile(member)
                if fh is not None:
                    header = fh.read(65536)
                    wav_probe = _parse_wav_header(header)
                if first_label_csv is not None:
                    break
            if idx + 1 >= max_members:
                break

    return {
        "archive": str(path),
        "scanned_members": sum(suffixes.values()),
        "suffix_counts_in_scan": dict(sorted(suffixes.items())),
        "first_members": first_members,
        "first_regular_file": first_regular_file,
        "first_regular_magic_hex": sample_magic,
        "first_label_csv": first_label_csv,
        "first_label_rows": label_rows,
        "first_wav": first_wav,
        "first_wav_probe": wav_probe,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", type=Path, default=Path("data/downloads"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase2/audio_dataset_smoke"))
    parser.add_argument("--musicnet-max-members", type=int, default=200)
    args = parser.parse_args()

    download_dir = args.download_dir
    summary = {
        "maestro_midi": smoke_maestro(download_dir / "maestro-v3.0.0-midi.zip"),
        "fma_metadata": smoke_fma_metadata(download_dir / "fma_metadata.zip"),
        "fma_small": smoke_fma_small(download_dir / "fma_small.zip"),
        "musicnet": smoke_musicnet(download_dir / "musicnet.tar.gz", args.musicnet_max_members),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "smoke_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
