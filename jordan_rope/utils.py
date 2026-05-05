from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for this experiment. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc


def choose_device(name: str = "auto"):
    require_torch()
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    require_torch()
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def copy_source_config(src: str | Path, out_dir: str | Path) -> None:
    src = Path(src)
    out_dir = ensure_dir(out_dir)
    if src.exists():
        shutil.copy2(src, out_dir / src.name)


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, full))
        else:
            out[full] = value
    return out


def env_snapshot() -> dict[str, Any]:
    snapshot = {
        "cwd": os.getcwd(),
        "python": shutil.which("python3"),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snapshot["cuda_device"] = torch.cuda.get_device_name(0)
    except ModuleNotFoundError:
        snapshot["torch"] = None
    return snapshot
