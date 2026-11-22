"""Config loading, paths, seeding and small IO helpers."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config/config.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # yaml reads ".inf" as float inf already
    return cfg


def path(cfg: dict, key: str, *parts: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p.joinpath(*parts) if parts else p


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def write_parquet(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def read_parquet(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p)


def write_json(obj, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
