"""Stage 2: factors (METHODOLOGY section 4) for the default setting and the sensitivity variants.

usage: python scripts/02_build_features.py [--variants default pass70 nonsub_fail]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.pipeline import build_features  # noqa: E402
from src.utils.io import load_config, seed_everything  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["default", "pass70", "nonsub_fail"])
    args = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    for v in args.variants:
        print(build_features(cfg, v))
