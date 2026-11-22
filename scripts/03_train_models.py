"""Stage 3: knowledge models (METHODOLOGY section 5).

Identification check on Δ, baselines, PFA, the four forgetting variants, DAS3H-window benchmark,
bootstrap coefficient intervals, knowledge_state.parquet (validation + test decision points and exam dates).
Model hyper-parameters are tuned on validation only.

usage: python scripts/03_train_models.py [--variant default] [--n-boot 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from src.models.knowledge_state import knowledge_state  # noqa: E402
from src.models.train import load_variant, train_all  # noqa: E402
from src.utils.io import get_logger, load_config, path, seed_everything, write_parquet  # noqa: E402

log = get_logger("03_train")


def run(cfg, variant="default", n_boot=None, do_bootstrap=True):
    data = load_variant(cfg, variant)
    graphs = ("chain", "data") if not data["edges"]["source"].eq("data").sum() == 0 else ("chain",)
    fitted, summary = train_all(cfg, variant, n_boot=n_boot, graphs=graphs, do_bootstrap=do_bootstrap,
                                tune_windows=(variant == "default"))
    forget = {k: v for k, v in fitted.items() if k.split("_")[0] in {"none", "fixed", "learned", "learnedI"}}
    parts = []
    for sp in ["val", "test"]:
        parts.append(knowledge_state(data["queries"], data["state"][sp], forget, data["graphs"], data["tindex"],
                                     data["diff"]))
    ks = pd.concat(parts, ignore_index=True)
    write_parquet(ks, data["dir"] / "knowledge_state.parquet")
    log.info("knowledge_state rows: %d; graphs: %s", len(ks), graphs)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="default")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--no-bootstrap", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    print(run(cfg, a.variant, a.n_boot, not a.no_bootstrap))
