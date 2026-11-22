"""v3 Step 5 (post-hoc): recommender on the v3 knowledge state, with v2's relative filters and six-term score.
Filters are frozen from v2; only the weights are re-tuned on validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..recommend import v2 as V2
from ..recommend.v2 import TERMS6
from .pipeline import knowledge_at


def add_v3_columns(ks: pd.DataFrame, var, data, split, name):
    """Adds M_/K_/R_/p_{name} for decision points (kind == decision) to a v1-style knowledge-state frame."""
    q = data["queries"]
    qsel = q[(q["kind"] == "decision") & (q["split"] == split)]
    kv = knowledge_at(var, data, qsel, split)[["qid", "j", "K", "Mhat", "R", "p"]]
    kv = kv.rename(columns={"K": f"K_{name}", "Mhat": f"M_{name}", "R": f"R_{name}", "p": f"p_{name}"})
    out = ks.merge(kv, on=["qid", "j"], how="left")
    # topics that do not exist in the presentation keep v1 values (masked later by RecDataV2.exists)
    for c in ["K", "M", "R", "p"]:
        base = f"{c}_none" if c != "R" else None
        if base:
            out[f"{c}_{name}"] = out[f"{c}_{name}"].fillna(out[base])
    return out


def tune_weights(rd, conf, filters, rcfg):
    W = V2.weight_grid6(conf["active"], rcfg["weight_step"], rcfg["eps_max"])
    kw = {"L": rcfg["K"], "max_rev": rcfg["max_revision"], "horizon": rcfg["horizon"]}
    nd, rc, _ = V2.objective_v2(rd, conf, W, filters, **kw)
    bi = 0
    for i in range(len(W)):
        if (nd[i] > nd[bi] + 1e-12) or (abs(nd[i] - nd[bi]) <= 1e-12 and rc[i] > rc[bi] + 1e-12):
            bi = i
    return {"weights": dict(zip(TERMS6, W[bi].round(2).tolist())), "val_NDCG@5": float(nd[bi]),
            "val_R@5": float(rc[bi]), "n_weight_vectors": int(len(W))}
