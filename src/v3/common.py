"""Shared helpers for v3 scripts: build the selected variant, inject the v3 knowledge state into a v2 RecDataV2,
evaluate recommender configurations."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..evaluation.ranking_metrics import student_mean  # noqa: F401
from ..evaluation.stats import bootstrap_mean_ci, wilcoxon_holm
from ..recommend import v2 as V2
from ..recommend.engine import evaluate_paths
from ..utils.io import ROOT
from .data import practice
from .memory import Series
from .pipeline import Variant, knowledge_at


def selected():
    return json.load(open(ROOT / "results/v3/selected_variant.json", encoding="utf-8"))


def build_variants(sel, cs, data, splits, v1fit=None):
    """(variant without interest, twin with interest), both unfitted. Series keys: submitted pairs in `splits`."""
    inter = data["inter"]
    hp = sel["frozen_hyperparameters"]
    series = None
    if sel["kind"] != "hl_v2":
        keys = inter[(inter["is_submitted"] == 1) & inter["split"].isin(splits)][["sid", "topic"]]
        series = Series(practice(cs, data, sel["attribution"]), inter, keys)
    mk = lambda interest: Variant(sel["kind"], series, sel["params"], interest=interest,  # noqa: E731
                                  l2=hp["l2_interest"] if interest else hp["l2"], S=hp["S"], C=hp["das3h_C"])
    return mk(False), mk(True)


def inject(rd, kv: pd.DataFrame, name: str, kcol: str = "K"):
    """Put K/M/R/p of a knowledge_at frame (decision queries) into rd.models[name] aligned with rd rows."""
    pos = {q: i for i, q in enumerate(rd.qid)}
    r = kv["qid"].map(pos)
    ok = r.notna().to_numpy()
    r = r[ok].astype(int).to_numpy()
    c = kv["j"].to_numpy()[ok] - 1
    out = {}
    for key, col, fill in [("K", kcol, 1.0), ("M", "Mhat", 1.0), ("R", "R", np.nan), ("p", "p", 1.0)]:
        a = np.full((rd.N, rd.T), fill)
        a[r, c] = kv[col].to_numpy()[ok]
        out[key] = a
    rd.models[name] = out
    return rd


def v3_paths(rd, final_v3, cids, rcfg):
    paths = {}
    for cid in cids:
        t = final_v3["ablations"][cid]
        W = np.array([[t["weights"][k] for k in V2.TERMS6]])
        P, aux = V2.paths_v2(rd, t["config"], W, t["filters"], rcfg["K"], rcfg["max_revision"], rcfg["horizon"])
        paths[f"v3:{cid}"] = (P[0], aux, {**t["config"], "label": f"v3 {cid} {t['config']['label']}"}, W[0])
    return paths


def v2_paths(rd, final_v2, cids, rcfg):
    paths = {}
    for cid in cids:
        t = final_v2["ablations"][cid]
        W = np.array([[t["weights"][k] for k in V2.TERMS6]])
        P, aux = V2.paths_v2(rd, t["config"], W, t["filters"], rcfg["K"], rcfg["max_revision"], rcfg["horizon"])
        paths[f"v2:{cid}"] = (P[0], aux, {**t["config"], "label": f"v2 {cid} {t['config']['label']}"}, W[0])
    return paths


def per_student(rd, paths):
    out = {}
    for name, (P, aux, conf, W) in paths.items():
        (_, ps), = evaluate_paths(rd, P[None], rels=("A", "B", "AB"))
        out[name] = ps
    return out


def ranking_rows(ps: dict, labels: dict, nb, seed, extra=None):
    rows = []
    for name, d in ps.items():
        for met, v in d.items():
            est, lo, hi = bootstrap_mean_ci(v, nb, seed)
            b, rel = met.split("[")
            rows.append({"config": name, "label": labels.get(name, name), "relevance": rel.rstrip("]"), "metric": b,
                         "value": est, "ci_lo": lo, "ci_hi": hi, "n_students": int(np.isfinite(v).sum()),
                         **(extra or {})})
    return rows


COMPARISONS = [("v3:A5", "v2:A5", "v3 A5 vs v2 A5"), ("v3:A5", "v3:A3", "v3 A5 vs v3 A3"),
               ("v3:A9", "v3:A8", "v3 A9 vs v3 A8")]


def wilcoxon_table(ps):
    return pd.concat([wilcoxon_holm({n: (ps[a][m], ps[b][m]) for a, b, n in COMPARISONS}).assign(metric=m)
                      for m in ["NDCG@5[AB]", "NDCG@5[A]", "NDCG@5[B]"]], ignore_index=True)


def knowledge_for_decisions(var, data, split):
    q = data["queries"]
    return knowledge_at(var, data, q[(q["kind"] == "decision") & (q["split"] == split)], split)
