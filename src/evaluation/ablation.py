"""Ablation configurations A0-A7 (METHODOLOGY 7.5) and their validation tuning (6.3).

Tuning per configuration: thresholds (tau_m, tau_q, tau_r) and p* with equal weights over the active terms
-> weight grid -> thresholds once more. Objective: NDCG@5 (combined relevance) on validation, ties broken by
Recall@5.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..recommend.engine import RecData, objective
from ..recommend.scoring import weight_grid

ALL = ["G", "Q", "F", "D", "I"]


def ablation_configs(has_data_graph: bool) -> dict:
    return {
        "A0": {"kind": "random", "label": "Random"},
        "A1": {"kind": "popularity", "label": "Popularity"},
        "A2": {"kind": "score", "label": "Mastery only", "model": "none", "active": ["G"], "graph": None,
               "use_q": False},
        "A3": {"kind": "score", "label": "Mastery + prerequisites", "model": "none", "active": ["G", "Q", "D"],
               "graph": "chain"},
        "A4": {"kind": "score", "label": "+ fixed forgetting", "model": "fixed", "active": ["G", "Q", "D", "F"],
               "graph": "chain"},
        "A5": {"kind": "score", "label": "+ learned forgetting (proposed)", "model": "learned",
               "active": ["G", "Q", "D", "F"], "graph": "chain"},
        "A6": {"kind": "score", "label": "+ interest", "model": "learnedI", "active": ALL, "graph": "chain"},
        "A7": {"kind": "score", "label": "A6 + data-derived prerequisites",
               "model": "learnedI_data" if has_data_graph else "learnedI", "active": ALL,
               "graph": "data"},
    }


def _better(a, b):
    """(ndcg, recall) lexicographic, NaN-safe."""
    return (a[0] > b[0] + 1e-12) or (abs(a[0] - b[0]) <= 1e-12 and a[1] > b[1] + 1e-12)


def tune_config(rd: RecData, conf: dict, rcfg: dict, log=None) -> dict:
    grid = rcfg["threshold_grid"]
    pgrid = rcfg["p_star_grid"] if "D" in conf["active"] else [rcfg["start"]["p_star"]]
    kw = {"L": rcfg["K"], "max_rev": rcfg["max_revision"], "horizon": rcfg["horizon"]}
    tq_grid = grid if conf.get("use_q", True) else [rcfg["start"]["tau_q"]]
    W_all = weight_grid(conf["active"], rcfg["weight_step"], rcfg["eps_max"])
    w_eq = np.zeros((1, 5))
    idx = [ALL.index(a) for a in conf["active"]]
    w_eq[0, idx] = 1.0 / len(idx)
    history = []

    def thr_search(w):
        best, best_thr = (-np.inf, -np.inf), None
        for tm, tq, tr, ps in itertools.product(grid, tq_grid, grid, pgrid):
            thr = {"tau_m": tm, "tau_q": tq, "tau_r": tr, "p_star": ps}
            nd, rc = objective(rd, conf, w, thr, **kw)
            v = (nd[0], rc[0])
            history.append({"stage": "thresholds", **thr, "w": w[0].round(2).tolist(), "NDCG@5": v[0], "R@5": v[1]})
            if _better(v, best):
                best, best_thr = v, thr
        return best, best_thr

    # stage 1: thresholds with equal weights
    _, thr = thr_search(w_eq)
    # stage 2: weights
    nd, rc = objective(rd, conf, W_all, thr, **kw)
    best_i = 0
    for i in range(len(W_all)):
        if _better((nd[i], rc[i]), (nd[best_i], rc[best_i])):
            best_i = i
    w_best = W_all[best_i:best_i + 1]
    for i in range(len(W_all)):
        history.append({"stage": "weights", **thr, "w": W_all[i].round(2).tolist(), "NDCG@5": nd[i], "R@5": rc[i]})
    # stage 3: thresholds again
    best, thr = thr_search(w_best)
    if log:
        log.info("tuned %s: thr=%s w=%s NDCG@5=%.4f R@5=%.4f (%d weight vectors)", conf["label"], thr,
                 w_best[0].round(2).tolist(), best[0], best[1], len(W_all))
    return {"thresholds": thr, "weights": dict(zip(ALL, w_best[0].round(2).tolist())),
            "val_NDCG@5": float(best[0]), "val_R@5": float(best[1]), "n_weight_vectors": int(len(W_all)),
            "history": history}


def weights_array(tuned: dict) -> np.ndarray:
    return np.array([[tuned["weights"][t] for t in ALL]])
