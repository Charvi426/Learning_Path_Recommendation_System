"""Ground-truth relevance per decision point (METHODOLOGY 7.2).

Window = [t_decision, t_due_k] (what happened until the next assessment).
Rel-A (students who passed assessment k): topic j relevant if the student made >= 10 clicks on j's resources
       in the window.
Rel-B (students who failed assessment k, score < pass mark): j = k or j in the chain prerequisites of k up to
       two levels back (k-1, k-2).
Graded relevance = 1[Rel-A] + 1[Rel-B]. As defined, Rel-A and Rel-B apply to disjoint students, so the
grade never reaches 2 (see DECISIONS.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.engagement import ClickCube


def window_clicks(vd: pd.DataFrame, n_sid: int, n_topics: int, sid, a, b) -> np.ndarray:
    """clicks on every topic in days [a, b] -> (n, n_topics)."""
    cube = ClickCube(vd, n_sid, n_topics)
    return cube.clicks(np.asarray(sid, np.int64), np.asarray(b, float) + 1, np.inf,
                       start=np.asarray(a, float)[:, None])


def relevance_table(dp: pd.DataFrame, inter: pd.DataFrame, vd: pd.DataFrame, n_sid: int, n_topics: int,
                    min_clicks: float = 10) -> pd.DataFrame:
    """dp: decision points with qid, sid, k (= k_now), t. Returns one row per decision point with
    outcome, relevance arrays relA_j, relB_j (j = 1..T) and window clicks."""
    d = dp.merge(inter[["sid", "topic", "y", "t_due", "is_submitted"]].rename(columns={"topic": "k"}),
                 on=["sid", "k"], how="left")
    clicks = window_clicks(vd, n_sid, n_topics, d["sid"].to_numpy(), d["t"].to_numpy(), d["t_due"].to_numpy())
    passed = (d["y"] == 1).to_numpy()
    failed = (d["y"] == 0).to_numpy()
    T = np.arange(1, n_topics + 1)[None, :]
    k = d["k"].to_numpy()[:, None]
    relA = (clicks >= min_clicks) & passed[:, None]
    relB = ((T == k) | (T == k - 1) | (T == k - 2)) & failed[:, None]
    for j in range(n_topics):
        d[f"relA_{j+1}"] = relA[:, j].astype(np.int8)
        d[f"relB_{j+1}"] = relB[:, j].astype(np.int8)
        d[f"wclicks_{j+1}"] = clicks[:, j]
    d["group"] = np.where(passed, "A", np.where(failed, "B", "none"))
    return d


def rel_matrix(rt: pd.DataFrame, n_topics: int, which: str = "AB") -> np.ndarray:
    A = rt[[f"relA_{j}" for j in range(1, n_topics + 1)]].to_numpy(float)
    B = rt[[f"relB_{j}" for j in range(1, n_topics + 1)]].to_numpy(float)
    if which == "A":
        return A
    if which == "B":
        return B
    return A + B
