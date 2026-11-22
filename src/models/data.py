"""Design matrices shared by every model.

A *target row* is (qid, target topic k): predict P(y_{s,k} = 1) from the state at the query time t.
Training rows use t = t_due_k; knowledge-state rows use a decision time and every topic j as target.
Prerequisite information is kept in long form (one row per (target row, prerequisite j)) so the
forgetting model can learn a half-life per pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features.build import wname

OWN_COLS = ["ability", "prev_attempts", "c_7", "c_14", "c_30", "c_inf", "d_inf", "I", "dt_last"]
PAIR_COLS = ["assessed", "M", "delta", "c_inf", "d_inf", "I", "y_j", "t_submit_j"]


@dataclass
class Design:
    rows: pd.DataFrame            # qid, sid, k, t, module, topic_idx, y, y_score + own features
    pairs: pd.DataFrame           # row (index into rows), j, w, M, delta, c_inf, d_inf, I, assessed, y_j ...
    topic_index: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.rows)


def topic_index_map(modules_topics: pd.DataFrame) -> dict:
    mt = modules_topics.drop_duplicates().sort_values(["code_module", "topic"])
    return {(m, int(k)): i for i, (m, k) in enumerate(zip(mt["code_module"], mt["topic"]))}


def build_design(targets: pd.DataFrame, state: pd.DataFrame, prereq: dict, tindex: dict,
                 windows=(7, 30, np.inf)) -> Design:
    """targets: qid, sid, k, t, code_module (+ y, y_score optional)."""
    tg = targets.reset_index(drop=True).copy()
    tg["row"] = np.arange(len(tg))
    st = state.set_index(["qid", "j"])
    own = st.loc[list(zip(tg["qid"], tg["k"])), OWN_COLS].reset_index(drop=True)
    rows = pd.concat([tg, own], axis=1)
    rows["topic_idx"] = [tindex[(m, int(k))] for m, k in zip(rows["code_module"], rows["k"])]
    # prerequisite pairs
    pr = []
    mods = rows["code_module"].to_numpy()
    ks = rows["k"].to_numpy()
    for m in np.unique(mods):
        for k, plist in prereq.get(m, {}).items():
            sel = np.flatnonzero((mods == m) & (ks == k))
            if len(sel) == 0 or not plist:
                continue
            for j in plist:
                pr.append(pd.DataFrame({"row": sel, "j": j, "w": 1.0 / len(plist)}))
    if pr:
        pairs = pd.concat(pr, ignore_index=True)
        pairs["qid"] = rows["qid"].to_numpy()[pairs["row"].to_numpy()]
        pv = st.loc[list(zip(pairs["qid"], pairs["j"])), PAIR_COLS].reset_index(drop=True)
        pairs = pd.concat([pairs, pv], axis=1)
    else:
        pairs = pd.DataFrame(columns=["row", "j", "w", "qid"] + PAIR_COLS)
    pairs["assessed"] = pairs["assessed"].astype(bool)
    pairs["M"] = np.where(pairs["assessed"], pairs["M"], 0.0)      # missed prerequisite: no knowledge
    pairs["delta"] = np.where(pairs["assessed"], pairs["delta"], 0.0)
    pairs = pairs.sort_values("row").reset_index(drop=True)
    # succ / fail counts of prerequisites in windows before t (assessment event = submission day)
    t_row = rows["t"].to_numpy()[pairs["row"].to_numpy()]
    for w in windows:
        inw = pairs["assessed"].to_numpy() & ((t_row - pairs["t_submit_j"].to_numpy()) <= w)
        yj = pairs["y_j"].to_numpy()
        rows[f"succ_{wname(w)}"] = np.bincount(pairs["row"], weights=(inw & (yj == 1)).astype(float), minlength=len(rows))
        rows[f"fail_{wname(w)}"] = np.bincount(pairs["row"], weights=(inw & (yj == 0)).astype(float), minlength=len(rows))
    rows["Mbar"] = np.bincount(pairs["row"], weights=pairs["w"] * pairs["M"], minlength=len(rows))
    rows["has_prereq"] = np.bincount(pairs["row"], minlength=len(rows)) > 0
    return Design(rows=rows, pairs=pairs, topic_index=tindex)


def training_targets(queries: pd.DataFrame, inter: pd.DataFrame, split: str) -> pd.DataFrame:
    q = queries[(queries["kind"] == "due") & (queries["split"] == split)]
    t = q.merge(inter[["sid", "topic", "y", "y_score", "is_submitted"]].rename(columns={"topic": "k"}),
                on=["sid", "k"], how="left")
    t = t[t["y"].notna()]
    return t[["qid", "sid", "k", "t", "code_module", "code_presentation", "y", "y_score"]]


class Standardizer:
    def __init__(self, cols):
        self.cols = cols

    def fit(self, df):
        self.mu = df[self.cols].mean()
        self.sd = df[self.cols].std().replace(0, 1).fillna(1)
        return self

    def transform(self, df):
        return (df[self.cols] - self.mu) / self.sd
