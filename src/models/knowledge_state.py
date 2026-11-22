"""Knowledge state M, R, K for every (decision query, topic j) (METHODOLOGY 5.5).

assessed j:      M = y~_{s,j} (normalised score), R = model retention at t, K = M * R
not assessed j:  M = K = predicted P(y_{s,j}=1) if attempted now, R undefined (NaN)
One set of columns per model variant (suffix _none, _fixed, _learned, _learnedI[, _data]).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import build_design


def knowledge_state(queries: pd.DataFrame, state: pd.DataFrame, models: dict, graphs: dict, tindex: dict,
                    difficulty: pd.DataFrame, kinds=("decision", "exam")) -> pd.DataFrame:
    q = queries[queries["kind"].isin(kinds) & queries["qid"].isin(state["qid"].unique())]
    st = state[state["qid"].isin(q["qid"])].copy()
    ks = st.merge(q[["qid", "sid", "k", "t", "kind", "split", "code_presentation"]], on="qid", how="left")
    ks = ks.rename(columns={"k": "k_now"})
    ks = ks[["qid", "sid", "j", "k_now", "t", "kind", "split", "code_module", "code_presentation", "assessed",
             "M", "y_j", "delta", "c_inf", "d_inf", "I", "clicks_raw_inf", "registered", "window_open",
             "ability", "c_7", "c_30"]].rename(columns={"M": "M_obs"})
    ks = ks.merge(difficulty[["code_module", "topic", "diff"]].rename(columns={"topic": "j"}),
                  on=["code_module", "j"], how="left")
    targets = ks[["qid", "sid", "j", "t", "code_module"]].rename(columns={"j": "k"})
    assessed = ks["assessed"].to_numpy(bool)
    for name, m in models.items():
        graph = "data" if name.endswith("_data") else "chain"
        D = build_design(targets, state, graphs[graph], tindex)
        p = m.predict_proba(D)
        R, h = m.retention_state(ks)
        M = np.where(assessed, ks["M_obs"].to_numpy(), p)
        ks[f"p_{name}"] = p
        ks[f"R_{name}"] = np.where(assessed, R, np.nan)
        ks[f"h_{name}"] = np.where(assessed, h, np.nan)
        ks[f"M_{name}"] = M
        ks[f"K_{name}"] = np.where(assessed, M * R, p)
    return ks
