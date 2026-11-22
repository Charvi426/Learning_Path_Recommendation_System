"""Presentation-based temporal split (METHODOLOGY 3.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_split(df: pd.DataFrame, cfg_split: dict, presentations_by_module: dict[str, set]) -> pd.Series:
    """train: 2013B/2013J, val: 2014B, test: 2014J. Modules without 2014B: 2014J students are
    split 50/50 by id_student (even ids -> val, odd -> test; deterministic)."""
    tr, va, te = set(cfg_split["train"]), set(cfg_split["val"]), set(cfg_split["test"])
    out = np.full(len(df), "none", dtype=object)
    mod = df["code_module"].to_numpy()
    pres = df["code_presentation"].to_numpy()
    sid = df["id_student"].to_numpy()
    for i_m in np.unique(mod):
        has_val = bool(presentations_by_module[i_m] & va)
        sel = mod == i_m
        out[sel & np.isin(pres, list(tr))] = "train"
        if has_val:
            out[sel & np.isin(pres, list(va))] = "val"
            out[sel & np.isin(pres, list(te))] = "test"
        elif not cfg_split.get("fallback", True):
            # v2 holdout modules: no validation split; the latest presentation is the test set
            out[sel & np.isin(pres, list(te))] = "test"
        else:
            t = sel & np.isin(pres, list(te))
            out[t & (sid % 2 == 0)] = "val"
            out[t & (sid % 2 == 1)] = "test"
    return pd.Series(out, index=df.index, name="split")
