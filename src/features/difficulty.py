"""Topic difficulty (METHODOLOGY 4.6): 1 - pass rate among submitters, training split only."""
from __future__ import annotations

import pandas as pd


def topic_difficulty(train_inter: pd.DataFrame) -> pd.DataFrame:
    d = train_inter[train_inter["is_submitted"] == 1]
    g = d.groupby(["code_module", "topic"]).agg(n_submitted=("y", "size"), pass_rate=("y", "mean")).reset_index()
    g["diff"] = 1 - g["pass_rate"]
    return g


def cohort_mean_y(train_inter: pd.DataFrame) -> dict:
    d = train_inter[train_inter["is_submitted"] == 1]
    return d.groupby("code_module")["y"].mean().to_dict()
