"""Behavioural interest proxy I(s,k) (METHODOLOGY 4.4).

raw = vol + l1 * early + l2 * post (each log1p of clicks before t)
share = raw_k / sum_j raw_j over the topics the student has opened so far (J_s)
z = (share - mu) / sd, with mu, sd from the TRAINING split per (module, topic, stage), where
stage = number of topic windows already open at t (share depends mechanically on how many
topics are open, so the cohort reference is taken at the same stage).
I = sigmoid(z - mean_{j in J_s} z); I = 0.5 if the student has no voluntary/early/post activity yet,
and for topics outside J_s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def raw_signal(state: pd.DataFrame, lam_early: float = 1.0, lam_post: float = 1.0,
               use_vol: bool = True) -> np.ndarray:
    v = state["vol"].to_numpy() if use_vol else 0.0
    return v + lam_early * state["early"].to_numpy() + lam_post * state["post"].to_numpy()


def add_share(state: pd.DataFrame, raw: np.ndarray) -> pd.DataFrame:
    st = state
    st["raw"] = raw
    opened = st["opened"].to_numpy()
    r_open = np.where(opened, raw, 0.0)
    tot = pd.Series(r_open).groupby(st["qid"].to_numpy()).transform("sum").to_numpy()
    st["raw_total"] = tot
    st["share"] = np.where(opened & (tot > 0), r_open / np.where(tot > 0, tot, 1), np.nan)
    return st


def fit_zstats(train_state: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """mean / sd of share per (module, topic j, stage), training split only."""
    d = train_state[train_state["share"].notna()]
    g = d.groupby(["code_module", "j", "stage"])["share"].agg(["mean", "std", "size"]).reset_index()
    g2 = d.groupby(["code_module", "j"])["share"].agg(["mean", "std"]).reset_index()
    g = g.merge(g2, on=["code_module", "j"], suffixes=("", "_mj"))
    small = (g["size"] < min_n) | g["std"].isna() | (g["std"] < 1e-6)
    g.loc[small, "mean"] = g.loc[small, "mean_mj"]
    g.loc[small, "std"] = g.loc[small, "std_mj"]
    g["std"] = g["std"].fillna(1.0).clip(lower=1e-3)
    return g[["code_module", "j", "stage", "mean", "std", "size"]]


def add_interest(state: pd.DataFrame, zstats: pd.DataFrame, col: str = "I") -> pd.DataFrame:
    st = state
    z = st[["code_module", "j", "stage"]].merge(zstats, on=["code_module", "j", "stage"], how="left")
    # stages not seen in training fall back to the module-topic mean over stages
    fb = zstats.groupby(["code_module", "j"])[["mean", "std"]].mean().reset_index()
    zf = st[["code_module", "j"]].merge(fb, on=["code_module", "j"], how="left")
    mu = z["mean"].fillna(zf["mean"]).to_numpy()
    sd = z["std"].fillna(zf["std"]).fillna(1.0).to_numpy()
    zz = (st["share"].to_numpy() - mu) / sd
    valid = st["share"].notna().to_numpy()
    zz = np.where(valid, zz, np.nan)
    zmean = pd.Series(zz).groupby(st["qid"].to_numpy()).transform("mean").to_numpy()
    I = 1.0 / (1.0 + np.exp(-(zz - zmean)))
    st[col] = np.where(valid, I, 0.5)
    st[col + "_z"] = zz
    return st
