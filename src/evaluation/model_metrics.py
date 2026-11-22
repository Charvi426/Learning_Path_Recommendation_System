"""Knowledge-model metrics (METHODOLOGY 7.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def log_loss(y, p, eps=1e-12):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def all_metrics(y, p, y_score=None) -> dict:
    out = {"AUC": auc(y, p), "logloss": log_loss(y, p), "Brier": brier(y, p), "RMSE": rmse(y, p), "MAE": mae(y, p)}
    if y_score is not None:
        out["RMSE_score"] = rmse(y_score, p)
        out["MAE_score"] = mae(y_score, p)
    return out


def calibration_table(y, p, n_bins=10) -> pd.DataFrame:
    """Equal-width bins on [0, 1]."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"bin": b, "y": y, "p": p})
    g = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean")).reset_index()
    g["bin_lo"], g["bin_hi"] = g["bin"] / n_bins, (g["bin"] + 1) / n_bins
    return g


def ece(y, p, n_bins=10) -> float:
    c = calibration_table(y, p, n_bins)
    return float(np.sum(c["n"] * np.abs(c["mean_pred"] - c["obs_rate"])) / c["n"].sum())
