"""Precision@K, Recall@K, NDCG@K for recommended paths (METHODOLOGY 7.2), vectorised.

path: (R, L) int array of topic indices (0-based), -1 = empty slot.
rel:  (R, T) graded relevance (0, 1, 2, ...).
NDCG uses linear gain rel / log2(position + 1) (position starting at 1); IDCG from the ideal ordering of
the relevance vector. Rows with no relevant topic are undefined (NaN).
"""
from __future__ import annotations

import numpy as np


def _gather(rel, path):
    R = rel.shape[0]
    safe = np.where(path >= 0, path, 0)
    g = rel[np.arange(R)[:, None], safe]
    return np.where(path >= 0, g, 0.0)


def precision_at_k(path, rel, k):
    g = _gather(rel, path[:, :k])
    out = (g > 0).sum(1) / k
    return np.where((rel > 0).any(1), out, np.nan)


def recall_at_k(path, rel, k):
    g = _gather(rel, path[:, :k])
    n_rel = (rel > 0).sum(1)
    return np.where(n_rel > 0, (g > 0).sum(1) / np.maximum(n_rel, 1), np.nan)


def ndcg_at_k(path, rel, k):
    g = _gather(rel, path[:, :k]).astype(float)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (g * disc[: g.shape[1]]).sum(1)
    ideal = -np.sort(-rel.astype(float), axis=1)[:, :k]
    idcg = (ideal * disc[: ideal.shape[1]]).sum(1)
    return np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1), np.nan)


def all_ranking(path, rel, ks=(3, 5)) -> dict:
    out = {}
    for k in ks:
        out[f"P@{k}"] = precision_at_k(path, rel, k)
        out[f"R@{k}"] = recall_at_k(path, rel, k)
        out[f"NDCG@{k}"] = ndcg_at_k(path, rel, k)
    return out


def student_mean(values: np.ndarray, student_idx: np.ndarray, n_students: int | None = None) -> np.ndarray:
    """Average over decision points within student (ignoring NaN), returns per-student means (NaN if none)."""
    n = n_students or int(student_idx.max()) + 1
    ok = np.isfinite(values)
    s = np.bincount(student_idx[ok], weights=values[ok], minlength=n)
    c = np.bincount(student_idx[ok], minlength=n)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def mean_over_students(values, student_idx, n_students=None) -> float:
    sm = student_mean(values, student_idx, n_students)
    return float(np.nanmean(sm)) if np.isfinite(sm).any() else np.nan
