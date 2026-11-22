"""Bootstrap intervals over students and paired Wilcoxon tests with Holm correction (METHODOLOGY 7.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def bootstrap_mean_ci(per_student: np.ndarray, n_boot=1000, seed=0, alpha=0.05):
    v = np.asarray(per_student, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    bs = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


def bootstrap_rows_by_student(fn, student: np.ndarray, *arrays, n_boot=1000, seed=0, alpha=0.05):
    """CI for a row-level statistic fn(*arrays) by resampling students (all rows of a student together)."""
    uniq, inv = np.unique(student, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rng = np.random.default_rng(seed)
    est = fn(*arrays)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([order[starts[p]:starts[p] + counts[p]] for p in pick])
        try:
            vals.append(fn(*[a[rows] for a in arrays]))
        except ValueError:
            continue
    vals = np.array(vals, float)
    return float(est), float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def fast_student_bootstrap_indices(student: np.ndarray, n_boot=1000, seed=0):
    """Pre-computes row-weight matrix (n_boot, n_rows) for bootstrap by student (multiplicity weights)."""
    uniq, inv = np.unique(student, return_inverse=True)
    rng = np.random.default_rng(seed)
    W = np.empty((n_boot, len(student)), np.float32)
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        W[b] = cnt[inv]
    return W


def wilcoxon_holm(pairs: dict[str, tuple[np.ndarray, np.ndarray]], alpha=0.05) -> pd.DataFrame:
    """pairs: name -> (x, y) per-student values (aligned). Two-sided paired Wilcoxon signed-rank,
    Holm-Bonferroni across the family."""
    rows = []
    for name, (x, y) in pairs.items():
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        d = x[ok] - y[ok]
        nz = d[d != 0]
        if len(nz) == 0:
            stat, p = np.nan, 1.0
        else:
            res = sps.wilcoxon(x[ok], y[ok], zero_method="wilcox", alternative="two-sided")
            stat, p = float(res.statistic), float(res.pvalue)
        rows.append({"comparison": name, "n_students": int(ok.sum()), "n_nonzero_diff": int(len(nz)),
                     "mean_x": float(x[ok].mean()), "mean_y": float(y[ok].mean()), "mean_diff": float(d.mean()),
                     "median_diff": float(np.median(d)), "share_x_better": float((d > 0).mean()),
                     "share_y_better": float((d < 0).mean()), "W": stat, "p_raw": p})
    df = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    m = len(df)
    adj, running = [], 0.0
    for i, p in enumerate(df["p_raw"]):
        running = max(running, min(1.0, (m - i) * p))
        adj.append(running)
    df["p_holm"] = adj
    df["significant_holm"] = df["p_holm"] < alpha
    return df
