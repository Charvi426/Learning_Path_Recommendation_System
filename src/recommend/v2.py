"""v2 recommender (post-hoc): relative candidate filters (Fix 2) and a popularity score term (Fix 3).

Relative filters, two modes:
  within  revision: assessed topic j is flagged when K_{s,j} is among the lowest ceil(q_rev * n) of the
                    student's n assessed topics (ties -> lower topic index first);
          new:      among the student's unassessed topics inside the horizon, the floor(q_new * n) topics with the
                    HIGHEST predicted mastery are excluded as already mastered.
  cohort  revision: flagged when K_{s,j} < the q_rev-quantile of K for topic j in the TRAINING cohort's decision
                    points;
          new:      excluded when predicted mastery > the (1 - q_new)-quantile of the training cohort's predicted
                    mastery for unassessed topic j.
The new-topic filter mirrors the revision filter ("bottom q% flagged for revision" <-> "top q% treated as
mastered"), which is the relative version of M < tau_m.

Pop(k, t) in [0, 1] (row-max normalised), leakage-free options:
  train_window   mean clicks of TRAINING-presentation students on topic k in the same (module, k_now) decision
                 window (= v1 A1 score, past cohorts only);
  cohort_w{w}    clicks of all students of the SAME presentation on topic k in days [t - w, t - 1].
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd

from ..evaluation.ranking_metrics import ndcg_at_k, recall_at_k, student_mean
from .engine import RecData
from .path import build_paths
from .scoring import term_matrix

TERMS6 = ["G", "Q", "F", "D", "I", "Pop"]


# ------------------------------------------------------------------------------------------ data
class RecDataV2(RecData):
    def __init__(self, ks, rel, graphs, models, popularity, topics, n_topics):
        super().__init__(ks, rel, graphs, models, popularity, n_topics)
        tp = topics.set_index(["code_module", "code_presentation", "topic"])
        ex = np.zeros((self.N, n_topics), bool)
        idx = set(tp.index)
        for j in range(1, n_topics + 1):
            ex[:, j - 1] = [(m, p, j) in idx for m, p in zip(self.module, self.pres)]
        self.exists = ex
        self.universe &= ex
        self.pop_terms = {}
        self.cohort_thr = {}

    def subset(self, mask):
        r = super().subset(mask)
        r.exists = self.exists[mask]
        r.pop_terms = {k: v[mask] for k, v in self.pop_terms.items()}
        r.cohort_thr = {k: {kk: vv[mask] for kk, vv in d.items()} for k, d in self.cohort_thr.items()}
        return r


def row_max_normalise(a: np.ndarray, exists: np.ndarray) -> np.ndarray:
    a = np.where(exists, a, 0.0)
    mx = a.max(1, keepdims=True)
    return np.where(mx > 0, a / np.where(mx > 0, mx, 1), 0.0)


def pop_train_window(rd: RecDataV2) -> np.ndarray:
    return row_max_normalise(rd.pop, rd.exists)


def cohort_daily(vd: pd.DataFrame, students: pd.DataFrame) -> pd.DataFrame:
    d = vd.merge(students[["sid", "code_module", "code_presentation"]], on="sid", suffixes=("_x", ""))
    return d.groupby(["code_module", "code_presentation", "topic", "date"], as_index=False)["clicks"].sum()


def pop_cohort_before(rd: RecDataV2, cd: pd.DataFrame, w: int) -> np.ndarray:
    """Clicks of the whole presentation cohort on each topic in days [t - w, t - 1] (strictly before t)."""
    out = np.zeros((rd.N, rd.T))
    for (m, p), g in cd.groupby(["code_module", "code_presentation"]):
        sel = np.flatnonzero((rd.module == m) & (rd.pres == p))
        if not len(sel):
            continue
        dmin, dmax = int(g["date"].min()), int(g["date"].max())
        D = dmax - dmin + 1
        arr = np.zeros((rd.T, D + 1))
        np.add.at(arr, (g["topic"].to_numpy() - 1, g["date"].to_numpy() - dmin + 1), g["clicks"].to_numpy())
        cum = np.cumsum(arr, axis=1)
        t = rd.t[sel]
        hi = np.clip(t - 1 - dmin + 1, 0, D).astype(int)
        lo = np.clip(t - w - 1 - dmin + 1, 0, D).astype(int)
        out[sel] = (cum[:, hi] - cum[:, lo]).T
    return row_max_normalise(out, rd.exists)


def cohort_thresholds(ks_train: pd.DataFrame, rd: RecDataV2, model: str, q_grid) -> dict:
    """Per (module, topic) quantiles from the TRAINING cohort's decision points, broadcast to rd's rows."""
    kt = ks_train[ks_train["kind"] == "decision"]
    out = {}
    for q in q_grid:
        a = kt[kt["assessed"]].groupby(["code_module", "j"])[f"K_{model}"].quantile(q / 100)
        u = kt[~kt["assessed"]].groupby(["code_module", "j"])[f"M_{model}"].quantile(1 - q / 100)
        rev = np.full((rd.N, rd.T), -np.inf)
        new = np.full((rd.N, rd.T), np.inf)
        for j in range(1, rd.T + 1):
            rev[:, j - 1] = [a.get((m, j), -np.inf) for m in rd.module]
            new[:, j - 1] = [u.get((m, j), np.inf) for m in rd.module]
        out[("rev", q)] = rev
        out[("new", q)] = new
    return out


# ------------------------------------------------------------------------------------------ filters
def relative_candidates(rd: RecDataV2, m: dict, Qn, f: dict, horizon=2, use_q=True, model=None):
    """f = {mode, q_rev, q_new, tau_q}. Returns new, rev, excluded_by_mastery (all (N, T) bool)."""
    M, K, assessed = m["M"], m["K"], rd.assessed
    within = rd.exists & (rd.topic_no[None, :] <= rd.k_now[:, None] + horizon)
    pool = ~assessed & within
    tie = 1e-9 * np.arange(rd.T)[None, :]
    if f["mode"] == "within":
        Ka = np.where(assessed & rd.exists, K + tie, np.inf)
        rank = np.argsort(np.argsort(Ka, axis=1, kind="stable"), axis=1, kind="stable")
        n_ass = (assessed & rd.exists).sum(1, keepdims=True)
        rev = assessed & rd.exists & (rank < np.ceil(f["q_rev"] / 100 * n_ass))
        Mp = np.where(pool, -(M - tie), np.inf)                    # descending mastery
        rk = np.argsort(np.argsort(Mp, axis=1, kind="stable"), axis=1, kind="stable")
        n_pool = pool.sum(1, keepdims=True)
        excl = pool & (rk < np.floor(f["q_new"] / 100 * n_pool))
    else:
        thr = rd.cohort_thr[model]
        rev = assessed & rd.exists & (K < thr[("rev", f["q_rev"])])
        excl = pool & (M > thr[("new", f["q_new"])])
    new = pool & ~excl
    if use_q:
        new &= Qn >= f["tau_q"]
    return new, rev, excl


def filter_diagnostics(rd: RecDataV2, new, rev, excl, paths=None) -> dict:
    up = rd.k_now - 1
    rows = np.arange(rd.N)
    up_unassessed = ~rd.assessed[rows, up]
    has_assessed = (rd.assessed & rd.exists).any(1)
    d = {
        "mean_candidates": float((new | rev).sum(1).mean()),
        "mean_new_candidates": float(new.sum(1).mean()),
        "mean_revision_candidates": float(rev.sum(1).mean()),
        "share_upcoming_removed_by_mastery_filter": float(excl[rows, up][up_unassessed].mean()),
        "share_dp_with_revision_flag": float(rev.any(1)[has_assessed].mean()),
        "share_dp_mastery_filter_active": float(excl.any(1).mean()),
    }
    if paths is not None:
        inpath = np.zeros((rd.N, rd.T), bool)
        for pos in range(paths.shape[1]):
            ok = paths[:, pos] >= 0
            inpath[rows[ok], paths[ok, pos]] = True
        d["share_paths_with_revise"] = float((inpath & rev).any(1).mean())
        d["share_paths_with_upcoming"] = float(inpath[rows, up].mean())
        d["mean_path_length"] = float((paths >= 0).sum(1).mean())
    return d


def admissible(diag: dict, max_up_removed=0.10, min_active=0.01) -> bool:
    return (diag["share_upcoming_removed_by_mastery_filter"] <= max_up_removed
            and diag["share_dp_with_revision_flag"] >= min_active
            and diag["share_dp_mastery_filter_active"] >= min_active)


# ------------------------------------------------------------------------------------------ scoring
def terms6(rd: RecDataV2, m: dict, prereq, p_star, pop_key):
    t5, Qn = term_matrix(m["M"], m["K"], rd.assessed, rd.I, prereq, p_star)
    pop = rd.pop_terms[pop_key] if pop_key else np.zeros((rd.N, rd.T))
    return np.concatenate([t5, pop[..., None]], axis=-1), Qn


def weight_grid6(active, step=0.1, eps_max=0.2) -> np.ndarray:
    n = int(round(1 / step))
    idx = [TERMS6.index(a) for a in active]
    out = []
    for comb in itertools.product(range(n + 1), repeat=len(active)):
        if sum(comb) != n:
            continue
        w = np.zeros(6)
        w[idx] = np.array(comb) * step
        if w[4] > eps_max + 1e-9:
            continue
        out.append(w)
    return np.array(out)


def _setup(rd, conf, f, horizon):
    m = rd.models[conf["model"]]
    prereq = rd.prereq[conf["graph"]] if conf.get("graph") else np.zeros((rd.N, rd.T, rd.T), bool)
    terms, Qn = terms6(rd, m, prereq, f["p_star"], conf.get("pop"))
    new, rev, excl = relative_candidates(rd, m, Qn, f, horizon, conf.get("use_q", True), conf["model"])
    ordering = prereq if conf.get("graph") else None
    return terms, new, rev, excl, ordering


def paths_v2(rd, conf, W, f, L=5, max_rev=3, horizon=2):
    terms, new, rev, excl, ordering = _setup(rd, conf, f, horizon)
    cand = new | rev
    P = np.stack([build_paths(terms @ w, cand, rev, ordering, L, max_rev) for w in W])
    return P, {"terms": terms, "new": new, "rev": rev, "excl": excl}


def objective_v2(rd, conf, W, f, L=5, max_rev=3, horizon=2, chunk=48):
    terms, new, rev, excl, ordering = _setup(rd, conf, f, horizon)
    cand = new | rev
    nd, rc = [], []
    for i in range(0, len(W), chunk):
        Wc = W[i:i + chunk]
        nw = len(Wc)
        S = np.einsum("ntc,wc->wnt", terms, Wc).reshape(nw * rd.N, rd.T)
        rep = lambda a: np.broadcast_to(a, (nw,) + a.shape).reshape((nw * rd.N,) + a.shape[1:])  # noqa: E731
        P = build_paths(S, rep(cand), rep(rev), None if ordering is None else rep(ordering), L, max_rev)
        relr = rep(rd.relAB)
        n5 = ndcg_at_k(P, relr, 5).reshape(nw, rd.N)
        r5 = recall_at_k(P, relr, 5).reshape(nw, rd.N)
        for w in range(nw):
            nd.append(float(np.nanmean(student_mean(n5[w], rd.sidx, rd.n_students))))
            rc.append(float(np.nanmean(student_mean(r5[w], rd.sidx, rd.n_students))))
    return np.array(nd), np.array(rc), filter_diagnostics(rd, new, rev, excl)


def _better(a, b):
    return (a[0] > b[0] + 1e-12) or (abs(a[0] - b[0]) <= 1e-12 and a[1] > b[1] + 1e-12)


def tune_v2(rd, conf, rcfg, v2cfg, log=None):
    """Filter settings (admissible only) with equal weights -> weight grid -> filter settings again."""
    kw = {"L": rcfg["K"], "max_rev": rcfg["max_revision"], "horizon": rcfg["horizon"]}
    pgrid = rcfg["p_star_grid"] if "D" in conf["active"] else [rcfg["start"]["p_star"]]
    tq = rcfg["threshold_grid"] if conf.get("use_q", True) else [rcfg["start"]["tau_q"]]
    fgrid = [{"mode": mo, "q_rev": a, "q_new": b, "tau_q": c, "p_star": d}
             for mo, a, b, c, d in itertools.product(v2cfg["modes"], v2cfg["q_grid"], v2cfg["q_grid"], tq, pgrid)]
    W_all = weight_grid6(conf["active"], rcfg["weight_step"], rcfg["eps_max"])
    w_eq = np.zeros((1, 6))
    w_eq[0, [TERMS6.index(a) for a in conf["active"]]] = 1 / len(conf["active"])
    hist, diag_cache = [], {}

    def fsearch(w):
        best, bf = (-np.inf, -np.inf), None
        for f in fgrid:
            nd, rc, dg = objective_v2(rd, conf, w, f, **kw)
            ok = admissible(dg, v2cfg["max_upcoming_removed"], v2cfg["min_active_share"])
            hist.append({"stage": "filters", **f, "w": w[0].round(2).tolist(), "NDCG@5": nd[0], "R@5": rc[0],
                         "admissible": ok, **dg})
            if ok and _better((nd[0], rc[0]), best):
                best, bf = (nd[0], rc[0]), f
        return best, bf

    best, f = fsearch(w_eq)
    if f is None:
        raise RuntimeError(f"no admissible filter setting for {conf['label']}")
    nd, rc, _ = objective_v2(rd, conf, W_all, f, **kw)
    bi = 0
    for i in range(len(W_all)):
        if _better((nd[i], rc[i]), (nd[bi], rc[bi])):
            bi = i
    for i in range(len(W_all)):
        hist.append({"stage": "weights", **f, "w": W_all[i].round(2).tolist(), "NDCG@5": nd[i], "R@5": rc[i]})
    wb = W_all[bi:bi + 1]
    best, f = fsearch(wb)
    P, aux = paths_v2(rd, conf, wb, f, **kw)
    diag = filter_diagnostics(rd, aux["new"], aux["rev"], aux["excl"], P[0])
    if log:
        log.info("v2 tuned %s: filters=%s w=%s NDCG@5=%.4f R@5=%.4f", conf["label"], f, wb[0].round(2).tolist(),
                 best[0], best[1])
    return {"filters": f, "weights": dict(zip(TERMS6, wb[0].round(2).tolist())), "val_NDCG@5": float(best[0]),
            "val_R@5": float(best[1]), "n_weight_vectors": int(len(W_all)), "val_filter_diagnostics": diag,
            "history": hist}
