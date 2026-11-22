"""Vectorised recommender over all decision points of a split: candidates -> scores -> paths -> metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.ranking_metrics import all_ranking, student_mean
from ..evaluation.relevance import rel_matrix
from .candidates import candidates
from .path import build_paths
from .scoring import term_matrix


class RecData:
    def __init__(self, ks: pd.DataFrame, rel: pd.DataFrame, graphs: dict, models: list[str],
                 popularity: pd.DataFrame | None = None, n_topics: int = 6):
        ks = ks[ks["kind"] == "decision"].sort_values(["qid", "j"])
        qids = ks["qid"].drop_duplicates().to_numpy()
        rel = rel.set_index("qid").reindex(qids).reset_index()
        self.T = T = n_topics
        self.N = N = len(qids)
        self.qid = qids
        g = lambda c: ks[c].to_numpy().reshape(N, T)  # noqa: E731
        self.assessed = g("assessed").astype(bool)
        self.I = g("I").astype(float)
        self.opened = g("clicks_raw_inf") > 0
        self.delta = g("delta").astype(float)
        self.M_obs = g("M_obs").astype(float)
        first = ks.groupby("qid", sort=True).first().reindex(qids)
        self.k_now = first["k_now"].to_numpy()
        self.sid = first["sid"].to_numpy()
        self.t = first["t"].to_numpy()
        self.module = first["code_module"].astype(str).to_numpy()
        self.pres = first["code_presentation"].astype(str).to_numpy()
        self.ability = first["ability"].to_numpy()
        _, self.sidx = np.unique(self.sid, return_inverse=True)
        self.n_students = int(self.sidx.max()) + 1
        self.topic_no = np.arange(1, T + 1)
        self.models = {}
        for m in models:
            self.models[m] = {"M": g(f"M_{m}").astype(float), "K": g(f"K_{m}").astype(float),
                              "R": g(f"R_{m}").astype(float), "p": g(f"p_{m}").astype(float)}
        mods = sorted(set(self.module))
        self.prereq = {}
        for gname, gd in graphs.items():
            arr = np.zeros((len(mods), T, T), bool)
            for mi, m in enumerate(mods):
                for k, plist in gd.get(m, {}).items():
                    for j in plist:
                        arr[mi, k - 1, j - 1] = True
            self.prereq[gname] = arr[np.searchsorted(mods, self.module)]
        self.relA = rel_matrix(rel, T, "A")
        self.relB = rel_matrix(rel, T, "B")
        self.relAB = rel_matrix(rel, T, "AB")
        self.group = rel["group"].to_numpy()
        self.universe = self.assessed | (self.topic_no[None, :] <= self.k_now[:, None] + 2)
        if popularity is not None:
            pop = pd.DataFrame({"code_module": np.repeat(self.module, T), "k_now": np.repeat(self.k_now, T),
                                "j": np.tile(self.topic_no, N)})
            pop = pop.merge(popularity, on=["code_module", "k_now", "j"], how="left")
            self.pop = pop["pop_clicks"].fillna(0).to_numpy().reshape(N, T)

    def subset(self, mask: np.ndarray) -> "RecData":
        import copy
        r = copy.copy(self)
        for a in ["assessed", "I", "opened", "delta", "M_obs", "k_now", "sid", "t", "module", "pres", "ability",
                  "relA", "relB", "relAB", "group", "universe", "qid"]:
            setattr(r, a, getattr(self, a)[mask])
        if hasattr(self, "pop"):
            r.pop = self.pop[mask]
        r.models = {m: {k: v[mask] for k, v in d.items()} for m, d in self.models.items()}
        r.prereq = {g: v[mask] for g, v in self.prereq.items()}
        r.N = int(mask.sum())
        _, r.sidx = np.unique(r.sid, return_inverse=True)
        r.n_students = int(r.sidx.max()) + 1 if r.N else 0
        return r


def make_paths(rd: RecData, conf: dict, W: np.ndarray, thr: dict, L=5, max_rev=3, horizon=2, seed=0,
               return_terms=False):
    """Returns paths (n_w, N, L) for a scoring configuration (or baseline)."""
    kind = conf["kind"]
    if kind in ("random", "popularity"):
        if kind == "random":
            s = np.random.default_rng(seed).random((rd.N, rd.T))
        else:
            s = rd.pop + 1e-6 * np.random.default_rng(seed).random((rd.N, rd.T))
        cand = rd.universe
        p = build_paths(s, cand, np.zeros_like(cand), None, L, max_rev=L)
        return p[None], None
    m = rd.models[conf["model"]]
    prereq = rd.prereq[conf["graph"]] if conf.get("graph") else np.zeros((rd.N, rd.T, rd.T), bool)
    terms, Qn = term_matrix(m["M"], m["K"], rd.assessed, rd.I, prereq, thr["p_star"])
    new, rev = candidates(m["M"], m["K"], Qn, rd.assessed, rd.topic_no, rd.k_now, thr["tau_m"], thr["tau_q"],
                          thr["tau_r"], horizon, use_q=conf.get("use_q", True))
    cand = new | rev
    paths = []
    ordering = prereq if conf.get("graph") else None
    for w in W:
        s = terms @ w
        paths.append(build_paths(s, cand, rev, ordering, L, max_rev))
    P = np.stack(paths)
    if return_terms:
        return P, {"terms": terms, "new": new, "rev": rev, "Qn": Qn}
    return P, None


def evaluate_paths(rd: RecData, P: np.ndarray, rels=("AB",), ks=(3, 5)):
    """P (n_w, N, L). Returns list (per weight) of dict metric -> mean over students, and per-student arrays."""
    out = []
    for w in range(P.shape[0]):
        row = {}
        per_student = {}
        for rn in rels:
            rel = getattr(rd, f"rel{rn}")
            mets = all_ranking(P[w], rel, ks)
            for mname, v in mets.items():
                sm = student_mean(v, rd.sidx, rd.n_students)
                row[f"{mname}[{rn}]"] = float(np.nanmean(sm)) if np.isfinite(sm).any() else np.nan
                per_student[f"{mname}[{rn}]"] = sm
        out.append((row, per_student))
    return out


def objective(rd, conf, W, thr, chunk=48, **kw):
    """NDCG@5 and Recall@5 on combined relevance for each weight vector (vectorised over weights)."""
    nd, rc = [], []
    m = rd.models[conf["model"]] if conf["kind"] == "score" else None
    prereq = rd.prereq[conf["graph"]] if conf.get("graph") else np.zeros((rd.N, rd.T, rd.T), bool)
    terms, Qn = term_matrix(m["M"], m["K"], rd.assessed, rd.I, prereq, thr["p_star"])
    new, rev = candidates(m["M"], m["K"], Qn, rd.assessed, rd.topic_no, rd.k_now, thr["tau_m"], thr["tau_q"],
                          thr["tau_r"], kw.get("horizon", 2), use_q=conf.get("use_q", True))
    cand = new | rev
    ordering = prereq if conf.get("graph") else None
    rel = rd.relAB
    from ..evaluation.ranking_metrics import ndcg_at_k, recall_at_k
    for i in range(0, len(W), chunk):
        Wc = W[i:i + chunk]
        nw = len(Wc)
        S = np.einsum("ntc,wc->wnt", terms, Wc).reshape(nw * rd.N, rd.T)
        rep = lambda a: np.broadcast_to(a, (nw,) + a.shape).reshape((nw * rd.N,) + a.shape[1:])  # noqa: E731
        P = build_paths(S, rep(cand), rep(rev), None if ordering is None else rep(ordering),
                        kw.get("L", 5), kw.get("max_rev", 3))
        relr = rep(rel)
        n5 = ndcg_at_k(P, relr, 5).reshape(nw, rd.N)
        r5 = recall_at_k(P, relr, 5).reshape(nw, rd.N)
        for a, store in [(n5, nd), (r5, rc)]:
            for w in range(nw):
                sm = student_mean(a[w], rd.sidx, rd.n_students)
                store.append(float(np.nanmean(sm)))
    return np.array(nd), np.array(rc)
