"""v3 orchestration: variants V0-V4, validation metrics, exam check, knowledge state for any query set."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from ..models.das3h import ForgettingPFA
from ..models.data import build_design, training_targets
from .memory import ActRModel, AttributedDAS3H, Series

WINDOWS = (7, 14, 30, np.inf)


def _wn(w):
    return "inf" if not np.isfinite(w) else str(int(w))


# ------------------------------------------------------------------------------------------------ designs
def design(data, targets, split):
    return build_design(targets, data["state"][split], data["graphs"]["chain"], data["tindex"], WINDOWS)


def due_design(data, split):
    return design(data, training_targets(data["queries"], data["inter"], split), split)


def pair_keys(D):
    p = D.pairs.copy()
    p["sid"] = D.rows["sid"].to_numpy()[p["row"].to_numpy()]
    p["t"] = D.rows["t"].to_numpy()[p["row"].to_numpy()]
    return p


def window_doses(series: Series, rows, t, w):
    D, X = series.day[np.maximum(rows, 0)], series.dose[np.maximum(rows, 0)]
    P = series.valid[np.maximum(rows, 0)] & ~series.is_sub[np.maximum(rows, 0)]
    tc = np.asarray(t, float)[:, None]
    m = P & (D < tc) & (D >= tc - w) if np.isfinite(w) else P & (D < tc)
    return np.where(rows >= 0, np.log1p(np.sum(np.where(m, X, 0.0), axis=1)), 0.0)


# ------------------------------------------------------------------------------------------------ wrappers
class Variant:
    """kind in {'hl_v2', 'hl_att', 'actr', 'das3h_att'}."""

    def __init__(self, kind, series=None, params=None, interest=False, hl_model=None, l2=1e-5, C=1.0, S=120.0):
        self.kind, self.series, self.params, self.interest = kind, series, params or {}, interest
        self.hl_model, self.l2, self.C, self.S = hl_model, l2, C, S
        self.dd = None
        if hl_model is not None and kind == "hl_v2":
            self.m = hl_model

    # --- pair features
    def _decay(self):
        if self.kind == "actr" and self.dd is None:
            self.dd = self.series.decay_rates(self.params)
        return self.dd

    def prep(self, D):
        if self.kind == "hl_v2":
            return D
        p = pair_keys(D)
        rows = self.series.rows_for(p["sid"], p["j"])
        if self.kind == "actr":
            p["m"] = self.series.activation(rows, p["t"], p["t"], self.params, self._decay())
        elif self.kind == "hl_att":
            de, c, d = self.series.attributed_hl_features(rows, p["t"], p["t_submit_j"].fillna(0))
            p["delta"] = np.where(p["assessed"], de, 0.0)
            p["c_inf"], p["d_inf"] = c, d
        elif self.kind == "das3h_att":
            r = D.rows.copy()
            for w in WINDOWS:
                v = window_doses(self.series, rows, p["t"], w)
                r[f"pa_{_wn(w)}"] = np.bincount(p["row"], weights=p["w"] * v, minlength=len(r))
            return type(D)(rows=r, pairs=p, topic_index=D.topic_index)
        return type(D)(rows=D.rows, pairs=p, topic_index=D.topic_index)

    def fit(self, data, D_train):
        Dp = self.prep(D_train)
        if self.kind == "actr":
            self.m = ActRModel(len(data["tindex"]), self.interest, self.l2).fit(Dp.rows, Dp.pairs)
        elif self.kind in ("hl_v2", "hl_att"):
            if self.hl_model is not None and self.kind == "hl_v2":
                self.m = self.hl_model
            else:
                from ..models.train import fit_forgetting
                none = ForgettingPFA("none", False, l2=1e-5, n_topics=len(data["tindex"])).fit(Dp)
                self.m = fit_forgetting("learnedI" if self.interest else "learned", {"train": Dp}, data, self.l2,
                                        self.S, init_from=none)
        else:
            self.m = AttributedDAS3H("das3h", C=self.C, windows=WINDOWS).fit(Dp)
        return self

    def predict(self, D):
        Dp = self.prep(D)
        if self.kind == "actr":
            return self.m.predict(Dp.rows, Dp.pairs)
        return self.m.predict_proba(Dp)

    def K_assessed(self, frame, t_cut, t_eval):
        """frame: sid, j, assessed, M (score), I, c_inf, d_inf, delta (state at t_cut), t_submit_j.
        Returns K and R for assessed rows at t_eval with no practice after t_cut."""
        gap = np.asarray(t_eval, float) - np.asarray(t_cut, float)
        ass = frame["assessed"].to_numpy(bool)
        if self.kind == "actr":
            rows = self.series.rows_for(frame["sid"], frame["j"])
            f = frame.copy()
            f["m"] = self.series.activation(rows, t_cut, t_eval, self.params, self._decay())
            K = self.m.K_of(f)
            ts = frame["t_submit_j"].fillna(0).to_numpy(float) + 1
            f0 = frame.copy()
            f0["m"] = self.series.activation(rows, ts, ts, self.params, self._decay())
            K0 = self.m.K_of(f0)
            R = np.clip(np.where(K0 > 0, K / np.where(K0 > 0, K0, 1), 1.0), 0, 1)
            return np.where(ass, K, np.nan), np.where(ass, R, np.nan)
        if self.kind in ("hl_v2", "hl_att"):
            f = frame.copy()
            if self.kind == "hl_att":
                rows = self.series.rows_for(frame["sid"], frame["j"])
                de, c, d = self.series.attributed_hl_features(rows, t_cut, frame["t_submit_j"].fillna(0))
                f["delta"], f["c_inf"], f["d_inf"] = de, c, d
            f["delta"] = f["delta"].fillna(0) + gap
            R, _ = self.m.retention_state(f)
            M = frame["M"].to_numpy(float)
            return np.where(ass, M * R, np.nan), np.where(ass, R, np.nan)
        return None, None   # das3h: K comes from predictions for every topic


# ------------------------------------------------------------------------------------------------ metrics
def within_course_auc(y, p, module):
    tot, n = 0.0, 0
    for m in np.unique(module):
        s = module == m
        if len(np.unique(y[s])) == 2:
            tot += roc_auc_score(y[s], p[s]) * s.sum()
            n += s.sum()
    return tot / n if n else np.nan


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def knowledge_at(var: Variant, data, qsel: pd.DataFrame, split, t_eval_col="t"):
    """Knowledge state for every (query, topic j that exists) at t_cut = query t and t_eval = qsel[t_eval_col].
    Returns qid, sid, j, assessed, M_obs, p (prediction if attempted at t_cut), K, M, R."""
    st = data["state"][split]
    st = st[st["qid"].isin(qsel["qid"])]
    ex = data["topics"][["code_module", "code_presentation", "topic"]].rename(columns={"topic": "j"})
    q = qsel[["qid", "sid", "t", "code_module", "code_presentation"] + ([t_eval_col] if t_eval_col != "t" else [])]
    f = st.merge(q.drop(columns="code_module"), on="qid").merge(ex, on=["code_module", "code_presentation", "j"])
    f = f.sort_values(["qid", "j"]).reset_index(drop=True)
    targets = f[["qid", "sid", "j", "t", "code_module"]].rename(columns={"j": "k"})
    p = var.predict(design(data, targets, split))
    inter = data["inter"][["sid", "topic", "t_submit"]].rename(columns={"topic": "j", "t_submit": "t_submit_j2"})
    f = f.merge(inter, on=["sid", "j"], how="left")
    f["t_submit_j"] = f["t_submit_j"].fillna(f["t_submit_j2"])
    t_eval = f[t_eval_col].to_numpy(float)
    K, R = var.K_assessed(f, f["t"].to_numpy(float), t_eval)
    ass = f["assessed"].to_numpy(bool)
    f["p"] = p
    f["M_obs"] = f["M"]
    if K is None:                                   # das3h: K = prediction for every topic
        f["K"] = p
        f["R"] = np.nan
    else:
        f["K"] = np.where(ass, K, p)
        f["R"] = R
    # mastery-scale knowledge for the recommender/planner: K_score = M * R (assessed), p otherwise
    f["K_score"] = np.where(ass, f["M_obs"].to_numpy(float) * np.nan_to_num(f["R"].to_numpy(float), nan=1.0), p)
    f["Mhat"] = np.where(ass, f["M_obs"], p)
    return f


def exam_delta_rho(kst: pd.DataFrame, students: pd.DataFrame):
    g = kst.groupby("sid").agg(M=("Mhat", "mean"), K=("K", "mean")).reset_index().merge(
        students[["sid", "exam_score"]], on="sid")
    g = g[g["exam_score"].notna()]
    rM = sps.spearmanr(g["M"], g["exam_score"])[0]
    rK = sps.spearmanr(g["K"], g["exam_score"])[0]
    return {"n": len(g), "rho_M": rM, "rho_K": rK, "delta_rho": rK - rM}, g


def paired_delta_rho_diff(g_new, g_ref, nb=1000, seed=0):
    """bootstrap CI of (Δρ_new − Δρ_ref) over students (paired)."""
    m = g_new.merge(g_ref, on=["sid", "exam_score"], suffixes=("_n", "_r"))
    ex = m["exam_score"].to_numpy()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nb):
        i = rng.integers(0, len(m), len(m))
        dn = sps.spearmanr(m["K_n"].to_numpy()[i], ex[i])[0] - sps.spearmanr(m["M_n"].to_numpy()[i], ex[i])[0]
        dr = sps.spearmanr(m["K_r"].to_numpy()[i], ex[i])[0] - sps.spearmanr(m["M_r"].to_numpy()[i], ex[i])[0]
        out.append(dn - dr)
    dn = sps.spearmanr(m["K_n"], ex)[0] - sps.spearmanr(m["M_n"], ex)[0]
    dr = sps.spearmanr(m["K_r"], ex)[0] - sps.spearmanr(m["M_r"], ex)[0]
    return {"delta_rho_new": dn, "delta_rho_ref": dr, "improvement": dn - dr,
            "ci_lo": float(np.percentile(out, 2.5)), "ci_hi": float(np.percentile(out, 97.5)), "n": len(m)}


# ------------------------------------------------------------------------------------------------ tuning
def evaluate_variant(var, data, D_train, D_val, exam_q=None):
    var.fit(data, D_train)
    p = var.predict(D_val)
    y = D_val.rows["y"].to_numpy(float)
    out = {"val_logloss": logloss(y, p), "val_within_course_auc": within_course_auc(
        y, p, D_val.rows["code_module"].to_numpy())}
    if exam_q is not None:
        kst = knowledge_at(var, data, exam_q, "val")
        d, _ = exam_delta_rho(kst, data["students"])
        out.update({f"exam_{k}": v for k, v in d.items()})
    return out


def tune_actr(rule, series, data, D_train, D_val, log=None, maxiter=25):
    """Outer L-BFGS-B (finite differences) on validation log-loss; inner coefficient fit on train."""
    if rule == "fixed":
        x0, bounds = np.array([0.5, 1.0]), [(0.05, 2.0), (np.log(0.1), np.log(30.0))]
        mk = lambda x: {"rule": "fixed", "d": float(x[0]), "x_sub": float(np.exp(x[1]))}  # noqa: E731
    else:
        x0, bounds = np.array([0.177, 0.217, 1.0]), [(0.02, 2.0), (0.0, 0.5), (np.log(0.1), np.log(30.0))]
        mk = lambda x: {"rule": "spacing", "a": float(x[0]), "c": float(x[1]), "x_sub": float(np.exp(x[2]))}  # noqa: E731
    cache = {}

    def f(x):
        key = tuple(np.round(x, 6))
        if key not in cache:
            var = Variant("actr", series, mk(x))
            var.fit(data, D_train)
            p = var.predict(D_val)
            cache[key] = logloss(D_val.rows["y"].to_numpy(float), p)
            if log:
                log.info("  actr %s %s -> val logloss %.5f", rule, mk(x), cache[key])
        return cache[key]

    # coarse grid first: the validation loss is flat at the scale of finite-difference steps (a first attempt with
    # eps = 1e-3 stopped at the starting point), then L-BFGS-B refinement from the best grid point
    import itertools
    if rule == "fixed":
        grid = [np.array([d, np.log(x)]) for d, x in itertools.product([0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5],
                                                                        [0.5, 2.7, 10.0])]
    else:
        grid = [np.array([a, c, np.log(x)]) for a, c, x in itertools.product([0.05, 0.177, 0.35, 0.6],
                                                                              [0.0, 0.1, 0.217, 0.4], [0.5, 2.7, 10.0])]
    vals = [f(g) for g in grid]
    x0 = grid[int(np.argmin(vals))]
    res = minimize(f, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "eps": 2e-2})
    xb = res.x if res.fun <= min(vals) else x0
    return mk(xb), float(min(res.fun, min(vals))), len(cache)
