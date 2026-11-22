"""Main forgetting-aware PFA (METHODOLOGY 5.1, variants 5.2) and the DAS3H-window benchmark (5.3).

logit P(y_{s,k}=1) = beta_k + th_a*ability + th_K*Kbar_{s,P(k)}(t_k) + th_c*c_{s,k,inf}
                     [+ th_I*I_{s,k} + th_Ic*I_{s,k}*c_{s,k,inf}]
Kbar = mean_{j in P(k)} M_{s,j} * R_{s,j}(t_k)
R: none -> 1; fixed -> exp(-Delta/S); learned -> 2^(-Delta/h), log2 h = phi0 + phi_c c_j + phi_d d_j [+ phi_I I_j]
Fitted by L-BFGS-B on L2-regularised mean log-loss with an analytic gradient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from ..features.build import wname
from .data import Design
from .decay import LN2, log2_half_life, retention_fixed, retention_hlr


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


class ForgettingPFA:
    def __init__(self, forgetting: str = "learned", interest: bool = False, S: float = 30.0, l2: float = 1e-3,
                 h_min: float = 1.0, h_max: float = 365.0, n_topics: int | None = None):
        assert forgetting in {"none", "fixed", "learned"}
        self.forgetting, self.interest, self.S, self.l2 = forgetting, interest, S, l2
        self.h_min, self.h_max = h_min, h_max
        self.n_topics = n_topics

    # ---------------- data preparation -----------------
    def _fit_scalers(self, D: Design):
        r = D.rows
        self.mu_ = {c: float(r[c].mean()) for c in ["ability", "c_inf", "I"]}
        self.sd_ = {c: float(r[c].std() or 1.0) for c in ["ability", "c_inf", "I"]}
        p = D.pairs[D.pairs["assessed"]]
        self.pmu_ = {c: float(p[c].mean()) for c in ["c_inf", "d_inf", "I"]}
        self.psd_ = {c: float(p[c].std() or 1.0) for c in ["c_inf", "d_inf", "I"]}

    def own_X(self, rows: pd.DataFrame) -> np.ndarray:
        a = (rows["ability"].to_numpy() - self.mu_["ability"]) / self.sd_["ability"]
        c = (rows["c_inf"].to_numpy() - self.mu_["c_inf"]) / self.sd_["c_inf"]
        cols = [a, c]
        if self.interest:
            i = (rows["I"].to_numpy() - self.mu_["I"]) / self.sd_["I"]
            cols += [i, i * c]
        return np.column_stack(cols)

    def pair_Z(self, pairs: pd.DataFrame) -> np.ndarray:
        cols = [(pairs["c_inf"].to_numpy(float) - self.pmu_["c_inf"]) / self.psd_["c_inf"],
                (pairs["d_inf"].to_numpy(float) - self.pmu_["d_inf"]) / self.psd_["d_inf"]]
        if self.interest:
            cols.append((pairs["I"].to_numpy(float) - self.pmu_["I"]) / self.psd_["I"])
        return np.column_stack(cols)

    def _prep(self, D: Design):
        P = {"topic": D.rows["topic_idx"].to_numpy(), "X": self.own_X(D.rows), "n": D.n,
             "prow": D.pairs["row"].to_numpy(np.int64), "pwM": (D.pairs["w"] * D.pairs["M"]).to_numpy(float),
             "delta": D.pairs["delta"].to_numpy(float)}
        P["Z"] = self.pair_Z(D.pairs) if self.forgetting == "learned" else np.zeros((len(D.pairs), 0))
        return P

    # ---------------- parameters -----------------
    def _layout(self, nX, nZ):
        nT = self.n_topics
        sl = {"beta": slice(0, nT), "theta": slice(nT, nT + nX), "thK": nT + nX}
        nphi = (1 + nZ) if self.forgetting == "learned" else 0
        sl["phi"] = slice(nT + nX + 1, nT + nX + 1 + nphi)
        return sl, nT + nX + 1 + nphi

    def _R(self, w, P):
        if self.forgetting == "none":
            return np.ones_like(P["delta"]), None, None
        if self.forgetting == "fixed":
            return retention_fixed(P["delta"], self.S), None, None
        g, inside = log2_half_life(w[self.sl_["phi"]], P["Z"], self.h_min, self.h_max)
        return retention_hlr(P["delta"], g), g, inside

    def _eta(self, w, P):
        R, g, inside = self._R(w, P)
        Kbar = np.bincount(P["prow"], weights=P["pwM"] * R, minlength=P["n"])
        eta = w[self.sl_["beta"]][P["topic"]] + P["X"] @ w[self.sl_["theta"]] + w[self.sl_["thK"]] * Kbar
        return eta, Kbar, R, g, inside

    def _loss_grad(self, w, P, y, sw):
        eta, Kbar, R, g, inside = self._eta(w, P)
        p = _sig(eta)
        eps = 1e-12
        N = sw.sum()
        loss = -np.sum(sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) / N
        r = sw * (p - y) / N
        grad = np.zeros_like(w)
        grad[self.sl_["beta"]] = np.bincount(P["topic"], weights=r, minlength=self.n_topics)
        grad[self.sl_["theta"]] = P["X"].T @ r
        grad[self.sl_["thK"]] = Kbar @ r
        pen_idx = list(range(self.sl_["theta"].start, self.sl_["theta"].stop)) + [self.sl_["thK"]]
        if self.forgetting == "learned":
            phi_sl = self.sl_["phi"]
            dRdg = R * LN2 * LN2 * P["delta"] * np.exp2(-g) * inside
            dEta_dg = w[self.sl_["thK"]] * P["pwM"] * dRdg            # per pair
            rp = r[P["prow"]] * dEta_dg
            Zf = np.column_stack([np.ones(len(rp)), P["Z"]])
            grad[phi_sl] = Zf.T @ rp
            pen_idx += list(range(phi_sl.start + 1, phi_sl.stop))   # phi0 not penalised
        pen_idx = np.array(pen_idx)
        loss += 0.5 * self.l2 * np.sum(w[pen_idx] ** 2)
        grad[pen_idx] += self.l2 * w[pen_idx]
        return loss, grad

    # ---------------- public API -----------------
    def fit(self, D: Design, y=None, init: np.ndarray | None = None, sample_weight=None, scalers_from=None):
        if scalers_from is None:
            self._fit_scalers(D)
        else:
            for a in ["mu_", "sd_", "pmu_", "psd_"]:
                setattr(self, a, getattr(scalers_from, a))
        y = D.rows["y"].to_numpy(float) if y is None else y
        sw = np.ones(len(y)) if sample_weight is None else sample_weight
        P = self._prep(D)
        self.sl_, npar = self._layout(P["X"].shape[1], P["Z"].shape[1])
        bounds = [(None, None)] * npar
        if self.forgetting == "learned":
            lo, hi = np.log2(self.h_min), np.log2(self.h_max)
            bounds[self.sl_["phi"].start] = (lo, hi)
        starts = [init] if init is not None else self._default_starts(npar, y)
        best = None
        for w0 in starts:
            res = minimize(self._loss_grad, w0, args=(P, y, sw), jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 2000, "gtol": 1e-7})
            if best is None or res.fun < best.fun:
                best = res
        self.w_ = best.x
        self.converged_ = bool(best.success)
        self.train_loss_ = float(best.fun)
        return self

    def _default_starts(self, npar, y):
        w = np.zeros(npar)
        pbar = np.clip(y.mean(), 1e-3, 1 - 1e-3)
        w[self.sl_["beta"]] = np.log(pbar / (1 - pbar))
        w[self.sl_["thK"]] = 1.0
        if self.forgetting != "learned":
            return [w]
        starts = []
        for g0 in [np.log2(7), np.log2(30), np.log2(120)]:
            w2 = w.copy()
            w2[self.sl_["phi"].start] = g0
            starts.append(w2)
        return starts

    def predict_proba(self, D: Design) -> np.ndarray:
        P = self._prep(D)
        return _sig(self._eta(self.w_, P)[0])

    def retention_state(self, state_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """R and h for assessed topics given state rows (delta, c_inf, d_inf, I at the query time)."""
        delta = state_rows["delta"].to_numpy(float)
        if self.forgetting == "none":
            return np.ones_like(delta), np.full_like(delta, np.inf)
        if self.forgetting == "fixed":
            return retention_fixed(delta, self.S), np.full_like(delta, self.S * LN2)
        Z = self.pair_Z(state_rows)
        g, _ = log2_half_life(self.w_[self.sl_["phi"]], Z, self.h_min, self.h_max)
        return retention_hlr(delta, g), np.exp2(g)

    def coef_table(self, topic_names: list[str]) -> pd.DataFrame:
        names = [f"beta[{t}]" for t in topic_names] + ["theta_a", "theta_c"]
        if self.interest:
            names += ["theta_I", "theta_Ic"]
        names += ["theta_K"]
        if self.forgetting == "learned":
            names += ["phi_0", "phi_c", "phi_d"] + (["phi_I"] if self.interest else [])
        return pd.DataFrame({"param": names, "estimate": self.w_})


# ---------------------------------------------------------------------------------------------
class SklearnLogit:
    """Logistic models on hand-built columns: topic difficulty (IRT-like), PFA, DAS3H windows."""

    def __init__(self, kind: str, C: float = 1.0, windows=(7, 30, np.inf), seed: int = 0):
        self.kind, self.C, self.windows, self.seed = kind, C, list(windows), seed

    def columns(self) -> list[str]:
        if self.kind == "irt":
            return ["ability"]
        if self.kind == "pfa":
            return ["ability", "succ_inf", "fail_inf", "c_inf"]
        if self.kind == "das3h":
            cols = ["ability"]
            for w in self.windows:
                cols += [f"c_{wname(w)}", f"succ_{wname(w)}", f"fail_{wname(w)}"]
            return cols + ["I", "Ixc"]
        raise ValueError(self.kind)

    def _X(self, D: Design, fit=False):
        r = D.rows.copy()
        r["Ixc"] = (r["I"] - 0.5) * r["c_inf"]
        X = r[self.columns()].to_numpy(float)
        if fit:
            self.mu_, self.sd_ = X.mean(0), X.std(0)
            self.sd_[self.sd_ == 0] = 1
        X = (X - self.mu_) / self.sd_
        T = np.zeros((len(r), len(D.topic_index)))
        T[np.arange(len(r)), r["topic_idx"].to_numpy()] = 1
        return np.hstack([T, X])

    def fit(self, D: Design, y=None, sample_weight=None):
        y = D.rows["y"].to_numpy(float) if y is None else y
        self.m_ = LogisticRegression(C=self.C, max_iter=5000, fit_intercept=False)
        self.m_.fit(self._X(D, fit=True), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, D: Design) -> np.ndarray:
        return self.m_.predict_proba(self._X(D))[:, 1]

    def coef_table(self) -> pd.DataFrame:
        names = [f"beta[{i}]" for i in range(self.m_.coef_.shape[1] - len(self.columns()))] + self.columns()
        return pd.DataFrame({"param": names, "estimate": self.m_.coef_[0]})


class Majority:
    def fit(self, D, y=None, sample_weight=None):
        y = D.rows["y"].to_numpy(float) if y is None else y
        self.p_ = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        return self

    def predict_proba(self, D):
        return np.full(D.n, self.p_)
