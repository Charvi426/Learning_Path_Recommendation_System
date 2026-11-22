"""v3 Step 3 (post-hoc): history-based memory (ACT-R activation, Pavlik & Anderson 2005) and a learned
combination of mastery and activation.

Events of (student s, topic j): every day d with attributed practice dose x > 0, plus the assessment submission
of j as one event with dose x_sub (a parameter). For a query (cut-off t_cut, evaluation day t_eval >= t_cut):

    m(s, j) = ln( sum_{i: day_i < t_cut} x_i * (t_eval - day_i + 1) ** (-d_i) )

  fixed decay:      d_i = d
  spacing-sensitive d_i = c * exp(m at day_i) + a      (exp(m at day_i) = summed decayed dose before day_i)

K(s, j, t) = sigmoid(b0 + bM * logit(M_sj) + bm * z(m) [+ bI * z(I_sj)]),  M = normalised score clipped to
[0.01, 0.99]. Outcome model: logit P(y_k) = beta_k + theta . x + theta_K * mean_{j in P(k)} K(s, j, t_k);
a prerequisite never submitted before t contributes K = 0 (as in v1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..models.das3h import SklearnLogit


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


# ------------------------------------------------------------------------------------------------ series
class Series:
    """Padded per-(sid, topic) event arrays: day, dose (practice), is_sub."""

    def __init__(self, practice: pd.DataFrame, inter: pd.DataFrame, keys: pd.DataFrame | None = None):
        sub = inter[inter["is_submitted"] == 1][["sid", "topic", "t_submit"]].rename(columns={"t_submit": "date"})
        sub = sub.assign(dose=0.0, is_sub=True)
        pr = practice[["sid", "topic", "date", "dose"]].assign(is_sub=False)
        if keys is not None:
            k = keys[["sid", "topic"]].drop_duplicates()
            pr = pr.merge(k, on=["sid", "topic"])
            sub = sub.merge(k, on=["sid", "topic"])
        ev = pd.concat([pr, sub], ignore_index=True).sort_values(["sid", "topic", "date", "is_sub"])
        g = ev.groupby(["sid", "topic"], sort=True)
        n = g.size()
        self.keys = n.index
        self.row = {k: i for i, k in enumerate(self.keys)}
        S, E = len(n), int(n.max()) if len(n) else 1
        self.E = E
        pos = g.cumcount().to_numpy()
        r = np.repeat(np.arange(S), n.to_numpy())
        self.day = np.full((S, E), np.inf)
        self.dose = np.zeros((S, E))
        self.is_sub = np.zeros((S, E), bool)
        self.day[r, pos] = ev["date"].to_numpy(float)
        self.dose[r, pos] = ev["dose"].to_numpy(float)
        self.is_sub[r, pos] = ev["is_sub"].to_numpy(bool)
        self.n = n.to_numpy()
        self.valid = np.isfinite(self.day)

    def rows_for(self, sid, topic) -> np.ndarray:
        return np.array([self.row.get((int(s), int(j)), -1) for s, j in zip(sid, topic)])

    def doses(self, x_sub):
        return np.where(self.is_sub, x_sub, self.dose)

    def decay_rates(self, params: dict, chunk=4000) -> np.ndarray:
        """per-event decay d_i (fixed: d; spacing: c * exp(m_i) + a with m_i from strictly earlier days)."""
        if params["rule"] == "fixed":
            return np.full(self.day.shape, params["d"])
        a, c = params["a"], params["c"]
        X = self.doses(params["x_sub"])
        dd = np.zeros_like(self.day)
        order = np.argsort(self.n)
        for s0 in range(0, len(order), chunk):
            idx = order[s0:s0 + chunk]
            e = int(self.n[idx].max())
            D, Xc = self.day[idx, :e], X[idx, :e]
            V = self.valid[idx, :e]
            dc = np.zeros((len(idx), e))
            np.seterr(invalid="ignore")
            for i in range(e):
                if i == 0:
                    dc[:, 0] = a
                    continue
                prior = V[:, :i] & (D[:, :i] < D[:, i:i + 1])
                age = np.where(prior, D[:, i:i + 1] - D[:, :i] + 1, 1.0)
                S = np.sum(np.where(prior, Xc[:, :i] * age ** (-dc[:, :i]), 0.0), axis=1)
                dc[:, i] = c * S + a
            dd[idx, :e] = dc
        return dd

    def activation(self, rows, t_cut, t_eval, params, dd=None, chunk=20000) -> np.ndarray:
        """m for each query (series row, cut-off, evaluation day). rows == -1 -> -inf."""
        X = self.doses(params["x_sub"])
        dd = self.decay_rates(params) if dd is None else dd
        out = np.full(len(rows), -np.inf)
        ok = np.flatnonzero(rows >= 0)
        for s0 in range(0, len(ok), chunk):
            q = ok[s0:s0 + chunk]
            r = rows[q]
            D, Xr, dr = self.day[r], X[r], dd[r]
            tc, te = np.asarray(t_cut, float)[q][:, None], np.asarray(t_eval, float)[q][:, None]
            m = self.valid[r] & (D < tc)
            age = np.where(m, te - D + 1, 1.0)
            S = np.sum(np.where(m, Xr * age ** (-dr), 0.0), axis=1)
            out[q] = np.where(S > 0, np.log(np.maximum(S, 1e-300)), -np.inf)
        return out

    def attributed_hl_features(self, rows, t_cut, t_submit, min_dose=np.log1p(3)):
        """For the learned half-life on attributed practice: delta (days since submission or a later day with
        dose >= min_dose), c = log1p(total dose before t), d = active days before t."""
        D, X = self.day[np.maximum(rows, 0)], self.dose[np.maximum(rows, 0)]
        P = self.valid[np.maximum(rows, 0)] & ~self.is_sub[np.maximum(rows, 0)]
        tc = np.asarray(t_cut, float)[:, None]
        before = P & (D < tc)
        c = np.log1p(np.sum(np.where(before, X, 0.0), axis=1))
        d = np.sum(before & (X > 0), axis=1).astype(float)
        ts = np.asarray(t_submit, float)[:, None]
        rev = before & (X >= min_dose) & (D > ts)
        last = np.max(np.where(rev, D, -np.inf), axis=1)
        lp = np.fmax(np.asarray(t_submit, float), last)
        delta = np.asarray(t_cut, float) - lp
        bad = rows < 0
        return np.where(bad, 0.0, delta), np.where(bad, 0.0, c), np.where(bad, 0.0, d)


# ------------------------------------------------------------------------------------------------ outcome model
class ActRModel:
    """Outcome model with the learned K combination. Hyper-parameters (decay rule and x_sub) are fixed inputs."""

    def __init__(self, n_topics, interest=False, l2=1e-5):
        self.n_topics, self.interest, self.l2 = n_topics, interest, l2

    def fit_scalers(self, rows, pairs):
        self.mu_ = {c: float(rows[c].mean()) for c in ["ability", "c_inf", "I"]}
        self.sd_ = {c: float(rows[c].std() or 1) for c in ["ability", "c_inf", "I"]}
        a = pairs[pairs["assessed"]]
        self.m_mu, self.m_sd = float(a["m"].mean()), float(a["m"].std() or 1)
        self.pI_mu, self.pI_sd = float(a["I"].mean()), float(a["I"].std() or 1)

    def X(self, rows):
        a = (rows["ability"].to_numpy() - self.mu_["ability"]) / self.sd_["ability"]
        c = (rows["c_inf"].to_numpy() - self.mu_["c_inf"]) / self.sd_["c_inf"]
        cols = [a, c]
        if self.interest:
            i = (rows["I"].to_numpy() - self.mu_["I"]) / self.sd_["I"]
            cols += [i, i * c]
        return np.column_stack(cols)

    def Kfeat(self, pairs):
        M = np.clip(pairs["M"].to_numpy(float), 0.01, 0.99)
        lM = np.log(M / (1 - M))
        mz = (np.where(np.isfinite(pairs["m"]), pairs["m"], self.m_mu) - self.m_mu) / self.m_sd
        cols = [np.ones(len(pairs)), lM, mz]
        if self.interest:
            cols.append((pairs["I"].to_numpy(float) - self.pI_mu) / self.pI_sd)
        return np.column_stack(cols)

    def _layout(self, nX, nB):
        nT = self.n_topics
        return {"beta": slice(0, nT), "theta": slice(nT, nT + nX), "thK": nT + nX,
                "b": slice(nT + nX + 1, nT + nX + 1 + nB)}, nT + nX + 1 + nB

    def _prep(self, rows, pairs):
        return {"topic": rows["topic_idx"].to_numpy(), "X": self.X(rows), "n": len(rows),
                "prow": pairs["row"].to_numpy(np.int64), "w": pairs["w"].to_numpy(float),
                "ass": pairs["assessed"].to_numpy(bool), "F": self.Kfeat(pairs)}

    def K_pairs(self, w, P):
        return np.where(P["ass"], _sig(P["F"] @ w[self.sl_["b"]]), 0.0)

    def _loss_grad(self, w, P, y, sw):
        K = self.K_pairs(w, P)
        Kbar = np.bincount(P["prow"], weights=P["w"] * K, minlength=P["n"])
        eta = w[self.sl_["beta"]][P["topic"]] + P["X"] @ w[self.sl_["theta"]] + w[self.sl_["thK"]] * Kbar
        p = _sig(eta)
        N = sw.sum()
        loss = -np.sum(sw * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))) / N
        r = sw * (p - y) / N
        g = np.zeros_like(w)
        g[self.sl_["beta"]] = np.bincount(P["topic"], weights=r, minlength=self.n_topics)
        g[self.sl_["theta"]] = P["X"].T @ r
        g[self.sl_["thK"]] = Kbar @ r
        dK = np.where(P["ass"], K * (1 - K), 0.0)
        gp = r[P["prow"]] * w[self.sl_["thK"]] * P["w"] * dK
        g[self.sl_["b"]] = P["F"].T @ gp
        pen = np.r_[np.arange(self.sl_["theta"].start, self.sl_["theta"].stop), self.sl_["thK"],
                    np.arange(self.sl_["b"].start + 1, self.sl_["b"].stop)]
        loss += 0.5 * self.l2 * np.sum(w[pen] ** 2)
        g[pen] += self.l2 * w[pen]
        return loss, g

    def fit(self, rows, pairs, y=None, sample_weight=None, init=None):
        self.fit_scalers(rows, pairs)
        P = self._prep(rows, pairs)
        self.sl_, npar = self._layout(P["X"].shape[1], P["F"].shape[1])
        y = rows["y"].to_numpy(float) if y is None else y
        sw = np.ones(len(y)) if sample_weight is None else sample_weight
        if init is None:
            init = np.zeros(npar)
            pb = np.clip(y.mean(), 1e-3, 1 - 1e-3)
            init[self.sl_["beta"]] = np.log(pb / (1 - pb))
            init[self.sl_["thK"]] = 1.0
            init[self.sl_["b"].start + 1] = 1.0
        res = minimize(self._loss_grad, init, args=(P, y, sw), jac=True, method="L-BFGS-B",
                       options={"maxiter": 3000, "gtol": 1e-8})
        self.w_ = res.x
        self.train_loss_ = float(res.fun)
        return self

    def predict(self, rows, pairs):
        P = self._prep(rows, pairs)
        K = self.K_pairs(self.w_, P)
        Kbar = np.bincount(P["prow"], weights=P["w"] * K, minlength=P["n"])
        return _sig(self.w_[self.sl_["beta"]][P["topic"]] + P["X"] @ self.w_[self.sl_["theta"]]
                    + self.w_[self.sl_["thK"]] * Kbar)

    def K_of(self, pairs):
        """K for assessed (student, topic) rows with columns M, m, I (and assessed)."""
        P = {"F": self.Kfeat(pairs), "ass": pairs["assessed"].to_numpy(bool)}
        return self.K_pairs(self.w_, P)

    def coef(self):
        names = ["theta_a", "theta_c"] + (["theta_I", "theta_Ic"] if self.interest else []) + ["theta_K", "b0", "bM",
                                                                                               "bm"] + (["bI"] if self.interest else [])
        return dict(zip(names, self.w_[self.sl_["theta"].start:].round(4).tolist()))


class AttributedDAS3H(SklearnLogit):
    """V4: DAS3H windows plus attributed practice of the prerequisites in each window (pa_w)."""

    def columns(self):
        cols = super().columns()
        return cols + [f"pa_{'inf' if not np.isfinite(w) else int(w)}" for w in self.windows]
