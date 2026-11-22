"""Windowed engagement (METHODOLOGY 4.2) via a dense student x topic x day cube.

Every query (sid, t) reads only days <= t - 1 (strictly before t). Cumulative arrays have a
leading zero slot so that "clicks in days [a, b]" = cum[idx(b)] - cum[idx(a) - 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ClickCube:
    def __init__(self, vle_daily: pd.DataFrame, n_sid: int, n_topics: int, revision_min_clicks: int = 3,
                 day_mask: str | None = None):
        """day_mask: None (all days), 'odd' or 'even' (used for split-half reliability)."""
        vd = vle_daily
        if day_mask == "odd":
            vd = vd[vd["date"] % 2 != 0]
        elif day_mask == "even":
            vd = vd[vd["date"] % 2 == 0]
        self.dmin = int(vle_daily["date"].min())
        self.dmax = int(vle_daily["date"].max())
        D = self.dmax - self.dmin + 1
        self.D = D
        self.n_topics = n_topics
        s = vd["sid"].to_numpy(np.int64)
        k = vd["topic"].to_numpy(np.int64) - 1
        d = vd["date"].to_numpy(np.int64) - self.dmin + 1  # 1..D
        shape = (n_sid, n_topics, D + 1)
        clicks = np.zeros(shape, np.float32)
        vol = np.zeros(shape, np.float32)
        np.add.at(clicks, (s, k, d), vd["clicks"].to_numpy(np.float32))
        np.add.at(vol, (s, k, d), vd["vol_clicks"].to_numpy(np.float32))
        active = clicks > 0
        rev = clicks >= revision_min_clicks
        dayval = np.arange(D + 1, dtype=np.int16)[None, None, :]
        # last day index (<= current index) with any / >=3 clicks; 0 = none
        self.last_any = np.maximum.accumulate(np.where(active, dayval, 0).astype(np.int16), axis=2)
        self.last_rev = np.maximum.accumulate(np.where(rev, dayval, 0).astype(np.int16), axis=2)
        self.cum_clicks = np.cumsum(clicks, axis=2, dtype=np.float32)
        self.cum_vol = np.cumsum(vol, axis=2, dtype=np.float32)
        self.cum_active = np.cumsum(active, axis=2, dtype=np.int16)
        del clicks, vol, active, rev

    # ---- index helpers -------------------------------------------------
    def idx(self, day) -> np.ndarray:
        """cube index of a calendar day, clipped to [0, D] (0 = before any data)."""
        d = np.asarray(day, np.float64) - self.dmin + 1
        d = np.nan_to_num(d, nan=0.0, posinf=float(self.D), neginf=0.0)
        return np.clip(d, 0, self.D).astype(np.int64)

    def day_of(self, ix: np.ndarray) -> np.ndarray:
        return np.where(ix > 0, ix + self.dmin - 1, np.nan)

    def _range(self, cum: np.ndarray, sid: np.ndarray, a_day, b_day) -> np.ndarray:
        """Sum over days [a_day, b_day] (inclusive) for every topic -> (n, n_topics). Empty if a > b."""
        ib = self.idx(b_day)
        ia = self.idx(np.asarray(a_day) - 1)
        ib = np.broadcast_to(ib, (len(sid), self.n_topics)) if np.ndim(ib) < 2 else ib
        ia = np.broadcast_to(ia, (len(sid), self.n_topics)) if np.ndim(ia) < 2 else ia
        kk = np.broadcast_to(np.arange(self.n_topics)[None, :], ib.shape)
        ss = np.broadcast_to(sid[:, None], ib.shape)
        out = cum[ss, kk, ib].astype(np.float64) - cum[ss, kk, ia].astype(np.float64)
        return np.where(ib >= ia, out, 0.0)

    # ---- public queries (all strictly before t) --------------------------
    def clicks(self, sid, t, w=np.inf, start=None):
        """clicks on each topic in days [max(t - w, start), t - 1]."""
        t = np.asarray(t, np.float64)
        a = t - w if np.isfinite(w) else np.full_like(t, self.dmin - 1)
        a = a[:, None] * np.ones((1, self.n_topics))
        if start is not None:
            a = np.maximum(a, start)
        b = np.minimum((t - 1)[:, None] * np.ones((1, self.n_topics)), np.inf)
        return self._range(self.cum_clicks, sid, a, b)

    def clicks_until(self, sid, t, end_day):
        """clicks on each topic in days <= min(end_day, t - 1); end_day is (n, n_topics)."""
        b = np.minimum((np.asarray(t, np.float64) - 1)[:, None], end_day)
        a = np.full_like(b, self.dmin - 1)
        return self._range(self.cum_clicks, sid, a, b)

    def clicks_after(self, sid, t, start_day):
        """clicks on each topic in days [start_day + 1, t - 1]; start_day is (n, n_topics)."""
        b = (np.asarray(t, np.float64) - 1)[:, None] * np.ones((1, self.n_topics))
        return self._range(self.cum_clicks, sid, start_day + 1, b)

    def vol_clicks(self, sid, t):
        t = np.asarray(t, np.float64)
        a = np.full((len(t), self.n_topics), self.dmin - 1.0)
        return self._range(self.cum_vol, sid, a, (t - 1)[:, None] * np.ones((1, self.n_topics)))

    def active_days(self, sid, t, w=np.inf):
        t = np.asarray(t, np.float64)
        a = (t - w if np.isfinite(w) else np.full_like(t, self.dmin - 1))[:, None] * np.ones((1, self.n_topics))
        return self._range(self.cum_active, sid, a, (t - 1)[:, None] * np.ones((1, self.n_topics)))

    def _last(self, arr, sid, t):
        it = self.idx(np.asarray(t, np.float64) - 1)
        kk = np.arange(self.n_topics)[None, :]
        ix = arr[sid[:, None], kk, it[:, None]].astype(np.int64)
        return self.day_of(ix)

    def last_click_day(self, sid, t):
        return self._last(self.last_any, sid, t)

    def last_revision_day(self, sid, t):
        return self._last(self.last_rev, sid, t)
