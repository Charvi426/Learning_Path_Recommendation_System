"""Assemble queries (student, time) and the per-topic knowledge-relevant state at those times.

Long "state" table: one row per (query, topic j) with every factor of METHODOLOGY section 4,
computed from events strictly before the query time t:
    c_w, d_w           clicks / active days on topic j in window w
    vol, early, post   interest raw signals (log1p)
    assessed, M, y_j   result of j if submitted before t
    delta              days since last meaningful practice of j (submission or later day with >=3 clicks)
    dt_last            days since last click on j or its submission (Delta t_{s,k} in 4.2)
    ability            shrunk mean y over topics assessed before t
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engagement import ClickCube

KEY = ["code_module", "code_presentation"]


def wname(w) -> str:
    return "inf" if not np.isfinite(w) else str(int(w))


class TopicArrays:
    """Dense (n_sid, n_topics) arrays of topic timing and outcomes per student."""

    def __init__(self, inter: pd.DataFrame, topics: pd.DataFrame, students: pd.DataFrame, n_topics: int):
        n = int(students["sid"].max()) + 1
        self.n_sid, self.n_topics = n, n_topics
        self.t_open = np.full((n, n_topics), np.nan)
        self.t_due = np.full((n, n_topics), np.nan)
        st = students[["sid"] + KEY].merge(topics[KEY + ["topic", "t_open", "t_due"]], on=KEY)
        self.t_open[st["sid"], st["topic"] - 1] = st["t_open"]
        self.t_due[st["sid"], st["topic"] - 1] = st["t_due"]
        self.registered = np.zeros((n, n_topics), bool)
        self.submitted = np.zeros((n, n_topics), bool)
        self.t_submit = np.full((n, n_topics), np.nan)
        self.y = np.full((n, n_topics), np.nan)
        self.score = np.full((n, n_topics), np.nan)
        s, k = inter["sid"].to_numpy(), inter["topic"].to_numpy() - 1
        self.registered[s, k] = True
        sub = inter["is_submitted"].to_numpy() == 1
        self.submitted[s[sub], k[sub]] = True
        self.t_submit[s[sub], k[sub]] = inter["t_submit"].to_numpy()[sub]
        self.y[s[sub], k[sub]] = inter["y"].to_numpy()[sub]
        self.score[s[sub], k[sub]] = inter["y_score"].to_numpy()[sub]
        self.module = students.set_index("sid")["code_module"].reindex(range(n)).to_numpy()
        self.prev_attempts = students.set_index("sid")["prev_attempts"].reindex(range(n)).fillna(0).to_numpy()


def make_queries(inter: pd.DataFrame, students: pd.DataFrame, ta: TopicArrays) -> pd.DataFrame:
    """train-type rows (one per student-topic at t = t_due), decision points (day after submitting k-1),
    and exam queries (t = exam date)."""
    q1 = inter[["sid", "topic", "t_due"]].rename(columns={"topic": "k", "t_due": "t"}).assign(kind="due")
    # decision points: k = 2..n, t = t_submit_{k-1} + 1, k registered, t <= t_due_k
    prev = inter[inter["is_submitted"] == 1][["sid", "topic", "t_submit"]].copy()
    prev["k"] = prev["topic"] + 1
    dp = prev.merge(inter[["sid", "topic", "t_due"]].rename(columns={"topic": "k"}), on=["sid", "k"], how="inner")
    dp["t"] = dp["t_submit"] + 1
    dp = dp[dp["t"] <= dp["t_due"]][["sid", "k", "t"]].assign(kind="decision")
    ex = students[students["exam_score"].notna()][["sid", "t_exam"]].rename(columns={"t_exam": "t"})
    ex = ex.assign(k=0, kind="exam")
    q = pd.concat([q1, dp, ex], ignore_index=True)
    q = q.merge(students[["sid", "split"] + KEY], on="sid", how="left")
    q["t"] = q["t"].astype(float)
    q = q.sort_values(["kind", "sid", "k"]).reset_index(drop=True)
    q["qid"] = np.arange(len(q), dtype=np.int64)
    return q[["qid", "sid", "k", "t", "kind", "split"] + KEY]


def compute_state(cube: ClickCube, ta: TopicArrays, queries: pd.DataFrame, windows: list,
                  cohort_mean_y: dict, ability_shrink: float = 2.0) -> pd.DataFrame:
    sid = queries["sid"].to_numpy(np.int64)
    t = queries["t"].to_numpy(np.float64)
    n, K = len(sid), ta.n_topics
    out = {}
    for w in windows:
        c = cube.clicks(sid, t, w)
        out[f"c_{wname(w)}"] = np.log1p(c)
        out[f"d_{wname(w)}"] = cube.active_days(sid, t, w)
    out["clicks_raw_inf"] = cube.clicks(sid, t, np.inf)
    t_open = ta.t_open[sid]
    t_sub = ta.t_submit[sid]
    tt = t[:, None]
    assessed = ta.submitted[sid] & (t_sub < tt)
    out["vol"] = np.log1p(cube.vol_clicks(sid, t))
    out["early"] = np.log1p(cube.clicks_until(sid, t, t_open))
    post = cube.clicks_after(sid, t, np.where(assessed, t_sub, np.inf))
    out["post"] = np.log1p(np.where(assessed, post, 0.0))
    out["opened"] = (t_open < tt) | (out["clicks_raw_inf"] > 0)
    out["window_open"] = t_open < tt
    out["assessed"] = assessed
    out["M"] = np.where(assessed, ta.score[sid], np.nan)
    out["y_j"] = np.where(assessed, ta.y[sid], np.nan)
    out["t_submit_j"] = np.where(assessed, t_sub, np.nan)
    out["registered"] = ta.registered[sid]
    last_rev = cube.last_revision_day(sid, t)
    last_click = cube.last_click_day(sid, t)
    lp = np.where(assessed, np.fmax(t_sub, np.where(last_rev > t_sub, last_rev, -np.inf)), np.nan)
    out["delta"] = tt - lp
    lc = np.fmax(last_click, np.where(assessed, t_sub, np.nan))
    out["dt_last"] = tt - lc
    out["revised"] = assessed & (last_rev > t_sub)
    # student-level
    yv = np.where(assessed, ta.y[sid], 0.0)
    n_ass = assessed.sum(1)
    mu = np.array([cohort_mean_y.get(m, np.nan) for m in ta.module[sid]])
    ability = (yv.sum(1) + ability_shrink * mu) / (n_ass + ability_shrink)
    stage = (t_open < tt).sum(1)

    df = pd.DataFrame({k: v.reshape(-1) for k, v in out.items()})
    df.insert(0, "j", np.tile(np.arange(1, K + 1), n))
    df.insert(0, "qid", np.repeat(queries["qid"].to_numpy(), K))
    df["ability"] = np.repeat(ability, K)
    df["n_assessed"] = np.repeat(n_ass, K)
    df["stage"] = np.repeat(np.maximum(stage, 1), K)
    df["prev_attempts"] = np.repeat(ta.prev_attempts[sid], K)
    df["code_module"] = np.repeat(ta.module[sid], K)
    return df


def total_clicks_before(total_daily: pd.DataFrame, queries: pd.DataFrame) -> np.ndarray:
    """log1p of all clicks (incl. navigation) of the student before t (for validity checks)."""
    td = total_daily.sort_values(["sid", "date"])
    td["cum"] = td.groupby("sid")["clicks"].cumsum()
    q = queries[["qid", "sid", "t"]].copy()
    q["tm1"] = q["t"] - 1
    q = q.sort_values("tm1")
    td = td.sort_values("date")
    m = pd.merge_asof(q, td[["sid", "date", "cum"]].rename(columns={"date": "tm1"}).astype({"tm1": float}),
                      on="tm1", by="sid", direction="backward")
    return np.log1p(m.set_index("qid").reindex(queries["qid"])["cum"].fillna(0).to_numpy())
