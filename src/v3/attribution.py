"""v3 Steps 1-2 (post-hoc): resource classes and attribution of practice to topics.

Step 1  NAVIGATION = homepage, plus plumbing activity types (forumng, subpage, glossary) whose cohort clicks are
        spread evenly over the course weeks (normalised weekly entropy >= 0.80). Everything else is CONTENT.
        Navigation clicks never count as practice of any topic.
Step 2  topic weights w(site, k) >= 0, sum 1, for CONTENT sites, from cohort clicks made BEFORE each topic's
        assessment (clicks in topic k's window (t_open_k, t_due_k]); revision-period clicks (after the last due
        date) are never used.
          direct     weight 1 on the topic the site is mapped to (v1/v2 behaviour)
          coaccess   w(site, k) proportional to cohort window-k clicks on the site
          shared     sites whose co-access distribution has normalised entropy > H, every final-window site and
                     every exam-preparation site (> 50% of its cohort clicks after the last due date) are spread
                     evenly, at click day d, over the topics assessed before d plus the topic whose window
                     contains d; all other sites use coaccess.
        Site ids are presentation-specific in OULAD, so the "cohort" is the site's own presentation (aggregate
        clicks of all its students; no outcomes), as for the v1 site -> topic mapping (DECISIONS #12).
Practice dose of student s on topic j, day d = sum over content sites of w(site, j[, d]) * log1p(clicks).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation"]
PLUMBING = ["forumng", "subpage", "glossary"]
ALWAYS_NAV = ["homepage"]
NAV_ENTROPY = 0.80


def _norm_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def classify_resources(svle: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    """svle: site-level capped clicks (code_module, code_presentation, id_site, date, clicks)."""
    s = svle[KEY + ["id_site", "date", "clicks"]].copy()
    for c in KEY:
        s[c] = s[c].astype(str)
    first = s.groupby(KEY)["date"].transform("min")
    s["week"] = (s["date"] - first) // 7
    nweeks = s.groupby(KEY)["week"].max().add(1).rename("n_weeks")
    w = s.groupby(KEY + ["id_site", "week"])["clicks"].sum().reset_index()
    ent = w.groupby(KEY + ["id_site"])["clicks"].apply(lambda x: _norm_entropy(x.to_numpy())).rename("H_weeks")
    ent = ent.reset_index().merge(nweeks.reset_index(), on=KEY)
    ent["H_weeks_norm"] = ent["H_weeks"] / np.log(ent["n_weeks"].clip(lower=2))
    tot = s.groupby("id_site")["clicks"].sum().rename("cohort_clicks")
    out = vle[["id_site", "code_module", "code_presentation", "activity_type"]].merge(
        ent[["id_site", "H_weeks_norm"]], on="id_site", how="left").merge(tot, on="id_site", how="left")
    out = out[out["code_module"].isin(s["code_module"].unique())]
    out["H_weeks_norm"] = out["H_weeks_norm"].fillna(0.0)
    out["cohort_clicks"] = out["cohort_clicks"].fillna(0.0)
    nav = out["activity_type"].isin(ALWAYS_NAV) | (out["activity_type"].isin(PLUMBING)
                                                    & (out["H_weeks_norm"] >= NAV_ENTROPY))
    out["resource_class"] = np.where(nav, "NAVIGATION", "CONTENT")
    out["rule"] = np.where(out["activity_type"].isin(ALWAYS_NAV), "homepage",
                           np.where(nav, f"plumbing type & weekly entropy >= {NAV_ENTROPY}", "content"))
    return out


def window_topic(day: np.ndarray, t_open: np.ndarray, t_due: np.ndarray) -> np.ndarray:
    """1-based topic whose window (t_open, t_due] contains day; 0 if after the last due date."""
    k = np.searchsorted(t_due, day, side="left") + 1
    return np.where(day > t_due[-1], 0, np.clip(k, 1, len(t_due)))


def site_weights(svle: pd.DataFrame, topics: pd.DataFrame, tmap: pd.DataFrame, classes: pd.DataFrame,
                 mode: str, H: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (static weights: id_site, topic, w) and the list of shared sites (id_site, module, pres).
    mode in {'direct', 'coaccess', 'shared'}."""
    content = classes[classes["resource_class"] == "CONTENT"][["id_site", "code_module", "code_presentation"]]
    tm = tmap[["id_site", "topic"]].dropna()
    direct = content.merge(tm, on="id_site")[["id_site", "topic"]].assign(w=1.0)
    direct["topic"] = direct["topic"].astype(int)
    if mode == "direct":
        return direct, content.iloc[0:0]
    # cohort window clicks per (site, topic): only clicks inside topic windows (pre-assessment)
    s = svle[KEY + ["id_site", "date", "clicks"]].copy()
    for c in KEY:
        s[c] = s[c].astype(str)
    s = s[s["id_site"].isin(content["id_site"])]
    sd = s.groupby(KEY + ["id_site", "date"])["clicks"].sum().reset_index()
    rows, extra = [], []
    for (m, p), g in sd.groupby(KEY):
        tp = topics[(topics["code_module"] == m) & (topics["code_presentation"] == p)].sort_values("topic")
        t_open, t_due = tp["t_open"].to_numpy(), tp["t_due"].to_numpy()
        g = g.copy()
        g["k"] = window_topic(g["date"].to_numpy(), t_open, t_due)
        win = g[g["k"] > 0].groupby(["id_site", "k"])["clicks"].sum().reset_index()
        rows.append(win.assign(code_module=m, code_presentation=p, n_topics=len(tp)))
        after = g.assign(after=g["k"] == 0).groupby("id_site").apply(
            lambda x: (x["clicks"] * x["after"]).sum() / x["clicks"].sum(), include_groups=False)
        extra.append(pd.DataFrame({"id_site": after.index, "share_after_last_due": after.to_numpy(),
                                   "last_topic": len(tp)}))
    win = pd.concat(rows, ignore_index=True)
    win["w"] = win["clicks"] / win.groupby("id_site")["clicks"].transform("sum")
    co = win.rename(columns={"k": "topic"})[["id_site", "topic", "w"]]
    # sites with no window clicks fall back to their direct mapping
    missing = direct[~direct["id_site"].isin(co["id_site"])]
    co = pd.concat([co, missing], ignore_index=True)
    if mode == "coaccess":
        return co, content.iloc[0:0]
    ent = win.groupby("id_site").apply(
        lambda x: _norm_entropy(x["w"].to_numpy()) / np.log(max(int(x["n_topics"].iloc[0]), 2)),
        include_groups=False).rename("H_topics")
    ex = pd.concat(extra, ignore_index=True)
    info = content.merge(ent.reset_index(), on="id_site", how="left").merge(ex, on="id_site", how="left").merge(
        tm, on="id_site", how="left")
    final_window = info["topic"].notna() & (info["topic"] == info["last_topic"])
    exam_prep = info["share_after_last_due"].fillna(0) > 0.5
    spread = info["H_topics"].fillna(0) > H
    shared = info[final_window | exam_prep | spread][["id_site", "code_module", "code_presentation"]]
    static = co[~co["id_site"].isin(shared["id_site"])]
    return static, shared


def eligible_matrix(tp: pd.DataFrame, dmin: int, dmax: int) -> np.ndarray:
    """(days, topics) uniform weights for shared sites: topics assessed (due) before day d plus the topic whose
    window contains d."""
    tp = tp.sort_values("topic")
    t_open, t_due = tp["t_open"].to_numpy(), tp["t_due"].to_numpy()
    days = np.arange(dmin, dmax + 1)
    el = (t_due[None, :] < days[:, None]) | ((t_open[None, :] < days[:, None]) & (days[:, None] <= t_due[None, :]))
    el[~el.any(1), 0] = True
    return el / el.sum(1, keepdims=True)


def practice_doses(svle: pd.DataFrame, students: pd.DataFrame, topics: pd.DataFrame, static: pd.DataFrame,
                   shared: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """Daily attributed practice per (sid, topic j, day): sum_site w(site, j[, d]) * log1p(clicks)."""
    s = svle[KEY + ["id_student", "id_site", "date", "clicks"]].copy()
    for c in KEY:
        s[c] = s[c].astype(str)
    s = s.merge(students[KEY + ["id_student", "sid"]], on=KEY + ["id_student"], how="inner")
    content_ids = set(classes.loc[classes["resource_class"] == "CONTENT", "id_site"])
    s = s[s["id_site"].isin(content_ids)]
    s["lc"] = np.log1p(s["clicks"].to_numpy(np.float64))
    parts = []
    st = s.merge(static, on="id_site", how="inner")
    st["dose"] = st["w"] * st["lc"]
    parts.append(st.groupby(["sid", "topic", "date"], as_index=False)["dose"].sum())
    if len(shared):
        sh = s[s["id_site"].isin(shared["id_site"])].groupby(KEY + ["sid", "date"], as_index=False)["lc"].sum()
        for (m, p), g in sh.groupby(KEY):
            tp = topics[(topics["code_module"] == m) & (topics["code_presentation"] == p)]
            dmin, dmax = int(g["date"].min()), int(g["date"].max())
            U = eligible_matrix(tp, dmin, dmax)
            W = U[g["date"].to_numpy() - dmin]                      # (rows, topics)
            dose = W * g["lc"].to_numpy()[:, None]
            r, k = np.nonzero(dose > 0)
            parts.append(pd.DataFrame({"sid": g["sid"].to_numpy()[r], "topic": k + 1,
                                       "date": g["date"].to_numpy()[r], "dose": dose[r, k]}))
    out = pd.concat(parts, ignore_index=True).groupby(["sid", "topic", "date"], as_index=False)["dose"].sum()
    out["topic"] = out["topic"].astype(int)
    return out
