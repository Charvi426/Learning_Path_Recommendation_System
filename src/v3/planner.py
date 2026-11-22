"""v3 Step 4 (post-hoc): pre-exam revision planner and its evaluation.

Weekly decision points at exam - 28, - 21, - 14, - 7 days for every exam-taker. At each point, for every topic
assessed (submitted) before the point: K(s, j, exam date) assuming no further practice (memory cut-off = decision
day, evaluated at the exam date); priority = M_sj - K(s, j, exam). The plan is the top 3 topics with priority > 0.

Observational check (NOT causal): OLS of exam score on mean M (at the exam), log1p(total clicks in the 28 days
before the exam, all resources) and the share of the student's attributed revision practice (same 28 days,
assessed topics) that went to the topics flagged at the first decision point; bootstrap CI over students.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import Variant

WEEKS = [28, 21, 14, 7]


def plans(var: Variant, data, split, n_top=3, scale="raw"):
    stu = data["students"]
    ex = stu[(stu["split"] == split) & stu["exam_score"].notna()][["sid", "code_module", "code_presentation", "t_exam"]]
    inter = data["inter"]
    rows = []
    for w in WEEKS:
        e = ex.assign(t=ex["t_exam"] - w, weeks_before=w // 7)
        it = inter[inter["is_submitted"] == 1][["sid", "topic", "score", "t_submit"]].merge(e, on="sid")
        it = it[it["t_submit"] < it["t"]]
        rows.append(it)
    f = pd.concat(rows, ignore_index=True).rename(columns={"topic": "j", "t_submit": "t_submit_j"})
    f["assessed"] = True
    f["M"] = f["score"] / 100.0
    f["I"] = 0.5                                       # interest is not used by the planner (A5-type model)
    for c in ["c_inf", "d_inf", "delta"]:
        f[c] = 0.0
    if var.kind in ("hl_v2",):
        raise NotImplementedError("planner needs a history-based variant (state at arbitrary days)")
    K, R = var.K_assessed(f, f["t"].to_numpy(float), f["t_exam"].to_numpy(float))
    f["K_exam"] = K if scale == "raw" else f["M"].to_numpy(float) * R   # 'score': M * R(exam)
    f["priority"] = f["M"] - f["K_exam"]
    f = f.sort_values(["sid", "weeks_before", "priority"], ascending=[True, False, False])
    f["rank"] = f.groupby(["sid", "weeks_before"]).cumcount() + 1
    f["flagged"] = (f["rank"] <= n_top) & (f["priority"] > 0)
    f["explanation"] = np.where(
        f["flagged"],
        "Revise " + f["code_module"] + " topic " + f["j"].astype(str) + ": you scored "
        + (100 * f["M"]).round(0).astype(int).astype(str) + "%, but estimated knowledge at the exam is "
        + (100 * f["K_exam"]).round(0).astype(int).astype(str) + "%.", "")
    return f


def observational_check(plan: pd.DataFrame, practice: pd.DataFrame, total_daily: pd.DataFrame, mean_M: pd.DataFrame,
                        students: pd.DataFrame, nb=1000, seed=0):
    first = plan[plan["weeks_before"] == 4]
    flagged = first[first["flagged"]][["sid", "j"]].assign(flag=True)
    assessed = first[["sid", "j", "t", "t_exam"]]
    pr = practice.rename(columns={"topic": "j"}).merge(assessed, on=["sid", "j"])
    pr = pr[(pr["date"] >= pr["t"]) & (pr["date"] < pr["t_exam"])]
    pr = pr.merge(flagged, on=["sid", "j"], how="left")
    pr["flag"] = pr["flag"].fillna(False).astype(bool)
    g = pr.groupby("sid").apply(lambda x: pd.Series({"rev_dose": x["dose"].sum(),
                                                      "flag_dose": x.loc[x["flag"], "dose"].sum()}),
                                include_groups=False).reset_index()
    g["share_flagged"] = g["flag_dose"] / g["rev_dose"].where(g["rev_dose"] > 0)
    ex = students[["sid", "exam_score", "t_exam"]].dropna()
    td = total_daily.merge(ex[["sid", "t_exam"]], on="sid")
    td = td[(td["date"] >= td["t_exam"] - 28) & (td["date"] < td["t_exam"])]
    clicks = td.groupby("sid")["clicks"].sum().rename("rev_clicks").reset_index()
    d = ex[ex["sid"].isin(first["sid"])].merge(mean_M, on="sid").merge(clicks, on="sid", how="left").merge(
        g[["sid", "share_flagged"]], on="sid", how="left")
    d["rev_clicks"] = d["rev_clicks"].fillna(0)
    n_all = len(d)
    d = d[d["share_flagged"].notna()]
    X = np.column_stack([np.ones(len(d)), d["mean_M"], np.log1p(d["rev_clicks"]), d["share_flagged"]])
    y = d["exam_score"].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nb):
        i = rng.integers(0, len(d), len(d))
        bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    bs = np.array(bs)
    names = ["intercept", "mean_M", "log1p_revision_clicks", "share_revision_on_flagged"]
    out = pd.DataFrame({"term": names, "coef": beta, "ci_lo": np.percentile(bs, 2.5, axis=0),
                        "ci_hi": np.percentile(bs, 97.5, axis=0)})
    out["n_students"] = len(d)
    out["n_excluded_no_revision_practice"] = n_all - len(d)
    out["label"] = "observational, not causal"
    return out
