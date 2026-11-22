"""Cleaning steps of METHODOLOGY section 3.2."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation"]


def non_exam_counts(assessments: pd.DataFrame) -> pd.DataFrame:
    """Number of non-exam assessments and of distinct non-exam due dates per module presentation."""
    ne = assessments[assessments["assessment_type"] != "Exam"]
    out = ne.groupby(KEY).agg(n_non_exam=("id_assessment", "size"),
                              n_distinct_dates=("date", "nunique")).reset_index()
    return out


def select_modules(assessments: pd.DataFrame, min_n: int, train: list[str], val: list[str],
                   test: list[str]) -> pd.DataFrame:
    """Audit table: which modules satisfy >= min_n non-exam assessments (and >= min_n distinct dates,
    because same-day assessments are merged into one topic) and run in the split presentations."""
    c = non_exam_counts(assessments)
    c["ok_count"] = c["n_non_exam"] >= min_n
    c["ok_topics"] = c["n_distinct_dates"] >= min_n
    m = c.groupby("code_module").agg(
        presentations=("code_presentation", lambda s: ",".join(sorted(s))),
        all_ok_count=("ok_count", "all"), all_ok_topics=("ok_topics", "all"),
        min_topics=("n_distinct_dates", "min"), max_topics=("n_distinct_dates", "max")).reset_index()
    m["has_train"] = m["presentations"].apply(lambda s: any(p in s for p in train))
    m["has_val"] = m["presentations"].apply(lambda s: any(p in s for p in val))
    m["has_test"] = m["presentations"].apply(lambda s: any(p in s for p in test))
    return m


def fill_exam_dates(assessments: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    a = assessments.merge(courses, on=KEY, how="left")
    miss = a["date"].isna()
    a.loc[miss, "date"] = a.loc[miss, "module_presentation_length"]
    a["date_filled"] = miss
    return a.drop(columns="module_presentation_length")


def drop_assessment_types(assessments: pd.DataFrame, rules: dict | None) -> pd.DataFrame:
    if not rules:
        return assessments
    keep = np.ones(len(assessments), bool)
    for mp, types in rules.items():
        mod, pres = mp.split("_")
        keep &= ~((assessments["code_module"] == mod) & (assessments["code_presentation"] == pres)
                  & assessments["assessment_type"].isin(types)).to_numpy()
    return assessments[keep].copy()


def clean_student_assessment(sa: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop banked rows and submissions without score."""
    log = {"rows_raw": len(sa), "banked": int((sa["is_banked"] == 1).sum())}
    sa = sa[sa["is_banked"] == 0]
    log["missing_score"] = int(sa["score"].isna().sum())
    sa = sa[sa["score"].notna()].copy()
    log["rows_clean"] = len(sa)
    return sa, log


def click_caps(svle: pd.DataFrame, vle: pd.DataFrame, q: float, fit_presentations: list[str]) -> pd.Series:
    """99.5th percentile of daily per-site clicks per activity type, computed on the training
    presentations only (no leakage of later cohorts)."""
    d = svle[svle["code_presentation"].isin(fit_presentations)][["id_site", "sum_click"]]
    d = d.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    return d.groupby("activity_type")["sum_click"].quantile(q)


def apply_click_caps(svle: pd.DataFrame, vle: pd.DataFrame, caps: pd.Series) -> pd.DataFrame:
    at = svle["id_site"].map(vle.set_index("id_site")["activity_type"])
    cap = at.map(caps).fillna(caps.max()).to_numpy()
    out = svle.copy()
    out["clicks"] = np.minimum(out["sum_click"].to_numpy(), cap).astype("float32")
    out["activity_type"] = at.astype("category")
    return out
