"""Topic windows (METHODOLOGY 3.3), id_site -> topic mapping and the interaction table (3.4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation"]


def build_topics(assess: pd.DataFrame, first_day: pd.Series) -> pd.DataFrame:
    """One row per topic. Non-exam assessments ordered by date; same-day assessments merged.

    Topic k covers the window (t_open, t_due] with t_open = t_{k-1}; for k = 1 we use
    t_open = first VLE day - 1 so that the first day of activity belongs to topic 1.
    """
    ne = assess[assess["assessment_type"] != "Exam"].copy()
    rows = []
    for (m, p), g in ne.groupby(KEY):
        dates = np.sort(g["date"].unique())
        t0 = float(first_day.get((m, p), 0)) - 1
        prev = t0
        for k, d in enumerate(dates, start=1):
            ga = g[g["date"] == d]
            rows.append({"code_module": m, "code_presentation": p, "topic": k,
                         "t_open": prev, "t_due": float(d),
                         "assessment_ids": list(ga["id_assessment"].astype(int)),
                         "assessment_types": "+".join(sorted(ga["assessment_type"].unique())),
                         "n_assessments": len(ga), "weight": float(ga["weight"].sum())})
            prev = float(d)
    t = pd.DataFrame(rows)
    t["n_topics"] = t.groupby(KEY)["topic"].transform("max")
    return t


def topic_of_day(day: np.ndarray, t_open: np.ndarray, t_due: np.ndarray) -> np.ndarray:
    """Topic index (1-based) whose window (t_open, t_due] contains day; days after the last
    due date go to the last topic, days before the first window to topic 1."""
    k = np.searchsorted(t_due, day, side="left") + 1  # first k with t_due >= day
    return np.clip(k, 1, len(t_due))


def map_sites(vle: pd.DataFrame, topics: pd.DataFrame, svle: pd.DataFrame,
              unmapped_types: list[str], voluntary_types: list[str]) -> pd.DataFrame:
    """id_site -> topic. week_from rule first; otherwise the window with most cohort clicks."""
    out = []
    for (m, p), tp in topics.groupby(KEY):
        tp = tp.sort_values("topic")
        t_open, t_due = tp["t_open"].to_numpy(), tp["t_due"].to_numpy()
        v = vle[(vle["code_module"] == m) & (vle["code_presentation"] == p)].copy()
        s = svle[(svle["code_module"] == m) & (svle["code_presentation"] == p)]
        # cohort clicks per site and window (clicks after the last due date are not counted
        # for the argmax unless the site has no other clicks)
        s = s[["id_site", "date", "clicks"]].copy()
        s["k"] = topic_of_day(s["date"].to_numpy(), t_open, t_due)
        s["in_window"] = s["date"].to_numpy() <= t_due[-1]
        cw = s[s["in_window"]].groupby(["id_site", "k"])["clicks"].sum().reset_index()
        best = cw.sort_values(["id_site", "clicks"], ascending=[True, False]).drop_duplicates("id_site")
        best_any = (s.groupby(["id_site", "k"])["clicks"].sum().reset_index()
                    .sort_values(["id_site", "clicks"], ascending=[True, False]).drop_duplicates("id_site"))
        v["topic_week"] = np.where(v["week_from"].notna(),
                                   topic_of_day(7 * v["week_from"].fillna(0).to_numpy(), t_open, t_due), 0)
        v = v.merge(best[["id_site", "k"]].rename(columns={"k": "topic_click"}), on="id_site", how="left")
        v = v.merge(best_any[["id_site", "k"]].rename(columns={"k": "topic_click_any"}), on="id_site", how="left")
        v["topic_click"] = v["topic_click"].fillna(v["topic_click_any"])
        v["topic"] = np.where(v["week_from"].notna(), v["topic_week"], v["topic_click"])
        v["map_method"] = np.where(v["week_from"].notna(), "week_from",
                                   np.where(v["topic_click"].notna(), "cohort_clicks", "unmapped_no_clicks"))
        nav = v["activity_type"].isin(unmapped_types)
        v.loc[nav, "topic"] = np.nan
        v.loc[nav, "map_method"] = "excluded_navigation"
        out.append(v)
    tm = pd.concat(out, ignore_index=True)
    tm["voluntary"] = tm["activity_type"].isin(voluntary_types)
    tm["topic"] = tm["topic"].astype("Int16")
    return tm[["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to",
               "topic", "map_method", "voluntary"]]


def build_interactions(topics: pd.DataFrame, assess: pd.DataFrame, sa_raw: pd.DataFrame,
                       info: pd.DataFrame, reg: pd.DataFrame, pass_mark: float) -> tuple[pd.DataFrame, dict]:
    """Student x topic table with t_open, t_due, t_submit, score, y, is_submitted."""
    log = {}
    ta = topics[KEY + ["topic", "assessment_ids", "t_open", "t_due"]].explode("assessment_ids")
    ta = ta.rename(columns={"assessment_ids": "id_assessment"})
    ta["id_assessment"] = ta["id_assessment"].astype("int64")
    ta = ta.merge(assess[["id_assessment", "weight"]], on="id_assessment", how="left")

    sa = sa_raw.merge(ta[["id_assessment"] + KEY + ["topic", "weight"]], on="id_assessment", how="inner")
    banked = sa[sa["is_banked"] == 1][KEY + ["id_student", "topic"]].drop_duplicates()
    noscore = sa[(sa["is_banked"] == 0) & sa["score"].isna()][KEY + ["id_student", "topic"]].drop_duplicates()
    sa = sa[(sa["is_banked"] == 0) & sa["score"].notna()].copy()
    sa["w"] = sa["weight"]
    sa["ws"] = sa["w"] * sa["score"]
    agg = sa.groupby(KEY + ["id_student", "topic"]).agg(
        ws=("ws", "sum"), w=("w", "sum"), mean_score=("score", "mean"),
        t_submit=("date_submitted", "max"), t_first_submit=("date_submitted", "min"),
        n_parts=("score", "size")).reset_index()
    agg["score"] = np.where(agg["w"] > 0, agg["ws"] / agg["w"].where(agg["w"] > 0, 1), agg["mean_score"])
    agg = agg.drop(columns=["ws", "w", "mean_score"])

    stu = info[KEY + ["id_student", "num_of_prev_attempts", "final_result"]].merge(
        reg[KEY + ["id_student", "date_registration", "date_unregistration"]], on=KEY + ["id_student"], how="left")
    stu = stu[stu["code_module"].isin(topics["code_module"].unique())]
    base = stu.merge(topics[KEY + ["topic", "t_open", "t_due", "n_topics"]], on=KEY, how="inner")
    log["rows_all_student_topic"] = len(base)
    # withdrawals: keep topics due before unregistration
    wd = base["date_unregistration"].notna() & ~(base["t_due"] < base["date_unregistration"])
    log["dropped_after_withdrawal"] = int(wd.sum())
    base = base[~wd]
    # banked topic rows / submissions with no score are removed entirely
    for name, drop in [("dropped_banked", banked), ("dropped_no_score", noscore)]:
        mk = base.merge(drop.assign(_d=1), on=KEY + ["id_student", "topic"], how="left")["_d"].notna().to_numpy()
        log[name] = int(mk.sum())
        base = base[~mk]
    df = base.merge(agg, on=KEY + ["id_student", "topic"], how="left")
    df["is_submitted"] = df["score"].notna().astype("int8")
    df["y"] = np.where(df["is_submitted"] == 1, (df["score"] >= pass_mark).astype(float), np.nan)
    df["y_score"] = df["score"] / 100.0
    log["rows_final"] = len(df)
    log["share_submitted"] = float(df["is_submitted"].mean())
    log["pass_rate_submitted"] = float(df["y"].mean())
    return df, log


def exam_results(assess: pd.DataFrame, sa_raw: pd.DataFrame) -> pd.DataFrame:
    ex = assess[assess["assessment_type"] == "Exam"][KEY + ["id_assessment", "date"]]
    r = sa_raw[(sa_raw["is_banked"] == 0) & sa_raw["score"].notna()].merge(ex, on="id_assessment")
    # CCC has two exams; take the last-dated exam per student (not used for DDD/FFF)
    r = r.sort_values("date").groupby(KEY + ["id_student"]).tail(1)
    return r.rename(columns={"date": "t_exam", "score": "exam_score"})[
        KEY + ["id_student", "t_exam", "exam_score", "date_submitted"]]
