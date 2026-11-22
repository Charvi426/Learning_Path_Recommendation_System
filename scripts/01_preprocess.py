"""Stage 1: data audit + preprocessing (METHODOLOGY section 3).

Outputs: data/interim/*.parquet, data/processed/{topics,topic_map,interactions,vle_daily,students,exams}.parquet,
results/tables/audit_*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import clean, load, topics as T  # noqa: E402
from src.data.split import assign_split  # noqa: E402
from src.utils.io import get_logger, load_config, path, write_json, write_parquet, seed_everything  # noqa: E402

KEY = ["code_module", "code_presentation"]
log = get_logger("01_preprocess")


def main(cfg=None):
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    raw = path(cfg, "raw")
    tabs = path(cfg, "tables")
    proc = path(cfg, "processed")
    interim = path(cfg, "interim")

    courses = load.load_courses(raw)
    assess = load.load_assessments(raw)
    sa = load.load_student_assessment(raw)
    vle = load.load_vle(raw)
    reg = load.load_registration(raw)
    info = load.load_info(raw)

    # ---------------- audit -----------------
    audit = {}
    for name, df in [("courses", courses), ("assessments", assess), ("studentAssessment", sa), ("vle", vle),
                     ("studentRegistration", reg), ("studentInfo", info)]:
        audit[name] = {"rows": len(df), "columns": {c: str(t) for c, t in df.dtypes.items()},
                       "missing": {c: int(v) for c, v in df.isna().sum().items() if v}}
    audit["presentations"] = sorted(courses["code_presentation"].unique().tolist())
    audit["activity_types"] = vle["activity_type"].value_counts().to_dict()
    write_json(audit, tabs / "audit_raw_tables.json")
    clean.non_exam_counts(assess).to_csv(tabs / "audit_non_exam_per_presentation.csv", index=False)
    sel = clean.select_modules(assess, cfg["min_non_exam_assessments"], cfg["split"]["train"],
                               cfg["split"]["val"], cfg["split"]["test"])
    sel["selected"] = sel["code_module"].isin(cfg["modules"])
    sel.to_csv(tabs / "audit_module_selection.csv", index=False)
    (vle.assign(has_week=vle["week_from"].notna()).groupby(["code_module", "activity_type"])
        .agg(n_sites=("id_site", "size"), share_week_from=("has_week", "mean")).reset_index()
        .to_csv(tabs / "audit_vle_activity_types.csv", index=False))
    log.info("module selection:\n%s", sel.to_string())

    modules = cfg["modules"]
    # ---------------- cleaning ----------------
    assess_f = clean.fill_exam_dates(assess, courses)
    assess_f = assess_f[assess_f["code_module"].isin(modules)]
    assess_topics = clean.drop_assessment_types(assess_f, cfg.get("drop_assessment_types"))
    vle_m = vle[vle["code_module"].isin(modules)].copy()

    log.info("reading studentVle in chunks ...")
    svle = load.load_student_vle(raw, modules)
    log.info("studentVle rows (selected modules, aggregated per student-site-day): %d", len(svle))
    caps = clean.click_caps(svle, vle_m, cfg["click_cap_quantile"], cfg["split"]["train"])
    caps.rename("cap").reset_index().to_csv(tabs / "audit_click_caps.csv", index=False)
    svle = clean.apply_click_caps(svle, vle_m, caps)
    write_parquet(svle, interim / "student_vle_capped.parquet")

    first_day = svle.groupby(KEY, observed=True)["date"].min()
    topics = T.build_topics(assess_topics, first_day)
    write_parquet(topics, proc / "topics.parquet")
    topics.drop(columns="assessment_ids").to_csv(tabs / "topic_windows.csv", index=False)

    tmap = T.map_sites(vle_m, topics, svle, cfg["unmapped_activity_types"], cfg["voluntary_types"])
    write_parquet(tmap, proc / "topic_map.parquet")
    # share of sites / clicks mapped by each method
    clicks_site = svle.groupby("id_site")["clicks"].sum()
    tmap["cohort_clicks"] = tmap["id_site"].map(clicks_site).fillna(0)
    share = tmap.groupby(["code_module", "map_method"]).agg(n_sites=("id_site", "size"),
                                                            clicks=("cohort_clicks", "sum")).reset_index()
    share["share_sites"] = share["n_sites"] / share.groupby("code_module")["n_sites"].transform("sum")
    share["share_clicks"] = share["clicks"] / share.groupby("code_module")["clicks"].transform("sum")
    share.to_csv(tabs / "topic_map_method_share.csv", index=False)
    log.info("mapping share:\n%s", share.to_string())

    inter, ilog = T.build_interactions(topics, assess_topics, sa, info, reg, cfg["pass_mark"])
    _, salog = clean.clean_student_assessment(sa)
    write_json({"interactions": ilog, "studentAssessment": salog}, tabs / "audit_cleaning_log.json")

    # students table with a unique integer sid per (module, presentation, student)
    stu = info[info["code_module"].isin(modules)].merge(
        reg[KEY + ["id_student", "date_registration", "date_unregistration"]], on=KEY + ["id_student"], how="left")
    stu = stu.sort_values(KEY + ["id_student"]).reset_index(drop=True)
    stu["sid"] = np.arange(len(stu), dtype=np.int32)
    pres_by_mod = stu.groupby("code_module")["code_presentation"].apply(set).to_dict()
    stu["split"] = assign_split(stu, cfg["split"], pres_by_mod)
    stu["prev_attempts"] = stu["num_of_prev_attempts"].clip(upper=cfg["features"]["prev_attempts_cap"])
    exams = T.exam_results(assess_f, sa)
    stu = stu.merge(exams.drop(columns="date_submitted"), on=KEY + ["id_student"], how="left")
    write_parquet(stu, proc / "students.parquet")

    inter = inter.merge(stu[KEY + ["id_student", "sid", "split"]], on=KEY + ["id_student"], how="left")
    write_parquet(inter, proc / "interactions.parquet")

    # vle_daily: student x topic x day
    sv = svle.merge(stu[KEY + ["id_student", "sid"]], on=KEY + ["id_student"], how="inner")
    sv = sv.merge(tmap[["id_site", "topic", "voluntary"]], on="id_site", how="left")
    unm = sv["topic"].isna()
    log.info("clicks on unmapped/navigation sites: %.1f%%", 100 * sv.loc[unm, "clicks"].sum() / sv["clicks"].sum())
    total_daily = sv.groupby(["sid", "date"], as_index=False)["clicks"].sum()
    write_parquet(total_daily, interim / "total_daily_clicks.parquet")
    sv = sv[~unm]
    sv["vol_clicks"] = np.where(sv["voluntary"], sv["clicks"], 0).astype("float32")
    vd = sv.groupby(["sid", "topic", "date"], as_index=False).agg(clicks=("clicks", "sum"),
                                                                  vol_clicks=("vol_clicks", "sum"))
    vd["topic"] = vd["topic"].astype("int16")
    vd = vd.merge(stu[["sid"] + KEY], on="sid", how="left")
    write_parquet(vd, proc / "vle_daily.parquet")
    # sanity check of the topic windows: share of each topic's clicks before / inside / after its window
    w = vd.merge(topics[KEY + ["topic", "t_open", "t_due"]], on=KEY + ["topic"])
    w["inside"] = (w["date"] > w["t_open"]) & (w["date"] <= w["t_due"])
    w["before"], w["after"] = w["date"] <= w["t_open"], w["date"] > w["t_due"]
    sh = w.assign(**{c: w["clicks"] * w[c] for c in ["inside", "before", "after"]}).groupby(
        ["code_module", "topic"])[["inside", "before", "after", "clicks"]].sum()
    sh = sh[["inside", "before", "after"]].div(sh["clicks"], axis=0).round(3).reset_index()
    sh.to_csv(tabs / "topic_click_share_inside_window.csv", index=False)
    split_counts = stu.groupby(["code_module", "code_presentation", "split"]).size().rename("n_students").reset_index()
    split_counts.to_csv(tabs / "split_counts.csv", index=False)
    log.info("splits:\n%s", split_counts.to_string())
    log.info("interactions: %s", ilog)
    log.info("vle_daily rows: %d", len(vd))


if __name__ == "__main__":
    main()
