"""Read-only audit of defects found while preparing the report (2026-11-22).

Nothing is fitted, tuned or scored and nothing under results/ or config/ is written. The script counts and dates
things in the stored intermediate data and writes report/audit/*.csv.

  python report/audit_defects.py      (needs data/raw and the intermediates made by demo/build_demo_data.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.v3 import attribution as A  # noqa: E402
from src.v3.data import HOLDOUT, classes, load  # noqa: E402

OUT = ROOT / "report" / "audit"
SETS = ["DDDFFF", *HOLDOUT]


def label_in_ability(sets: dict):
    """Defect D1: at t = t_due_k, ability averages y over topics submitted before t, which includes k itself."""
    rows, first = [], []
    for cs, d in sets.items():
        for split, st in d["state"].items():
            if st.empty:
                continue
            q = d["queries"]
            r = q[(q["kind"] == "due") & (q["split"] == split)].merge(
                st[["qid", "j", "assessed", "ability"]], left_on=["qid", "k"], right_on=["qid", "j"])
            r = r.merge(d["inter"][["sid", "topic", "y"]].rename(columns={"topic": "k"}), on=["sid", "k"])
            r = r[r["y"].notna()]
            for m, g in r.groupby("code_module"):
                rows.append({"module": m, "split": split, "n_rows": len(g), "n_label_in_feature": int(g["assessed"].sum()),
                             "share_label_in_feature": float(g["assessed"].mean())})
                g1 = g[g["k"] == 1]
                if g1["y"].nunique() == 2:
                    first.append({"module": m, "split": split, "n_rows": len(g1),
                                  "share_label_in_feature": float(g1["assessed"].mean()),
                                  "auc_of_stored_ability_alone": roc_auc_score(g1["y"], g1["ability"])})
            rows.append({"module": cs + " (all)", "split": split, "n_rows": len(r),
                         "n_label_in_feature": int(r["assessed"].sum()),
                         "share_label_in_feature": float(r["assessed"].mean())})
    return pd.DataFrame(rows), pd.DataFrame(first)


def exam_days():
    """Defect D2: the exam day used was the presentation's last day, not the day the exam was sat."""
    raw = ROOT / "data" / "raw"
    a = pd.read_csv(raw / "assessments.csv")
    sa = pd.read_csv(raw / "studentAssessment.csv")
    e = a[a["assessment_type"] == "Exam"].merge(pd.read_csv(raw / "courses.csv"))
    e["day_used"] = e["date"].fillna(e["module_presentation_length"])
    x = e.merge(sa, on="id_assessment")
    x = x[(x["is_banked"] == 0) & x["score"].notna()]
    x["days_late"] = x["day_used"] - x["date_submitted"]
    rows = []
    for (m, p), g in x.groupby(["code_module", "code_presentation"]):
        r = {"module": m, "presentation": p, "n_exam_takers": len(g), "day_used": g["day_used"].iloc[0],
             "median_day_sat": g["date_submitted"].median(), "median_days_late": g["days_late"].median()}
        for w in (4, 3, 2, 1):
            r[f"share_week{w}_point_on_or_after_exam"] = float((g["day_used"] - 7 * w >= g["date_submitted"]).mean())
        r["planner_points_after_median_exam"] = int(
            sum(r["day_used"] - 7 * w >= r["median_day_sat"] for w in (4, 3, 2, 1)))
        rows.append(r)
    return pd.DataFrame(rows)


def attribution_uses_revision_clicks(sets: dict):
    """Defect D3: v3 logged that attribution never uses clicks after the last due date; the shared rule does."""
    rows = []
    for cs, d in sets.items():
        sv = pd.read_parquet(d["dirs"]["svle"], columns=["code_module", "code_presentation", "id_site", "date", "clicks"])
        rc = classes(cs, d)
        last = d["topics"].groupby(["code_module", "code_presentation"])["t_due"].max()
        key = pd.MultiIndex.from_arrays([sv["code_module"].astype(str), sv["code_presentation"].astype(str)])
        _, full = A.site_weights(sv, d["topics"], d["tmap"], rc, "shared", 0.4)
        _, cut = A.site_weights(sv[sv["date"].to_numpy() <= last.reindex(key).to_numpy()], d["topics"], d["tmap"], rc,
                                "shared", 0.4)
        rows.append({"course_set": cs, "n_shared_sites": len(full),
                     "n_sites_shared_because_of_revision_clicks": len(set(full["id_site"]) ^ set(cut["id_site"]))})
    return pd.DataFrame(rows)


def scoring_scale():
    """Defect D3: the stored out-of-validation plans equal the raw-K plans, not the preregistered M*R plans."""
    sys.path.insert(0, str(ROOT / "demo"))
    import build_demo_data as B
    from src.v3 import planner as PN

    frozen, rows = B.load_frozen(), []
    for cs, f in (("CCC", "results/v3/holdout/plans_CCC.csv"),
                  ("DDDFFF", "results/v3/posthoc_2014J/plans_DDD_2014J.csv")):
        b = B.fit_course(cs, frozen)
        rec = pd.read_csv(ROOT / f)
        key = ["sid", "weeks_before", "j"]
        for scale in ("raw", "score"):
            m = rec.merge(PN.plans(b["var"], b["data"], "test", scale=scale)[key + ["K_exam"]], on=key,
                          suffixes=("_recorded", ""))
            rows.append({"recorded_file": f, "scale": scale, "is_preregistered_scale": scale == "score",
                         "n_rows": len(m), "max_abs_diff_K_exam": round(float((m["K_exam"] - m["K_exam_recorded"]).abs().max()), 9)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {cs: load(cs) for cs in SETS}
    leak, first = label_in_ability(sets)
    out = {"label_in_ability": leak, "first_topic_ability_auc": first, "exam_day": exam_days(),
           "attribution_revision_clicks": attribution_uses_revision_clicks(sets),
           "scoring_scale": scoring_scale()}
    for name, df in out.items():
        df.to_csv(OUT / f"{name}.csv", index=False, lineterminator="\n")
        print(f"== {name}\n{df.round(3).to_string()}")


if __name__ == "__main__":
    main()
