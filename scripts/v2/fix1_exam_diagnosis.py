"""v2 Fix 1 (post-hoc, validation only): audit and diagnose the exam-based retention check.

1. Audit: recompute Δ at the exam date independently from vle_daily for every assessed (student, topic) of the
   validation exam queries and compare with the pipeline's state table (also checks that no event on/after the
   exam date is used).
2. Diagnostics on validation: exam check per topic, Δ distribution at the exam date, share of students with
   revision-period clicks per topic, and what drives R.
Writes results/v2/exam_check/*.csv and results/v2/exam_check_diagnosis.md. Never touches v1 outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.utils.io import load_config  # noqa: E402

OUT = ROOT / "results" / "v2" / "exam_check"
P = ROOT / "data" / "processed"


def independent_delta(vd, inter, q, min_clicks=3):
    """Naive re-implementation of Methodology 5.1 for exam queries: last meaningful practice of j = max of its
    submission day and any later day (< t) with >= min_clicks clicks on j's resources."""
    rows = []
    vd_ok = vd[vd["clicks"] >= min_clicks]
    g = dict(tuple(vd_ok.groupby("sid")))
    for _, r in q.iterrows():
        sid, t = int(r["sid"]), float(r["t"])
        it = inter[(inter["sid"] == sid) & (inter["is_submitted"] == 1) & (inter["t_submit"] < t)]
        v = g.get(sid)
        for _, a in it.iterrows():
            last = a["t_submit"]
            if v is not None:
                d = v[(v["topic"] == a["topic"]) & (v["date"] > a["t_submit"]) & (v["date"] < t)]["date"]
                if len(d):
                    last = max(last, float(d.max()))
            rows.append({"qid": r["qid"], "j": int(a["topic"]), "delta_indep": t - last})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    q = pd.read_parquet(P / "queries.parquet")
    st = pd.read_parquet(P / "state_val.parquet")
    ks = pd.read_parquet(P / "knowledge_state.parquet")
    stu = pd.read_parquet(P / "students.parquet")
    inter = pd.read_parquet(P / "interactions.parquet")
    topics = pd.read_parquet(P / "topics.parquet")
    vd = pd.read_parquet(P / "vle_daily.parquet", columns=["sid", "topic", "date", "clicks"])
    qe = q[(q["kind"] == "exam") & (q["split"] == "val")]

    # ---------------- 1. audit -----------------
    ind = independent_delta(vd, inter, qe, cfg["features"]["revision_min_clicks"])
    pipe = st[st["qid"].isin(qe["qid"]) & st["assessed"]][["qid", "j", "delta"]]
    m = pipe.merge(ind, on=["qid", "j"], how="outer", indicator=True)
    audit = {
        "n_assessed_pairs_pipeline": int(len(pipe)), "n_assessed_pairs_independent": int(len(ind)),
        "pairs_only_in_one": int((m["_merge"] != "both").sum()),
        "max_abs_delta_difference": float(np.nanmax(np.abs(m["delta"] - m["delta_indep"]))),
    }
    # events on/after the exam date must not matter: the cube only reads days <= t-1
    after = vd.merge(qe[["sid", "t"]], on="sid")
    audit["clicks_on_or_after_exam_date_rows"] = int((after["date"] >= after["t"]).sum())
    audit["verdict"] = ("pipeline Δ equals the independent re-implementation for every pair"
                        if audit["pairs_only_in_one"] == 0 and audit["max_abs_delta_difference"] < 1e-9
                        else "MISMATCH")

    # ---------------- 2. diagnostics -----------------
    ke = ks[(ks["kind"] == "exam") & (ks["split"] == "val")].merge(
        stu[["sid", "exam_score", "code_presentation"]].rename(columns={"code_presentation": "pres"}), on="sid")
    ke = ke[ke["exam_score"].notna()]
    ex_pass = (ke["exam_score"] >= cfg["pass_mark"]).astype(int)
    ke["exam_pass"] = ex_pass
    rows = []
    for (mod, j), g in ke.groupby(["code_module", "j"]):
        a = g[g["assessed"]]
        r = {"module": mod, "topic": j, "n_students": len(g), "share_assessed": float(g["assessed"].mean())}
        for v in ["learned", "fixed"]:
            r[f"rho_M_{v}"] = sps.spearmanr(g[f"M_{v}"], g["exam_score"])[0]
            r[f"rho_K_{v}"] = sps.spearmanr(g[f"K_{v}"], g["exam_score"])[0]
            r[f"drho_{v}"] = r[f"rho_K_{v}"] - r[f"rho_M_{v}"]
        if len(a) > 10:
            r["rho_R_learned_assessed"] = sps.spearmanr(a["R_learned"], a["exam_score"])[0]
            r["mean_R_learned"] = float(a["R_learned"].mean())
            r["rho_R_vs_clicks_assessed"] = sps.spearmanr(a["R_learned"], a["c_inf"])[0]
            r["rho_clicks_vs_exam_assessed"] = sps.spearmanr(a["c_inf"], a["exam_score"])[0]
        rows.append(r)
    per_topic = pd.DataFrame(rows)

    # mean-over-topics check, learned (reproduces v1 validation number)
    agg = ke.groupby("sid").agg(M=("M_learned", "mean"), K=("K_learned", "mean"), R=("R_learned", "mean"),
                                ex=("exam_score", "first"), mod=("code_module", "first"))
    overall = {"rho_meanM": sps.spearmanr(agg["M"], agg["ex"])[0], "rho_meanK": sps.spearmanr(agg["K"], agg["ex"])[0]}
    overall["drho"] = overall["rho_meanK"] - overall["rho_meanM"]
    overall["rho_meanR_assessed_vs_exam"] = sps.spearmanr(agg["R"].fillna(1), agg["ex"])[0]
    # leave-one-topic-out: which topics drive the negative difference
    loo = []
    for (mod, j) in per_topic[["module", "topic"]].itertuples(index=False):
        sub = ke[~((ke["code_module"] == mod) & (ke["j"] == j))]
        a2 = sub.groupby("sid").agg(M=("M_learned", "mean"), K=("K_learned", "mean"), ex=("exam_score", "first"))
        d = sps.spearmanr(a2["K"], a2["ex"])[0] - sps.spearmanr(a2["M"], a2["ex"])[0]
        loo.append({"module": mod, "topic_dropped": j, "drho_without_topic": d,
                    "change_vs_all": d - overall["drho"]})
    loo = pd.DataFrame(loo)

    # Δ distribution at the exam date and revision-period activity
    tp = topics.copy()
    st_ex = st[st["qid"].isin(qe["qid"])].merge(qe[["qid", "sid", "t", "code_presentation"]], on="qid")
    st_ex = st_ex.merge(stu[["sid", "exam_score"]], on="sid")
    st_ex = st_ex[st_ex["exam_score"].notna()]
    # revision period = (due date of the last topic due strictly before the exam, exam date)
    last_due = {}
    for (mod, pres), g in tp.groupby(["code_module", "code_presentation"]):
        texam = qe[(qe["code_module"] == mod) & (qe["code_presentation"] == pres)]["t"]
        if len(texam):
            te = float(texam.iloc[0])
            last_due[(mod, pres)] = float(g.loc[g["t_due"] < te, "t_due"].max())
    st_ex["rev_start"] = [last_due[(m_, p_)] for m_, p_ in zip(st_ex["code_module"], st_ex["code_presentation"])]
    vv = vd.merge(st_ex[["sid", "t", "rev_start"]].drop_duplicates("sid"), on="sid")
    vv = vv[(vv["date"] > vv["rev_start"]) & (vv["date"] < vv["t"])]
    act = vv.groupby(["sid", "topic"]).agg(rev_clicks=("clicks", "sum"),
                                           rev_days3=("clicks", lambda s: int((s >= 3).sum()))).reset_index()
    st_ex = st_ex.merge(act.rename(columns={"topic": "j"}), on=["sid", "j"], how="left").fillna(
        {"rev_clicks": 0, "rev_days3": 0})
    dist = []
    for (mod, j), g in st_ex.groupby(["code_module", "j"]):
        a = g[g["assessed"]]
        dist.append({"module": mod, "topic": j, "n": len(g), "n_assessed": len(a),
                     "delta_p10": a["delta"].quantile(0.1), "delta_median": a["delta"].median(),
                     "delta_p90": a["delta"].quantile(0.9),
                     "share_any_revision_clicks": float((g["rev_clicks"] > 0).mean()),
                     "share_reset_in_revision_period": float((g["rev_days3"] > 0).mean()),
                     "share_reset_any_time_after_submission": float(a["revised"].mean()) if len(a) else np.nan})
    dist = pd.DataFrame(dist)

    per_topic.to_csv(OUT / "per_topic_exam_check_val.csv", index=False)
    loo.to_csv(OUT / "leave_one_topic_out_val.csv", index=False)
    dist.to_csv(OUT / "delta_and_revision_at_exam_val.csv", index=False)
    json.dump({"audit": audit, "overall_learned": overall}, open(OUT / "audit_val.json", "w"), indent=2,
              default=float)
    print(json.dumps({"audit": audit, "overall": overall}, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(per_topic.round(3).to_string())
    print(loo.round(3).to_string())
    print(dist.round(3).to_string())


if __name__ == "__main__":
    main()
