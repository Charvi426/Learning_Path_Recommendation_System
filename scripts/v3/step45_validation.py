"""v3 Steps 4 + 5 on VALIDATION only (DDD/FFF 2014B), then freeze config/final_config_v3.yaml.

Step 4: pre-exam revision planner on DDD validation exam-takers; exam check vs v2 (paired bootstrap);
        observational regression (not causal); 5 sample plans.
Step 5: recommender with the v3 knowledge state; v2 filters frozen; weights re-tuned on validation;
        validation comparison v2 vs v3 with Wilcoxon + Holm.
"""
from __future__ import annotations

import datetime as dt
import json
import runpy
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.utils.io import get_logger, load_config, seed_everything  # noqa: E402
from src.v3 import common as C  # noqa: E402
from src.v3 import pipeline as PL  # noqa: E402
from src.v3 import planner as PN  # noqa: E402
from src.v3.data import load, practice  # noqa: E402
from src.v3.recommend import tune_weights  # noqa: E402

warnings.filterwarnings("ignore")
log = get_logger("v3_step45")
T23 = runpy.run_path(str(ROOT / "scripts/v2/fix23_tune_validation.py"), run_name="t23")
OUT = ROOT / "results/v3"
SCALE = {}
V3_CONFIGS = {"A3": ("none", "A3"), "A5": ("v3", "A5"), "A6": ("v3I", "A6"), "A8": ("v3", "A8"), "A9": ("v3I", "A9")}


def sample_plans_md(plan, n=5, seed=0):
    rng = np.random.default_rng(seed)
    sids = plan.loc[plan["weeks_before"] == 4, "sid"].unique()
    pick = rng.choice(sids, size=min(n, len(sids)), replace=False)
    lines = ["# Sample pre-exam revision plans (v3, DDD validation 2014B)", "",
             "Weekly plans in the last 4 weeks before the exam: the 3 assessed topics with the largest expected loss "
             "M − K(exam), assuming no further practice. Students are internal pseudonymous ids.", ""]
    for s in pick:
        g = plan[plan["sid"] == s]
        lines.append(f"## Student sid {s} ({g['code_module'].iloc[0]} {g['code_presentation'].iloc[0]}, exam day "
                     f"{g['t_exam'].iloc[0]:.0f})")
        for w in [4, 3, 2, 1]:
            gw = g[(g["weeks_before"] == w) & g["flagged"]].sort_values("rank")
            lines.append(f"**{w} week(s) before the exam (day {g['t_exam'].iloc[0] - 7 * w:.0f})**")
            if gw.empty:
                lines.append("- nothing flagged (no assessed topic expected to lose knowledge)")
            for _, r in gw.iterrows():
                lines.append(f"- {r['rank']}. {r['explanation']} (priority {r['priority']:.2f})")
            lines.append("")
    return "\n".join(lines)


def main():
    cfg = load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    nb = cfg["evaluation"]["bootstrap"]
    rc = cfg["recommender"]
    v1 = joblib.load(ROOT / "models/fitted_models.joblib")
    sel = C.selected()
    data = load("DDDFFF")
    Dt = PL.due_design(data, "train")
    var, varI = C.build_variants(sel, "DDDFFF", data, ["train", "val"])
    var.fit(data, Dt)
    varI.fit(data, Dt)
    v0 = PL.Variant("hl_v2", hl_model=v1["learned"])
    (OUT / "planner").mkdir(parents=True, exist_ok=True)
    coef = {"v3": var.m.coef() if hasattr(var.m, "coef") and callable(var.m.coef) else None,
            "v3I": varI.m.coef() if hasattr(varI.m, "coef") and callable(varI.m.coef) else None}
    json.dump(coef, open(OUT / "coefficients_selected_variant_train.json", "w"), indent=2)

    # ---------------- knowledge-scale choice for recommender + planner (validation, A5) -----------------
    ks_train = T23["train_knowledge_state"](cfg)
    rd, _ = T23["build_rd"](cfg, "val")
    rd = T23["attach"](rd, cfg, ks_train)
    kv, kvI = C.knowledge_for_decisions(var, data, "val"), C.knowledge_for_decisions(varI, data, "val")
    v2final = yaml.safe_load(open(ROOT / "config/final_config_v2.yaml"))
    sc = []
    for scale, kcol in [("raw", "K"), ("score", "K_score")]:
        C.inject(rd, kv, "v3", kcol)
        t2 = v2final["ablations"]["A5"]
        r = tune_weights(rd, {**t2["config"], "model": "v3"}, t2["filters"], rc)
        sc.append({"scale": scale, "kcol": kcol, "A5_val_NDCG@5": r["val_NDCG@5"], "A5_val_R@5": r["val_R@5"],
                   "weights": json.dumps(r["weights"])})
    sc = pd.DataFrame(sc)
    sc.to_csv(OUT / "knowledge_scale_selection_val.csv", index=False)
    SCALE.update(sc.sort_values(["A5_val_NDCG@5", "A5_val_R@5"], ascending=False).iloc[0][["scale", "kcol"]].to_dict())
    log.info("knowledge scale for recommender/planner: %s | %s", SCALE, sc.to_dict("records"))

    # ---------------- Step 4: planner on DDD validation -----------------
    q = data["queries"]
    exam_q = q[(q["kind"] == "exam") & (q["split"] == "val")]
    k3, k0 = PL.knowledge_at(var, data, exam_q, "val"), PL.knowledge_at(v0, data, exam_q, "val")
    d3, g3 = PL.exam_delta_rho(k3, data["students"])
    d0, g0 = PL.exam_delta_rho(k0, data["students"])
    cmp = PL.paired_delta_rho_diff(g3, g0, nb, seed)
    pd.DataFrame([{"split": "DDD validation 2014B", "v3_delta_rho": d3["delta_rho"], "v3_rho_K": d3["rho_K"],
                   "v3_rho_M": d3["rho_M"], "v2_delta_rho": d0["delta_rho"], **cmp}]).to_csv(
        OUT / "planner/exam_check_val.csv", index=False)
    plan = PN.plans(var, data, "val", scale=SCALE["scale"])
    plan.to_csv(OUT / "planner/plans_val.csv", index=False)
    meanM = k3.groupby("sid")["Mhat"].mean().rename("mean_M").reset_index()
    td = pd.read_parquet(ROOT / "data/interim/total_daily_clicks.parquet")
    obs = PN.observational_check(plan, practice("DDDFFF", data, sel["attribution"]), td, meanM, data["students"], nb, seed)
    obs.to_csv(OUT / "planner/observational_check_val.csv", index=False)
    (OUT / "sample_revision_plans.md").write_text(sample_plans_md(plan, 5, seed), encoding="utf-8")
    log.info("planner val: v3 Δρ %.4f, v2 Δρ %.4f, improvement %s", d3["delta_rho"], d0["delta_rho"], cmp)

    # ---------------- Step 5: recommender on validation -----------------
    C.inject(rd, kv, "v3", SCALE["kcol"])
    C.inject(rd, kvI, "v3I", SCALE["kcol"])
    tuned = {}
    for cid, (model, v2cid) in V3_CONFIGS.items():
        t2 = v2final["ablations"][v2cid]
        conf = {**t2["config"], "model": model}
        r = tune_weights(rd, conf, t2["filters"], rc)
        tuned[cid] = {"config": conf, "filters": t2["filters"], **r}
        log.info("v3 %s weights %s val NDCG@5 %.4f", cid, r["weights"], r["val_NDCG@5"])
    final = {"frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
             "note": ("v3 (POST-HOC). Variant, decay parameters, attribution and recommender weights chosen on "
                      "DDD/FFF validation (2014B) only. Frozen before any holdout, CCC-exam or 2014J v3 scoring."),
             "memory_variant": sel, "knowledge_scale": dict(SCALE), "planner": {"weeks_before_exam": PN.WEEKS, "top_n": 3},
             "recommender": {**v2final["recommender"], "filters": "frozen from config/final_config_v2.yaml"},
             "pop_definition": v2final["pop_definition"], "ablations": tuned}
    with open(ROOT / "config/final_config_v3.yaml", "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(T23["S4"]["to_plain"](final), f, sort_keys=False, allow_unicode=False)
    paths = {**C.v2_paths(rd, v2final, ["A3", "A5", "A6", "A8", "A9"], rc),
             **C.v3_paths(rd, final, list(V3_CONFIGS), rc)}
    ps = C.per_student(rd, paths)
    rows = C.ranking_rows(ps, {k: v[2]["label"] for k, v in paths.items()}, nb, seed, {"split": "validation"})
    pd.DataFrame(rows).to_csv(OUT / "recommender_val.csv", index=False)
    C.wilcoxon_table(ps).to_csv(OUT / "recommender_wilcoxon_val.csv", index=False)
    log.info("done")


if __name__ == "__main__":
    main()
