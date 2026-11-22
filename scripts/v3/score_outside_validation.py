"""v3 scoring OUTSIDE validation (run only after the "v3 preregistration" commit), with frozen
config/final_config_v3.yaml.

  --part holdout   PRIMARY: AAA, BBB, CCC, EEE, GGG 2014J (coefficients refit per module, everything else frozen);
                   includes the CCC exam check and the CCC planner (EXPLORATORY).
  --part posthoc   DDD/FFF 2014J, POST-HOC / EXPLORATORY (scored once).
Outputs: results/v3/holdout/, results/v3/posthoc_2014J/, results/v3/preregistered_criteria.csv (holdout part).
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.evaluation.stats import fast_student_bootstrap_indices, wilcoxon_holm  # noqa: E402
from src.utils.io import get_logger, load_config, seed_everything  # noqa: E402
from src.v3 import common as C  # noqa: E402
from src.v3 import pipeline as PL  # noqa: E402
from src.v3 import planner as PN  # noqa: E402
from src.v3.data import HOLDOUT, load, practice  # noqa: E402

warnings.filterwarnings("ignore")
log = get_logger("v3_score")
CIDS = ["A3", "A5", "A6", "A8", "A9"]


def within_auc_boot(y, p_new, p_ref, module, sid, nb, seed):
    Wb = fast_student_bootstrap_indices(sid, nb, seed)

    def wauc(p, w):
        tot, n = 0.0, 0.0
        for m in np.unique(module):
            s = module == m
            tot += roc_auc_score(y[s], p[s], sample_weight=w[s]) * w[s].sum()
            n += w[s].sum()
        return tot / n
    one = np.ones(len(y))
    a_new, a_ref = wauc(p_new, one), wauc(p_ref, one)
    diffs = np.array([wauc(p_new, w) - wauc(p_ref, w) for w in Wb])
    news = np.array([wauc(p_new, w) for w in Wb])
    refs = np.array([wauc(p_ref, w) for w in Wb])
    return {"auc_v3": a_new, "auc_v3_lo": np.percentile(news, 2.5), "auc_v3_hi": np.percentile(news, 97.5),
            "auc_v2": a_ref, "auc_v2_lo": np.percentile(refs, 2.5), "auc_v2_hi": np.percentile(refs, 97.5),
            "diff": a_new - a_ref, "diff_lo": np.percentile(diffs, 2.5), "diff_hi": np.percentile(diffs, 97.5)}


def holdout(cfg, sel, v3final, v2final, v1fit, nb, seed):
    F4 = runpy.run_path(str(ROOT / "scripts/v2/fix4_holdout.py"), run_name="f4")
    out = ROOT / "results/v3/holdout"
    out.mkdir(parents=True, exist_ok=True)
    rc = cfg["recommender"]
    Y, P3, P2, MOD, SID, LL = [], [], [], [], [], []
    ps_all, labels, exam, plan_rows = {}, {}, None, []
    coefs = {}
    for i, mod in enumerate(HOLDOUT):
        cfg4 = F4["module_cfg"](cfg, mod)
        data4 = F4["load"](cfg4)
        D4, fitted = F4["fit_frozen"](data4, v1fit)
        rd, ks, _, _ = F4["recdata"](data4, cfg4, fitted)
        data = load(mod)
        Dt, Dte = PL.due_design(data, "train"), PL.due_design(data, "test")
        var, varI = C.build_variants(sel, mod, data, ["train", "test"])
        var.fit(data, Dt)
        varI.fit(data, Dt)
        coefs[mod] = var.m.coef() if callable(getattr(var.m, "coef", None)) else None
        y = Dte.rows["y"].to_numpy(float)
        assert np.array_equal(y, D4["test"].rows["y"].to_numpy(float))
        p3 = var.predict(Dte)
        p2 = fitted["learned"].predict_proba(D4["test"])
        Y.append(y), P3.append(p3), P2.append(p2), MOD.append(np.full(len(y), mod))
        SID.append(Dte.rows["sid"].to_numpy() + 10_000_000 * i)
        LL.append({"module": mod, "n_rows": len(y), "auc_v3": roc_auc_score(y, p3), "auc_v2_learned": roc_auc_score(y, p2),
                   "logloss_v3": PL.logloss(y, p3), "logloss_v2_learned": PL.logloss(y, p2)})
        # recommender
        C.inject(rd, C.knowledge_for_decisions(var, data, "test"), "v3")
        C.inject(rd, C.knowledge_for_decisions(varI, data, "test"), "v3I")
        paths = {**C.v2_paths(rd, v2final, CIDS, rc), **C.v3_paths(rd, v3final, CIDS, rc)}
        ps = C.per_student(rd, paths)
        for k, v in ps.items():
            ps_all.setdefault(k, []).append(v)
        labels.update({k: v[2]["label"] for k, v in paths.items()})
        # CCC: exam check + planner (exploratory)
        if data["students"].loc[data["students"]["split"] == "test", "exam_score"].notna().sum() > 30:
            q = data["queries"]
            eq = q[(q["kind"] == "exam") & (q["split"] == "test")]
            k3 = PL.knowledge_at(var, data, eq, "test")
            k2 = PL.knowledge_at(PL.Variant("hl_v2", hl_model=fitted["learned"]), data, eq, "test")
            d3, g3 = PL.exam_delta_rho(k3, data["students"])
            d2, g2 = PL.exam_delta_rho(k2, data["students"])
            exam = {"module": mod, "v3_rho_M": d3["rho_M"], "v3_rho_K": d3["rho_K"], "v3_delta_rho": d3["delta_rho"],
                    "v2_delta_rho_recomputed": d2["delta_rho"], **PL.paired_delta_rho_diff(g3, g2, nb, seed),
                    "label": "EXPLORATORY (CCC exam previously seen)"}
            plan = PN.plans(var, data, "test")
            plan.to_csv(out / f"plans_{mod}.csv", index=False)
            td = pd.read_parquet(ROOT / f"data/interim/v2_holdout/{mod}/total_daily_clicks.parquet")
            meanM = k3.groupby("sid")["Mhat"].mean().rename("mean_M").reset_index()
            obs = PN.observational_check(plan, practice(mod, data, sel["attribution"]), td, meanM, data["students"], nb,
                                         seed)
            obs.insert(0, "module", mod)
            obs.to_csv(out / f"observational_check_{mod}.csv", index=False)
        log.info("holdout %s done", mod)
    y, p3, p2, mod_, sid = map(np.concatenate, [Y, P3, P2, MOD, SID])
    auc = within_auc_boot(y, p3, p2, mod_, sid, nb, seed)
    pd.DataFrame([{**auc, "logloss_v3": PL.logloss(y, p3), "logloss_v2_learned": PL.logloss(y, p2), "n_rows": len(y)}]
                 ).to_csv(out / "model_metrics_holdout.csv", index=False)
    pd.DataFrame(LL).to_csv(out / "model_metrics_holdout_by_module.csv", index=False)
    json.dump(coefs, open(out / "coefficients_by_module.json", "w"), indent=2, default=float)
    ps = {k: {m: np.concatenate([d[m] for d in v]) for m in v[0]} for k, v in ps_all.items()}
    rows = C.ranking_rows(ps, labels, nb, seed, {"split": "holdout (pooled)"})
    pd.DataFrame(rows).to_csv(out / "recommender_holdout.csv", index=False)
    wt = C.wilcoxon_table(ps)
    wt.to_csv(out / "recommender_wilcoxon_holdout.csv", index=False)
    if exam:
        pd.DataFrame([exam]).to_csv(out / "exam_check_CCC.csv", index=False)
    # preregistered criteria
    rk = pd.DataFrame(rows)
    nd = lambda c: rk[(rk["config"] == c) & (rk["metric"] == "NDCG@5") & (rk["relevance"] == "AB")].iloc[0]  # noqa: E731
    a5v3, a5v2 = nd("v3:A5"), nd("v2:A5")
    crit = [
        {"criterion": "(a) holdout within-course AUC: v3 not worse than v2 learned half-life by more than 0.002",
         "value": f"v3 {auc['auc_v3']:.4f} [{auc['auc_v3_lo']:.4f}, {auc['auc_v3_hi']:.4f}] vs v2 {auc['auc_v2']:.4f} "
                  f"[{auc['auc_v2_lo']:.4f}, {auc['auc_v2_hi']:.4f}]; diff {auc['diff']:+.4f} [{auc['diff_lo']:+.4f}, {auc['diff_hi']:+.4f}]",
         "met": bool(auc["diff"] >= -0.002)},
        {"criterion": "(b) CCC exam Δρ higher than v2's −0.096 AND paired bootstrap CI of the improvement excludes 0",
         "value": (f"v3 Δρ {exam['v3_delta_rho']:+.4f}; v2 recomputed {exam['v2_delta_rho_recomputed']:+.4f}; improvement "
                   f"{exam['improvement']:+.4f} [{exam['ci_lo']:+.4f}, {exam['ci_hi']:+.4f}]") if exam else "not computed",
         "met": bool(exam and exam["v3_delta_rho"] > -0.096 and exam["ci_lo"] > 0)},
        {"criterion": "(c) holdout NDCG@5 (combined) of v3 A5 >= v2 A5",
         "value": f"v3 A5 {a5v3['value']:.4f} [{a5v3['ci_lo']:.4f}, {a5v3['ci_hi']:.4f}] vs v2 A5 {a5v2['value']:.4f} "
                  f"[{a5v2['ci_lo']:.4f}, {a5v2['ci_hi']:.4f}]",
         "met": bool(a5v3["value"] >= a5v2["value"])},
    ]
    pd.DataFrame(crit).to_csv(ROOT / "results/v3/preregistered_criteria.csv", index=False)
    log.info("criteria: %s", crit)


def posthoc(cfg, sel, v3final, v2final, v1fit, nb, seed):
    T23 = runpy.run_path(str(ROOT / "scripts/v2/fix23_tune_validation.py"), run_name="t23")
    out = ROOT / "results/v3/posthoc_2014J"
    out.mkdir(parents=True, exist_ok=True)
    rc = cfg["recommender"]
    data = load("DDDFFF")
    Dt, Dte = PL.due_design(data, "train"), PL.due_design(data, "test")
    var, varI = C.build_variants(sel, "DDDFFF", data, ["train", "test"])
    var.fit(data, Dt)
    varI.fit(data, Dt)
    v0 = PL.Variant("hl_v2", hl_model=v1fit["learned"])
    y = Dte.rows["y"].to_numpy(float)
    p3, p0 = var.predict(Dte), v0.predict(Dte)
    auc = within_auc_boot(y, p3, p0, Dte.rows["code_module"].to_numpy(), Dte.rows["sid"].to_numpy(), nb, seed)
    pd.DataFrame([{**auc, "logloss_v3": PL.logloss(y, p3), "logloss_v2_learned": PL.logloss(y, p0), "n_rows": len(y),
                   "label": "POST-HOC / EXPLORATORY"}]).to_csv(out / "model_metrics_2014J.csv", index=False)
    q = data["queries"]
    eq = q[(q["kind"] == "exam") & (q["split"] == "test")]
    k3, k0 = PL.knowledge_at(var, data, eq, "test"), PL.knowledge_at(v0, data, eq, "test")
    d3, g3 = PL.exam_delta_rho(k3, data["students"])
    d0, g0 = PL.exam_delta_rho(k0, data["students"])
    pd.DataFrame([{"split": "DDD 2014J", "v3_delta_rho": d3["delta_rho"], "v2_delta_rho": d0["delta_rho"],
                   **PL.paired_delta_rho_diff(g3, g0, nb, seed), "label": "POST-HOC / EXPLORATORY"}]).to_csv(
        out / "exam_check_DDD_2014J.csv", index=False)
    plan = PN.plans(var, data, "test")
    plan.to_csv(out / "plans_DDD_2014J.csv", index=False)
    td = pd.read_parquet(ROOT / "data/interim/total_daily_clicks.parquet")
    meanM = k3.groupby("sid")["Mhat"].mean().rename("mean_M").reset_index()
    PN.observational_check(plan, practice("DDDFFF", data, sel["attribution"]), td, meanM, data["students"], nb,
                           seed).to_csv(out / "observational_check_DDD_2014J.csv", index=False)
    ks_train = T23["train_knowledge_state"](cfg)
    rd, _ = T23["build_rd"](cfg, "test")
    rd = T23["attach"](rd, cfg, ks_train)
    C.inject(rd, C.knowledge_for_decisions(var, data, "test"), "v3")
    C.inject(rd, C.knowledge_for_decisions(varI, data, "test"), "v3I")
    paths = {**C.v2_paths(rd, v2final, CIDS, rc), **C.v3_paths(rd, v3final, CIDS, rc)}
    ps = C.per_student(rd, paths)
    pd.DataFrame(C.ranking_rows(ps, {k: v[2]["label"] for k, v in paths.items()}, nb, seed,
                                {"split": "DDD/FFF 2014J POST-HOC / EXPLORATORY"})).to_csv(out / "recommender_2014J.csv",
                                                                                          index=False)
    C.wilcoxon_table(ps).to_csv(out / "recommender_wilcoxon_2014J.csv", index=False)
    log.info("posthoc done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["holdout", "posthoc"])
    a = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    sel = C.selected()
    v3final = yaml.safe_load(open(ROOT / "config/final_config_v3.yaml", encoding="utf-8"))
    v2final = yaml.safe_load(open(ROOT / "config/final_config_v2.yaml"))
    v1fit = joblib.load(ROOT / "models/fitted_models.joblib")
    nb = cfg["evaluation"]["bootstrap"]
    (holdout if a.part == "holdout" else posthoc)(cfg, sel, v3final, v2final, v1fit, nb, cfg["seed"])
