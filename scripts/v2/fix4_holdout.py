"""v2 Fix 4 (post-hoc): holdout modules never used in v1 as a fresh test set.

For each holdout module (run independently, so cohort statistics and coefficients are module-specific):
  - topic windows per presentation with that presentation's own topic count (same-day merge, nothing dropped);
  - train = earlier presentations, test = 2014J (latest); no validation split, NOTHING is tuned here;
  - model coefficients refitted on the module's training data with FROZEN v1 hyper-parameters (L2, S, C, windows);
  - recommender configurations FROZEN: v1 (config/final_config.yaml: A3, A5, A6) and v2
    (config/final_config_v2.yaml: A1, A3, A5, A6, A8, A9). Relative-filter quantiles and the popularity term are
    computed from the module's own training presentations with the frozen q values / definition.
Outputs: results/v2/holdout/ (pooled + per-module tables). Intermediate data: data/processed/v2_holdout/<module>/.

usage: python scripts/v2/fix4_holdout.py [--modules AAA BBB CCC EEE GGG] [--skip-prep]
"""
from __future__ import annotations

import argparse
import copy
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.evaluation import evaluate as EV  # noqa: E402
from src.evaluation.model_metrics import all_metrics, ece  # noqa: E402
from src.evaluation.relevance import relevance_table  # noqa: E402
from src.evaluation.stats import bootstrap_mean_ci, fast_student_bootstrap_indices, wilcoxon_holm  # noqa: E402
from src.features.pipeline import build_features  # noqa: E402
from src.features.prerequisites import prereq_dict  # noqa: E402
from src.models.das3h import ForgettingPFA, Majority, SklearnLogit  # noqa: E402
from src.models.data import build_design, topic_index_map, training_targets  # noqa: E402
from src.models.knowledge_state import knowledge_state  # noqa: E402
from src.models.train import fit_forgetting  # noqa: E402
from src.recommend import v2 as V2  # noqa: E402
from src.recommend.engine import evaluate_paths  # noqa: E402
from src.utils.io import get_logger, load_config, seed_everything, write_parquet  # noqa: E402

log = get_logger("v2_holdout")
HOLDOUT = ["AAA", "BBB", "CCC", "EEE", "GGG"]
OUT = ROOT / "results" / "v2" / "holdout"
S1 = runpy.run_path(str(ROOT / "scripts" / "01_preprocess.py"), run_name="s1")
MODEL_ORDER = ["majority", "topic_difficulty", "pfa", "none", "fixed", "learned", "learnedI", "das3h"]
FORGET = ["none", "fixed", "learned", "learnedI"]
V1_CONFIGS = ["A3", "A5", "A6"]
V2_CONFIGS = ["A1", "A3", "A5", "A6", "A8", "A9"]
COMPARISONS = [("v2:A5", "v2:A3", "forgetting: A5 vs A3 (v2)"), ("v2:A8", "v2:A1", "popularity term: A8 vs A1 (v2)"),
               ("v2:A9", "v2:A8", "interest: A9 vs A8 (v2)")]


def module_cfg(base, mod, test_pres="2014J"):
    cfg = copy.deepcopy(base)
    raw = pd.read_csv(ROOT / "data" / "raw" / "courses.csv")
    pres = sorted(raw.loc[raw["code_module"] == mod, "code_presentation"])
    cfg["modules"] = [mod]
    cfg["drop_assessment_types"] = {}
    cfg["split"] = {"train": [p for p in pres if p != test_pres], "val": [], "test": [test_pres], "fallback": False}
    cfg["paths"] = {**cfg["paths"], "interim": f"data/interim/v2_holdout/{mod}",
                    "processed": f"data/processed/v2_holdout/{mod}", "models": f"models/v2_holdout/{mod}",
                    "tables": f"results/v2/holdout/{mod}/prep_tables", "figures": f"results/v2/holdout/{mod}/figures",
                    "sample_paths": f"results/v2/holdout/{mod}/sample_paths"}
    return cfg


def eligibility(base):
    a = pd.read_csv(ROOT / "data" / "raw" / "assessments.csv")
    ne = a[(a["assessment_type"] != "Exam") & a["code_module"].isin(HOLDOUT)]
    t = ne.groupby(["code_module", "code_presentation"])["date"].nunique().rename("n_topics").reset_index()
    rows = []
    for m, g in t.groupby("code_module"):
        ok_t = bool((g["n_topics"] >= 4).all())
        ok_p = len(g) >= 2
        rows.append({"module": m, "presentations": ",".join(g["code_presentation"]),
                     "topics_per_presentation": ",".join(str(x) for x in g["n_topics"]),
                     "min_topics_ok(>=4)": ok_t, "presentations_ok(>=2)": ok_p, "kept": ok_t and ok_p,
                     "reason": "" if ok_t and ok_p else ("< 4 topics in a presentation" if not ok_t else "< 2 presentations")})
    return pd.DataFrame(rows)


def prepare(cfg):
    S1["main"](cfg)
    build_features(cfg, "default", run_validity=False)


def load(cfg):
    d = ROOT / cfg["paths"]["processed"]
    data = {"queries": pd.read_parquet(d / "queries.parquet"),
            "inter": pd.read_parquet(d / "interactions_variant.parquet"),
            "edges": pd.read_csv(d / "prereq_edges.csv"), "topics": pd.read_parquet(d / "topics.parquet"),
            "students": pd.read_parquet(d / "students.parquet"), "diff": pd.read_csv(d / "topic_difficulty.csv"),
            "state": {sp: pd.read_parquet(d / f"state_{sp}.parquet") for sp in ["train", "test"]}, "dir": d}
    data["tindex"] = topic_index_map(data["topics"][["code_module", "topic"]])
    chain = prereq_dict(data["edges"][data["edges"]["source"] == "chain"])
    data["graphs"] = {"chain": chain, "data": chain}
    data["T"] = int(data["topics"]["topic"].max())
    return data


def fit_frozen(data, v1fit):
    D = {sp: build_design(training_targets(data["queries"], data["inter"], sp), data["state"][sp],
                          data["graphs"]["chain"], data["tindex"], (7, 14, 30, np.inf)) for sp in ["train", "test"]}
    fitted = {"majority": Majority().fit(D["train"])}
    fitted["topic_difficulty"] = SklearnLogit("irt", C=v1fit["topic_difficulty"].C).fit(D["train"])
    fitted["pfa"] = SklearnLogit("pfa", C=v1fit["pfa"].C).fit(D["train"])
    fitted["das3h"] = SklearnLogit("das3h", C=v1fit["das3h"].C, windows=v1fit["das3h"].windows).fit(D["train"])
    none = ForgettingPFA("none", False, l2=v1fit["none"].l2, n_topics=len(data["tindex"])).fit(D["train"])
    fitted["none"] = none
    for name in ["fixed", "learned", "learnedI"]:
        fitted[name] = fit_forgetting(name, D, data, v1fit[name].l2, v1fit["fixed"].S, init_from=none)
    return D, fitted


def recdata(data, cfg, fitted):
    forget = {k: fitted[k] for k in FORGET}
    ks = knowledge_state(data["queries"], data["state"]["test"], forget, data["graphs"], data["tindex"], data["diff"])
    ks_tr = knowledge_state(data["queries"], data["state"]["train"], forget, data["graphs"], data["tindex"],
                            data["diff"], kinds=("decision",))
    vd = pd.read_parquet(data["dir"] / "vle_daily.parquet", columns=["sid", "topic", "date", "clicks", "vol_clicks"])
    n_sid = int(data["students"]["sid"].max()) + 1
    dp = data["queries"][data["queries"]["kind"] == "decision"][["qid", "sid", "k", "t", "split", "code_module"]]
    rel = relevance_table(dp, data["inter"], vd, n_sid, data["T"], cfg["recommender"]["rel_a_min_clicks"])
    tr = rel[rel["split"] == "train"]
    pop = []
    for j in range(1, data["T"] + 1):
        g = tr.groupby(["code_module", "k"])[f"wclicks_{j}"].mean().rename("pop_clicks").reset_index()
        g["j"] = j
        pop.append(g)
    pop = pd.concat(pop).rename(columns={"k": "k_now"})
    rd = V2.RecDataV2(ks[ks["kind"] == "decision"], rel[rel["split"] == "test"], data["graphs"], FORGET, pop,
                      data["topics"], data["T"])
    # topics that do not exist in a student's presentation can never be recommended or relevant
    ne = ~rd.exists
    rd.assessed &= rd.exists
    for mname in FORGET:
        for k in ["M", "K", "p"]:
            rd.models[mname][k] = np.where(ne, 1.0, rd.models[mname][k])
    rd.I = np.where(ne, 0.5, rd.I)
    for a in ["relA", "relB", "relAB"]:
        setattr(rd, a, np.where(ne, 0.0, getattr(rd, a)))
    rd.pop = np.where(ne, 0.0, rd.pop)
    rd.pop_terms["train_window"] = V2.pop_train_window(rd)
    return rd, ks, ks_tr, vd


def module_run(base, mod, v1fit, v1final, v2final, nb, seed):
    cfg = module_cfg(base, mod)
    data = load(cfg)
    D, fitted = fit_frozen(data, v1fit)
    rd, ks, ks_tr, vd = recdata(data, cfg, fitted)
    pop_key = v2final["pop_definition"]
    if pop_key.startswith("cohort_w"):
        cd = V2.cohort_daily(vd, data["students"])
        rd.pop_terms[pop_key] = V2.pop_cohort_before(rd, cd, int(pop_key[8:]))
    q_all = sorted({t["filters"][k] for t in v2final["ablations"].values() if isinstance(t, dict) and "filters" in t
                    for k in ("q_rev", "q_new")})
    for mname in FORGET:
        rd.cohort_thr[mname] = V2.cohort_thresholds(ks_tr, rd, mname, q_all)
    rc = base["recommender"]
    paths, per_student, diag = {}, {}, {}
    for cid in V1_CONFIGS:
        P, aux, conf, W = EV.config_paths(rd, v1final, cid, seed, rc)
        paths[f"v1:{cid}"] = (P, aux, {**conf, "label": f"v1 {cid} {conf['label']}"}, W)
    for cid in V2_CONFIGS:
        t = v2final["ablations"][cid]
        if cid == "A1":
            P, aux, conf, W = EV.config_paths(rd, v2final, "A1", seed, rc)
            paths["v2:A1"] = (P, aux, {**conf, "label": "v2 A1 Popularity (= v1 A1)"}, W)
            continue
        W = np.array([[t["weights"][k] for k in V2.TERMS6]])
        P, aux = V2.paths_v2(rd, t["config"], W, t["filters"], rc["K"], rc["max_revision"], rc["horizon"])
        diag[f"v2:{cid}"] = V2.filter_diagnostics(rd, aux["new"], aux["rev"], aux["excl"], P[0])
        paths[f"v2:{cid}"] = (P[0], aux, {**t["config"], "label": f"v2 {cid} {t['config']['label']}"}, W[0])
    for name, (P, aux, conf, W) in paths.items():
        (_, ps), = evaluate_paths(rd, P[None], rels=("A", "B", "AB"))
        per_student[name] = ps
    # model predictions on test
    y = D["test"].rows["y"].to_numpy(float)
    preds = {n: fitted[n].predict_proba(D["test"]) for n in MODEL_ORDER}
    # simulated gain (DAS3H simulator refitted on this module's training data, frozen hyper-parameters)
    st = data["state"]["train"]
    due = data["queries"][(data["queries"]["kind"] == "due") & (data["queries"]["split"] == "train")]
    s = due[["qid", "k", "code_module"]].merge(st[["qid", "j", "clicks_raw_inf"]], left_on=["qid", "k"],
                                               right_on=["qid", "j"])
    med = s.groupby("code_module")["clicks_raw_inf"].median().to_dict()
    sg_paths = {k: (v[0], v[1] if v[1] is not None else None, v[2], v[3]) for k, v in paths.items()}
    sg, sg_ps = EV.simulated_gain(rd, sg_paths, fitted["das3h"], data["state"]["test"], data["queries"],
                                  data["graphs"], data["tindex"], med, nb, seed)
    # exam check (only modules with exam scores in OULAD)
    ke = ks[(ks["kind"] == "exam")]
    exam = None
    if len(ke) and data["students"]["exam_score"].notna().sum() > 30:
        exam, _ = EV.exam_check(ke, data["students"], ["fixed", "learned", "learnedI"], nb, seed, base["pass_mark"])
        exam.insert(0, "module", mod)
    info = {"module": mod, "train_presentations": cfg["split"]["train"], "test_presentation": "2014J",
            "n_test_students_with_outcomes": int(D["test"].rows["sid"].nunique()), "n_test_rows": int(len(y)),
            "n_decision_points": int(rd.N), "n_topics_max": data["T"],
            "groups": pd.Series(rd.group).value_counts().to_dict(),
            "test_pass_rate": float(y.mean()) if len(y) else None}
    return {"info": info, "per_student": per_student, "y": y, "preds": preds,
            "sid": D["test"].rows["sid"].to_numpy(), "sg": sg, "sg_ps": sg_ps, "exam": exam, "diag": diag,
            "labels": {k: v[2]["label"] for k, v in paths.items()}}


def pooled_model_metrics(res, nb, seed):
    y = np.concatenate([r["y"] for r in res])
    sid = np.concatenate([r["sid"] + 10_000_000 * i for i, r in enumerate(res)])
    Wb = fast_student_bootstrap_indices(sid, nb, seed)
    rows = []
    for n in MODEL_ORDER:
        p = np.concatenate([r["preds"][n] for r in res])
        m = all_metrics(y, p)
        m["ECE"] = ece(y, p)
        aucs = [roc_auc_score(y, p, sample_weight=w) for w in Wb] if n != "majority" else [np.nan]
        lls = []
        for w in Wb:
            pc = np.clip(p, 1e-12, 1 - 1e-12)
            lls.append(-np.sum(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))) / w.sum())
        rows.append({"model": n, "n_rows": len(y), **m, "AUC_lo": np.nanpercentile(aucs, 2.5),
                     "AUC_hi": np.nanpercentile(aucs, 97.5), "logloss_lo": np.percentile(lls, 2.5),
                     "logloss_hi": np.percentile(lls, 97.5)})
    pairs = [("learned", "none"), ("learned", "fixed"), ("learnedI", "learned"), ("learned", "pfa"),
             ("learned", "das3h")]
    pd_rows = []
    for a, b in pairs:
        pa = np.concatenate([r["preds"][a] for r in res])
        pb = np.concatenate([r["preds"][b] for r in res])
        d = [roc_auc_score(y, pa, sample_weight=w) - roc_auc_score(y, pb, sample_weight=w) for w in Wb]
        pd_rows.append({"model_x": a, "model_y": b, "AUC_diff": roc_auc_score(y, pa) - roc_auc_score(y, pb),
                        "AUC_diff_lo": np.percentile(d, 2.5), "AUC_diff_hi": np.percentile(d, 97.5)})
    return pd.DataFrame(rows), pd.DataFrame(pd_rows)


def main(modules, skip_prep):
    base = load_config()
    seed = base["seed"]
    seed_everything(seed)
    nb = base["evaluation"]["bootstrap"]
    OUT.mkdir(parents=True, exist_ok=True)
    el = eligibility(base)
    el.to_csv(OUT / "holdout_module_eligibility.csv", index=False)
    modules = [m for m in modules if m in set(el.loc[el["kept"], "module"])]
    v1fit = joblib.load(ROOT / "models" / "fitted_models.joblib")
    v1final = yaml.safe_load(open(ROOT / "config" / "final_config.yaml"))
    v2final = yaml.safe_load(open(ROOT / "config" / "final_config_v2.yaml"))
    res = []
    for mod in modules:
        cfg = module_cfg(base, mod)
        if not skip_prep:
            log.info("preparing %s (train %s, test 2014J)", mod, cfg["split"]["train"])
            prepare(cfg)
        log.info("evaluating %s", mod)
        r = module_run(base, mod, v1fit, v1final, v2final, nb, seed)
        log.info("%s", r["info"])
        res.append(r)
    # ---------------- pooled tables -----------------
    json.dump([r["info"] for r in res], open(OUT / "holdout_info.json", "w"), indent=2, default=str)
    mm, pdiff = pooled_model_metrics(res, nb, seed)
    mm.to_csv(OUT / "model_metrics_holdout.csv", index=False)
    pdiff.to_csv(OUT / "model_paired_diffs_holdout.csv", index=False)
    per_mod_mm = []
    for r in res:
        for n in MODEL_ORDER:
            y, p = r["y"], r["preds"][n]
            per_mod_mm.append({"module": r["info"]["module"], "model": n, **all_metrics(y, p)})
    pd.DataFrame(per_mod_mm).to_csv(OUT / "model_metrics_holdout_by_module.csv", index=False)
    names = list(res[0]["per_student"])
    pooled = {n: {met: np.concatenate([r["per_student"][n][met] for r in res]) for met in res[0]["per_student"][n]}
              for n in names}
    rows = []
    for n in names:
        for met, v in pooled[n].items():
            est, lo, hi = bootstrap_mean_ci(v, nb, seed)
            base_m, rel = met.split("[")
            rows.append({"config": n, "label": res[0]["labels"][n], "relevance": rel.rstrip("]"), "metric": base_m,
                         "value": est, "ci_lo": lo, "ci_hi": hi, "n_students": int(np.isfinite(v).sum())})
        for r in res:
            for met, v in r["per_student"][n].items():
                base_m, rel = met.split("[")
                rows.append({"config": n, "label": r["labels"][n], "relevance": rel.rstrip("]"), "metric": base_m,
                             "module": r["info"]["module"], "value": float(np.nanmean(v)) if np.isfinite(v).any()
                             else np.nan, "n_students": int(np.isfinite(v).sum())})
    rt = pd.DataFrame(rows)
    rt["module"] = rt.get("module", pd.Series(index=rt.index)).fillna("pooled")
    rt.to_csv(OUT / "ranking_metrics_holdout.csv", index=False)
    wt = pd.concat([wilcoxon_holm({name: (pooled[a][m], pooled[b][m]) for a, b, name in COMPARISONS}).assign(metric=m)
                    for m in ["NDCG@5[AB]", "NDCG@5[A]", "NDCG@5[B]"]], ignore_index=True)
    wt["family"] = np.where(wt["metric"] == "NDCG@5[AB]", "primary (combined)", "supplementary")
    wt.to_csv(OUT / "wilcoxon_holm_holdout.csv", index=False)
    # v1 vs v2 same configuration (supplementary)
    wv = wilcoxon_holm({f"v2 {c} vs v1 {c}": (pooled[f"v2:{c}"]["NDCG@5[AB]"], pooled[f"v1:{c}"]["NDCG@5[AB]"])
                        for c in V1_CONFIGS})
    wv.to_csv(OUT / "wilcoxon_holm_holdout_v1_vs_v2.csv", index=False)
    sg_rows = []
    for n in names:
        v = np.concatenate([r["sg_ps"][n] for r in res])
        est, lo, hi = bootstrap_mean_ci(v, nb, seed)
        sg_rows.append({"config": n, "label": res[0]["labels"][n], "module": "pooled", "mean_gain": est, "ci_lo": lo,
                        "ci_hi": hi})
    for r in res:
        s = r["sg"].copy()
        s["module"] = r["info"]["module"]
        sg_rows += s.rename(columns={"config": "config"}).to_dict("records")
    pd.DataFrame(sg_rows).to_csv(OUT / "simulated_gain_holdout.csv", index=False)
    ex = [r["exam"] for r in res if r["exam"] is not None]
    if ex:
        pd.concat(ex).to_csv(OUT / "exam_retention_check_holdout.csv", index=False)
    dg = [{"module": r["info"]["module"], "config": c, **d} for r in res for c, d in r["diag"].items()]
    pd.DataFrame(dg).to_csv(OUT / "filter_diagnostics_holdout.csv", index=False)
    log.info("holdout done: %s", [r["info"]["module"] for r in res])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", nargs="+", default=HOLDOUT)
    ap.add_argument("--skip-prep", action="store_true")
    a = ap.parse_args()
    main(a.modules, a.skip_prep)
