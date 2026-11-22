"""Stage 5: evaluation on the test split (METHODOLOGY section 7), using the frozen config/final_config.yaml.

Writes results/tables/{model_metrics_test, calibration_test, ranking_metrics_test, simulated_gain_test,
exam_retention_check, wilcoxon_holm, fairness, fairness_gaps, sensitivity}.csv, figures and sample paths.

usage: python scripts/05_evaluate.py [--skip-sensitivity]
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation import evaluate as EV  # noqa: E402
from src.evaluation.ablation import ALL, tune_config  # noqa: E402
from src.evaluation.stats import bootstrap_mean_ci, wilcoxon_holm  # noqa: E402
from src.models.train import designs, load_variant  # noqa: E402
from src.utils.io import ROOT, get_logger, load_config, path, seed_everything, write_json  # noqa: E402

log = get_logger("05_evaluate")
S4 = runpy.run_path(str(ROOT / "scripts" / "04_tune_recommender.py"), run_name="s4")
MAIN_COMPARISONS = [("A5", "A3", "forgetting: A5 vs A3"), ("A5", "A4", "learned vs fixed decay: A5 vs A4"),
                    ("A6", "A5", "interest: A6 vs A5")]
PAIRED_MODELS = [("learned", "none"), ("fixed", "none"), ("learned", "fixed"), ("learnedI", "learned"),
                 ("learned", "pfa"), ("learned", "das3h"), ("pfa", "topic_difficulty")]
MODEL_ORDER = ["majority", "topic_difficulty", "pfa", "none", "fixed", "learned", "learnedI", "das3h",
               "das3h_7_30_inf", "das3h_7_14_30_inf"]


def median_clicks(data):
    q = data["queries"]
    st = data["state"]["train"]
    due = q[(q["kind"] == "due") & (q["split"] == "train")][["qid", "k", "code_module"]]
    s = due.merge(st[["qid", "j", "clicks_raw_inf"]], left_on=["qid", "k"], right_on=["qid", "j"])
    return s.groupby("code_module")["clicks_raw_inf"].median().to_dict()


def wilcoxon_table(per_student, metric="NDCG@5[AB]"):
    pairs = {name: (per_student[a][metric], per_student[b][metric]) for a, b, name in MAIN_COMPARISONS}
    df = wilcoxon_holm(pairs)
    df.insert(1, "metric", metric)
    return df


def plot_ranking(rt, figpath):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, rel in zip(axes, ["AB", "A", "B"]):
        d = rt[(rt["metric"] == "NDCG@5") & (rt["relevance"] == rel)]
        x = np.arange(len(d))
        ax.bar(x, d["value"], yerr=[d["value"] - d["ci_lo"], d["ci_hi"] - d["value"]], capsize=3,
               color=["#bbbbbb", "#bbbbbb"] + ["#4c72b0"] * (len(d) - 2))
        ax.set_xticks(x)
        ax.set_xticklabels(d["config"])
        ax.set_title({"AB": "combined", "A": "Rel-A (passed: engaged topics)",
                      "B": "Rel-B (failed: needed topics)"}[rel], fontsize=9)
        ax.set_ylabel("NDCG@5 (mean over students, 95% CI)")
        lo = max(0, d["ci_lo"].min() - 0.05)
        ax.set_ylim(lo, min(1, d["ci_hi"].max() + 0.03))
    fig.suptitle("Test split (2014J): ranking quality of A0–A7", fontsize=10)
    fig.tight_layout()
    fig.savefig(figpath, dpi=150)
    plt.close(fig)


def plot_models(mt, figpath):
    d = mt[mt["model"] != "majority"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(d))
    ax.errorbar(d["AUC"], y, xerr=[d["AUC"] - d["AUC_lo"], d["AUC_hi"] - d["AUC"]], fmt="o", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=7)
    ax.set_xlabel("AUC on test split (95% CI, 1000 student resamples)")
    ax.set_title("Knowledge-model AUC (P(pass topic assessment))")
    fig.tight_layout()
    fig.savefig(figpath, dpi=150)
    plt.close(fig)


def plot_exam(agg, figpath, v="learned"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, col, lab in [(axes[0], f"meanM_{v}", "mean M (mastery, no decay)"),
                         (axes[1], f"meanK_{v}", "mean K (effective knowledge, learned decay)")]:
        ax.scatter(agg[col], agg["exam_score"], s=4, alpha=0.3)
        ax.set_xlabel(lab)
    axes[0].set_ylabel("final exam score")
    fig.suptitle("Exam-based retention check (test split)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figpath, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- sensitivity
def sens_configs(has_data):
    c = {
        "A3": {"kind": "score", "label": "Mastery + prerequisites", "model": "none", "active": ["G", "Q", "D"],
               "graph": "chain"},
        "A5": {"kind": "score", "label": "+ learned forgetting", "model": "learned", "active": ["G", "Q", "D", "F"],
               "graph": "chain"},
    }
    if has_data:
        c["A3_data"] = {**c["A3"], "label": "A3 with data-derived prerequisites", "model": "none_data", "graph": "data"}
        c["A5_data"] = {**c["A5"], "label": "A5 with data-derived prerequisites", "model": "learned_data",
                        "graph": "data"}
    return c


def run_sensitivity_variant(cfg, variant, n_boot, seed):
    s3 = runpy.run_path(str(ROOT / "scripts" / "03_train_models.py"), run_name="s3")
    s3["run"](cfg, variant, do_bootstrap=False)
    inp = S4["load_rec_inputs"](cfg, variant)
    rd_val = S4["recdata"](inp, "val")
    rd_test = S4["recdata"](inp, "test")
    confs = sens_configs(inp["has_data"])
    final = {"ablations": {}}
    for cid, conf in confs.items():
        r = tune_config(rd_val, conf, cfg["recommender"], log)
        r.pop("history")
        final["ablations"][cid] = {"config": conf, **r}
    with open(ROOT / "config" / f"final_config_sens_{variant}.yaml", "w") as f:
        yaml.safe_dump(S4["to_plain"](final), f, sort_keys=False)
    rt, ps, _ = EV.ranking_table(rd_test, final, cfg["recommender"], n_boot, seed)
    rows = []
    pairs = [("A5", "A3")] + ([("A5_data", "A3_data")] if inp["has_data"] else [])
    for a, b in pairs:
        w = wilcoxon_holm({f"{a} vs {b}": (ps[a]["NDCG@5[AB]"], ps[b]["NDCG@5[AB]"])})
        va = rt[(rt.config == a) & (rt.metric == "NDCG@5") & (rt.relevance == "AB")].iloc[0]
        vb = rt[(rt.config == b) & (rt.metric == "NDCG@5") & (rt.relevance == "AB")].iloc[0]
        rows.append({"sensitivity": variant, "comparison": f"{a} vs {b}", "NDCG@5_x": va["value"],
                     "x_ci": f"[{va['ci_lo']:.3f}, {va['ci_hi']:.3f}]", "NDCG@5_y": vb["value"],
                     "y_ci": f"[{vb['ci_lo']:.3f}, {vb['ci_hi']:.3f}]", "diff": va["value"] - vb["value"],
                     "p_wilcoxon": w["p_raw"].iloc[0], "x_greater_holds": bool(va["value"] > vb["value"]),
                     "significant_0.05": bool(w["p_raw"].iloc[0] < 0.05),
                     "n_data_edges": int((inp["dir"] / "prereq_edges.csv").exists() and
                                         (pd.read_csv(inp["dir"] / "prereq_edges.csv")["source"] == "data").sum())})
    rt.to_csv(path(cfg, "tables") / f"ranking_metrics_test_sens_{variant}.csv", index=False)
    return rows


def main(skip_sens=False, split="test"):
    cfg = load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    nb = cfg["evaluation"]["bootstrap"]
    tabs, figs = path(cfg, "tables"), path(cfg, "figures")
    if split != "test":  # dry run of the full code path on validation, written elsewhere
        tabs = tabs / f"dryrun_{split}"
        figs = figs / f"dryrun_{split}"
        tabs.mkdir(parents=True, exist_ok=True)
        figs.mkdir(parents=True, exist_ok=True)
        skip_sens = True
    final = yaml.safe_load(open(ROOT / "config" / "final_config.yaml"))
    data = load_variant(cfg, "default")
    fitted = joblib.load(ROOT / "models" / "fitted_models.joblib")
    D = designs(data, "chain")

    # 7.1 model metrics
    log.info("7.1 model metrics on test")
    mt, cal, preds = EV.model_metrics_test(fitted, D[split], nb, seed, figs / "calibration_test.png", MODEL_ORDER)
    mt.to_csv(tabs / "model_metrics_test.csv", index=False)
    cal.to_csv(tabs / "calibration_test.csv", index=False)
    plot_models(mt, figs / "model_auc_test.png")
    EV.paired_model_differences(preds, D[split], PAIRED_MODELS, nb, seed).to_csv(
        tabs / "model_paired_diffs_test.csv", index=False)

    # 7.2 ranking metrics
    log.info("7.2 ranking metrics on test")
    inp = S4["load_rec_inputs"](cfg, "default")
    rd = S4["recdata"](inp, split)
    log.info("test decision points %d, students %d, groups %s", rd.N, rd.n_students,
             pd.Series(rd.group).value_counts().to_dict())
    rt, per_student, paths = EV.ranking_table(rd, final, cfg["recommender"], nb, seed)
    rt.to_csv(tabs / "ranking_metrics_test.csv", index=False)
    wide = rt.pivot_table(index=["config", "label"], columns=["relevance", "metric"], values="value")
    wide.columns = [f"{m}[{r}]" for r, m in wide.columns]
    wide.reset_index().to_csv(tabs / "ranking_metrics_test_wide.csv", index=False)
    plot_ranking(rt, figs / "ranking_ndcg5_test.png")

    # 7.3 simulated gain (DAS3H simulator, model-based)
    log.info("7.3 simulated gain")
    med = median_clicks(data)
    sg, sg_ps = EV.simulated_gain(rd, paths, fitted["das3h"], data["state"][split], data["queries"],
                                  data["graphs"], data["tindex"], med, nb, seed)
    sg["median_clicks_added"] = json.dumps({k: float(v) for k, v in med.items()})
    sg.to_csv(tabs / "simulated_gain_test.csv", index=False)

    # 7.4 exam-based check
    log.info("7.4 exam check")
    ks = inp["ks"]
    ks_exam = ks[(ks["kind"] == "exam") & (ks["split"] == split)]
    ex, agg = EV.exam_check(ks_exam, data["students"], ["fixed", "learned", "learnedI"], nb, seed, cfg["pass_mark"])
    ex.to_csv(tabs / "exam_retention_check.csv", index=False)
    plot_exam(agg, figs / "exam_check_scatter.png")
    # supplementary: the same check on validation (not used for any decision)
    ks_exam_v = ks[(ks["kind"] == "exam") & (ks["split"] == "val")]
    exv, _ = EV.exam_check(ks_exam_v, data["students"], ["fixed", "learned", "learnedI"], nb, seed, cfg["pass_mark"])
    exv.to_csv(tabs / "exam_retention_check_val.csv", index=False)

    # 7.6 statistics
    log.info("7.6 Wilcoxon + Holm")
    wt = pd.concat([wilcoxon_table(per_student, m) for m in ["NDCG@5[AB]", "NDCG@5[A]", "NDCG@5[B]"]],
                   ignore_index=True)
    wt["family"] = np.where(wt["metric"] == "NDCG@5[AB]", "primary (combined relevance)", "supplementary")
    wt.to_csv(tabs / "wilcoxon_holm.csv", index=False)
    # simulated-gain paired tests (supplementary)
    sgw = wilcoxon_holm({name: (sg_ps[a], sg_ps[b]) for a, b, name in MAIN_COMPARISONS})
    sgw.to_csv(tabs / "wilcoxon_holm_simulated_gain.csv", index=False)

    # fairness
    log.info("fairness")
    fr, gaps = EV.fairness(rd, per_student, data["students"], preds, D[split])
    fr.to_csv(tabs / "fairness.csv", index=False)
    gaps.to_csv(tabs / "fairness_gaps.csv", index=False)

    # sample paths (6.5)
    for cid in ["A6", "A5"]:
        sp = EV.sample_paths(rd, paths, cid, ks, 10, seed)
        out = path(cfg, "sample_paths") if split == "test" else tabs
        write_json(sp, out / f"sample_paths_{cid}.json")
        W = paths[cid][3]
        (out / f"sample_paths_{cid}.md").write_text(EV.sample_paths_markdown(sp, W, cid), encoding="utf-8")

    # sensitivity
    sens = []
    # windows: the window set only enters the DAS3H benchmark/simulator
    for sim_name in ["das3h_7_30_inf", "das3h_7_14_30_inf"]:
        sgi, sgi_ps = EV.simulated_gain(rd, {k: paths[k] for k in ["A3", "A5"]}, fitted[sim_name],
                                        data["state"][split], data["queries"], data["graphs"], data["tindex"],
                                        med, nb, seed)
        w = wilcoxon_holm({"A5 vs A3": (sgi_ps["A5"], sgi_ps["A3"])})
        a5 = sgi[sgi.config == "A5"].iloc[0]
        a3 = sgi[sgi.config == "A3"].iloc[0]
        auc = mt.loc[mt["model"] == sim_name, "AUC"].iloc[0]
        sens.append({"sensitivity": f"windows {sim_name[6:]} (simulator)", "comparison": "A5 vs A3 (simulated gain)",
                     "NDCG@5_x": np.nan, "NDCG@5_y": np.nan, "gain_x": a5["mean_gain"], "gain_y": a3["mean_gain"],
                     "diff": a5["mean_gain"] - a3["mean_gain"], "p_wilcoxon": w["p_raw"].iloc[0],
                     "x_greater_holds": bool(a5["mean_gain"] > a3["mean_gain"]),
                     "significant_0.05": bool(w["p_raw"].iloc[0] < 0.05), "simulator_test_AUC": auc,
                     "note": "A3/A5 rankings do not use windows; only the simulator changes"})
    base = rt[(rt.metric == "NDCG@5") & (rt.relevance == "AB")].set_index("config")
    w0 = wt[(wt.metric == "NDCG@5[AB]") & (wt.comparison.str.startswith("forgetting"))].iloc[0]
    sens.insert(0, {"sensitivity": "default (pass 40, non-submission excluded, chain)", "comparison": "A5 vs A3",
                    "NDCG@5_x": base.loc["A5", "value"], "NDCG@5_y": base.loc["A3", "value"],
                    "diff": base.loc["A5", "value"] - base.loc["A3", "value"], "p_wilcoxon": w0["p_raw"],
                    "x_greater_holds": bool(base.loc["A5", "value"] > base.loc["A3", "value"]),
                    "significant_0.05": bool(w0["p_raw"] < 0.05),
                    "note": "chain = data graph here (no data-derived edge qualifies at pass mark 40)"})
    if not skip_sens:
        for v in ["pass70", "nonsub_fail"]:
            log.info("sensitivity variant %s", v)
            try:
                sens += run_sensitivity_variant(cfg, v, nb, seed)
            except Exception as e:  # never stop the run; record the failure
                log.exception("sensitivity %s failed", v)
                sens.append({"sensitivity": v, "comparison": "A5 vs A3", "note": f"FAILED: {e!r}"})
    pd.DataFrame(sens).to_csv(tabs / "sensitivity.csv", index=False)
    log.info("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--dry-run-val", action="store_true", help="run every step on validation (debugging)")
    a = ap.parse_args()
    main(a.skip_sensitivity, "val" if a.dry_run_val else "test")
