"""v2: evaluate v1 and v2 frozen configurations on DDD/FFF.

  --split val    validation comparison (v1 vs v2), used for the SUMMARY_v2 comparison table
  --split test   2014J, POST-HOC / EXPLORATORY: run ONCE, after config/final_config_v2.yaml is frozen
Knowledge models are v1's (unchanged; Fix 1 found no bug). Outputs: results/v2/validation/eval/ or
results/v2/posthoc_2014J/.
"""
from __future__ import annotations

import argparse
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

from src.evaluation import evaluate as EV  # noqa: E402
from src.evaluation.stats import bootstrap_mean_ci, wilcoxon_holm  # noqa: E402
from src.models.train import load_variant  # noqa: E402
from src.recommend import v2 as V2  # noqa: E402
from src.recommend.engine import evaluate_paths  # noqa: E402
from src.utils.io import get_logger, load_config, seed_everything  # noqa: E402

log = get_logger("v2_eval")
T23 = runpy.run_path(str(ROOT / "scripts" / "v2" / "fix23_tune_validation.py"), run_name="t23")
V1_CONFIGS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
V2_CONFIGS = ["A1", "A2", "A3", "A4", "A5", "A6", "A8", "A9"]
COMPARISONS = [("v2:A5", "v2:A3", "forgetting: A5 vs A3 (v2)"), ("v2:A8", "v2:A1", "popularity term: A8 vs A1 (v2)"),
               ("v2:A9", "v2:A8", "interest: A9 vs A8 (v2)")]


def main(split):
    cfg = load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    nb = cfg["evaluation"]["bootstrap"]
    out = ROOT / "results" / "v2" / ("validation/eval" if split == "val" else "posthoc_2014J")
    out.mkdir(parents=True, exist_ok=True)
    v1final = yaml.safe_load(open(ROOT / "config" / "final_config.yaml"))
    v2final = yaml.safe_load(open(ROOT / "config" / "final_config_v2.yaml"))
    ks_train = T23["train_knowledge_state"](cfg)
    rd, inp = T23["build_rd"](cfg, split)
    rd = T23["attach"](rd, cfg, ks_train)
    rc = cfg["recommender"]
    paths, diag = {}, {}
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
    for cid in ["A3", "A5", "A6"]:   # v1 filter diagnostics for comparison
        P, aux, conf, W = paths[f"v1:{cid}"]
        diag[f"v1:{cid}"] = V2.filter_diagnostics(rd, aux["new"], aux["rev"], np.zeros_like(aux["new"]), P)
    per_student, rows = {}, []
    for name, (P, aux, conf, W) in paths.items():
        (_, ps), = evaluate_paths(rd, P[None], rels=("A", "B", "AB"))
        per_student[name] = ps
        for met, v in ps.items():
            est, lo, hi = bootstrap_mean_ci(v, nb, seed)
            b, rel = met.split("[")
            rows.append({"config": name, "label": conf["label"], "relevance": rel.rstrip("]"), "metric": b,
                         "value": est, "ci_lo": lo, "ci_hi": hi, "n_students": int(np.isfinite(v).sum())})
    pd.DataFrame(rows).to_csv(out / "ranking_metrics.csv", index=False)
    pd.DataFrame([{"config": k, **v} for k, v in diag.items()]).to_csv(out / "filter_diagnostics.csv", index=False)
    wt = pd.concat([wilcoxon_holm({n: (per_student[a][m], per_student[b][m]) for a, b, n in COMPARISONS}).assign(metric=m)
                    for m in ["NDCG@5[AB]", "NDCG@5[A]", "NDCG@5[B]"]], ignore_index=True)
    wt.to_csv(out / "wilcoxon_holm.csv", index=False)
    wilcoxon_holm({f"v2 {c} vs v1 {c}": (per_student[f"v2:{c}"]["NDCG@5[AB]"], per_student[f"v1:{c}"]["NDCG@5[AB]"])
                   for c in ["A3", "A5", "A6"]}).to_csv(out / "wilcoxon_holm_v1_vs_v2.csv", index=False)
    # simulated gain with the v1 DAS3H simulator (unchanged)
    data = load_variant(cfg)
    fitted = joblib.load(ROOT / "models" / "fitted_models.joblib")
    med = runpy.run_path(str(ROOT / "scripts" / "05_evaluate.py"), run_name="s5")["median_clicks"](data)
    sg, _ = EV.simulated_gain(rd, paths, fitted["das3h"], data["state"][split], data["queries"], data["graphs"],
                              data["tindex"], med, nb, seed)
    sg.to_csv(out / "simulated_gain.csv", index=False)
    json.dump({"split": split, "decision_points": int(rd.N), "students": int(rd.n_students),
               "label": "validation" if split == "val" else "2014J POST-HOC / EXPLORATORY (scored once with frozen v2)"},
              open(out / "run_info.json", "w"), indent=2)
    log.info("done %s", split)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    main(ap.parse_args().split)
