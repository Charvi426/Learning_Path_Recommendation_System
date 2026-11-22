"""v3 Step 3 (post-hoc, VALIDATION ONLY): memory-model variants V0-V4 on DDD/FFF.

Coefficients are fitted on train; decay hyper-parameters (d or a, c, and the submission dose x_sub), the attribution
rule and H are chosen on validation log-loss. The DDD 2014B exam Δρ is computed for each final variant.
Selection rule (preregistered): lowest validation within-course log-loss among variants with exam Δρ >= 0; if none,
the highest exam Δρ among variants within 0.002 of the best validation log-loss.
Writes results/v3/variant_comparison_val.csv, results/v3/variant_tuning_val.csv, results/v3/selected_variant.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.io import get_logger, load_config, seed_everything  # noqa: E402
from src.v3 import pipeline as PL  # noqa: E402
from src.v3.data import load, practice  # noqa: E402
from src.v3.memory import Series  # noqa: E402

warnings.filterwarnings("ignore")
log = get_logger("v3_step3")
BC = ["coaccess", "shared_0.4", "shared_0.6", "shared_0.8"]


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    v1 = joblib.load(ROOT / "models/fitted_models.joblib")
    data = load("DDDFFF")
    Dt, Dv = PL.due_design(data, "train"), PL.due_design(data, "val")
    q = data["queries"]
    exam_q = q[(q["kind"] == "exam") & (q["split"] == "val")]
    inter = data["inter"]
    keys = inter[(inter["is_submitted"] == 1) & inter["split"].isin(["train", "val"])][["sid", "topic"]]
    series = {}

    def S(att):
        if att not in series:
            series[att] = Series(practice("DDDFFF", data, att), inter, keys)
        return series[att]

    tune, final = [], {}
    # V0: v2 learned half-life (reference, exact v1/v2 model)
    final["V0"] = {"kind": "hl_v2", "attribution": "direct (v1 features)", "params": {}}
    # V1: learned half-life on attributed practice, best of (b)/(c)
    best = None
    for att in BC:
        r = PL.evaluate_variant(PL.Variant("hl_att", S(att), l2=v1["learned"].l2, S=v1["fixed"].S), data, Dt, Dv)
        tune.append({"variant": "V1", "attribution": att, **r})
        log.info("V1 %s %s", att, r)
        if best is None or r["val_logloss"] < best[0]:
            best = (r["val_logloss"], att)
    final["V1"] = {"kind": "hl_att", "attribution": best[1], "params": {}}
    # V2: ACT-R fixed d, direct
    p2, ll2, n2 = PL.tune_actr("fixed", S("direct"), data, Dt, Dv, log)
    tune.append({"variant": "V2", "attribution": "direct", "rule": "fixed", "val_logloss": ll2, "params": json.dumps(p2),
                 "outer_evals": n2})
    final["V2"] = {"kind": "actr", "attribution": "direct", "params": p2}
    # V3: ACT-R, attribution chosen with the fixed rule, then fixed vs spacing
    best = None
    for att in BC:
        p, ll, n = PL.tune_actr("fixed", S(att), data, Dt, Dv, log)
        tune.append({"variant": "V3", "attribution": att, "rule": "fixed", "val_logloss": ll, "params": json.dumps(p),
                     "outer_evals": n})
        if best is None or ll < best[0]:
            best = (ll, att, p)
    att3 = best[1]
    ps, lls, ns = PL.tune_actr("spacing", S(att3), data, Dt, Dv, log)
    tune.append({"variant": "V3", "attribution": att3, "rule": "spacing", "val_logloss": lls, "params": json.dumps(ps),
                 "outer_evals": ns})
    final["V3"] = {"kind": "actr", "attribution": att3, "params": ps if lls < best[0] else best[2]}
    # V4: DAS3H windows on attributed practice, best of (b)/(c)
    best = None
    for att in BC:
        r = PL.evaluate_variant(PL.Variant("das3h_att", S(att), C=v1["das3h"].C), data, Dt, Dv)
        tune.append({"variant": "V4", "attribution": att, **r})
        if best is None or r["val_logloss"] < best[0]:
            best = (r["val_logloss"], att)
    final["V4"] = {"kind": "das3h_att", "attribution": best[1], "params": {}}
    pd.DataFrame(tune).to_csv(ROOT / "results/v3/variant_tuning_val.csv", index=False)
    # final comparison incl. DDD validation exam Δρ
    rows = []
    for vid, f in final.items():
        if f["kind"] == "hl_v2":
            var = PL.Variant("hl_v2", hl_model=v1["learned"])
        else:
            var = PL.Variant(f["kind"], S(f["attribution"]), f["params"], l2=v1["learned"].l2, S=v1["fixed"].S,
                             C=v1["das3h"].C)
        r = PL.evaluate_variant(var, data, Dt, Dv, exam_q)
        rows.append({"variant": vid, "kind": f["kind"], "attribution": f["attribution"], "params": json.dumps(f["params"]),
                     **r})
        log.info("final %s %s", vid, rows[-1])
    comp = pd.DataFrame(rows)
    # selection rule
    ok = comp[comp["exam_delta_rho"] >= 0]
    if len(ok):
        sel = ok.sort_values("val_logloss").iloc[0]
        why = "lowest validation log-loss among variants with DDD-validation exam Δρ >= 0"
    else:
        best_ll = comp["val_logloss"].min()
        near = comp[comp["val_logloss"] <= best_ll + 0.002]
        sel = near.sort_values("exam_delta_rho", ascending=False).iloc[0]
        why = "no variant reached Δρ >= 0: highest exam Δρ among variants within 0.002 of the best validation log-loss"
    comp["selected"] = comp["variant"] == sel["variant"]
    comp.to_csv(ROOT / "results/v3/variant_comparison_val.csv", index=False)
    out = {"selected": sel["variant"], "rule_applied": why, **{k: final[sel["variant"]][k] for k in ["kind", "attribution", "params"]},
           "frozen_hyperparameters": {"l2": v1["learned"].l2, "l2_interest": v1["learnedI"].l2, "S": v1["fixed"].S,
                                      "das3h_C": v1["das3h"].C}}
    json.dump(out, open(ROOT / "results/v3/selected_variant.json", "w", encoding="utf-8"), indent=2, default=float, ensure_ascii=True)
    print(comp.round(4).to_string())
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
