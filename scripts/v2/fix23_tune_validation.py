"""v2 Fixes 2 + 3 (post-hoc): tune relative candidate filters and the popularity term on VALIDATION only.

usage:
  python scripts/v2/fix23_tune_validation.py --part fix2    # A2-A7 with relative filters
  python scripts/v2/fix23_tune_validation.py --part fix3    # Pop definition, A8, A9
  python scripts/v2/fix23_tune_validation.py --part freeze  # write config/final_config_v2.yaml
Outputs: results/v2/validation/*, data/processed/v2/knowledge_state_train.parquet, config/final_config_v2.yaml.
"""
from __future__ import annotations

import argparse
import datetime as dt
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

from src.evaluation.ranking_metrics import ndcg_at_k, recall_at_k, student_mean  # noqa: E402
from src.models.knowledge_state import knowledge_state  # noqa: E402
from src.models.train import load_variant  # noqa: E402
from src.recommend import v2 as V2  # noqa: E402
from src.recommend.path import build_paths  # noqa: E402
from src.utils.io import get_logger, load_config, seed_everything, write_parquet  # noqa: E402

log = get_logger("v2_tune")
S4 = runpy.run_path(str(ROOT / "scripts" / "04_tune_recommender.py"), run_name="s4")
OUT = ROOT / "results" / "v2" / "validation"
PV2 = ROOT / "data" / "processed" / "v2"
V2CFG = {"modes": ["within", "cohort"], "q_grid": [10, 20, 30, 40, 50], "max_upcoming_removed": 0.10,
         "min_active_share": 0.01, "pop_options": ["train_window", "cohort_w7", "cohort_w14", "cohort_w30"]}
ALL5 = ["G", "Q", "F", "D", "I"]


def v2_configs(pop_key=None):
    c = {
        "A2": {"kind": "score", "label": "Mastery only", "model": "none", "active": ["G"], "graph": None, "use_q": False},
        "A3": {"kind": "score", "label": "Mastery + prerequisites", "model": "none", "active": ["G", "Q", "D"],
               "graph": "chain"},
        "A4": {"kind": "score", "label": "+ fixed forgetting", "model": "fixed", "active": ["G", "Q", "D", "F"],
               "graph": "chain"},
        "A5": {"kind": "score", "label": "+ learned forgetting", "model": "learned", "active": ["G", "Q", "D", "F"],
               "graph": "chain"},
        "A6": {"kind": "score", "label": "+ interest", "model": "learnedI", "active": ALL5, "graph": "chain"},
    }
    if pop_key:
        c["A8"] = {**c["A5"], "label": "A5 + popularity", "active": ["G", "Q", "D", "F", "Pop"], "pop": pop_key}
        c["A9"] = {**c["A6"], "label": "A6 + popularity", "active": ALL5 + ["Pop"], "pop": pop_key}
    return c


def build_rd(cfg, split, inp=None, data=None):
    inp = inp or S4["load_rec_inputs"](cfg, "default")
    ks = inp["ks"][inp["ks"]["split"] == split]
    rel = inp["rel"][inp["rel"]["split"] == split]
    topics = pd.read_parquet(ROOT / "data" / "processed" / "topics.parquet")
    rd = V2.RecDataV2(ks, rel, inp["graphs"], inp["models"], inp["pop"], topics, 6)
    return rd, inp


def train_knowledge_state(cfg):
    p = PV2 / "knowledge_state_train.parquet"
    if p.exists():
        return pd.read_parquet(p)
    data = load_variant(cfg)
    fitted = joblib.load(ROOT / "models" / "fitted_models.joblib")
    forget = {k: v for k, v in fitted.items() if k in {"none", "fixed", "learned", "learnedI"}}
    ks = knowledge_state(data["queries"], data["state"]["train"], forget, data["graphs"], data["tindex"],
                         data["diff"], kinds=("decision",))
    write_parquet(ks, p)
    return ks


def attach(rd, cfg, ks_train, vd_students=None):
    for mname in ["none", "fixed", "learned", "learnedI"]:
        rd.cohort_thr[mname] = V2.cohort_thresholds(ks_train, rd, mname, V2CFG["q_grid"])
    rd.pop_terms["train_window"] = V2.pop_train_window(rd)
    vd = pd.read_parquet(ROOT / "data" / "processed" / "vle_daily.parquet", columns=["sid", "topic", "date", "clicks"])
    stu = pd.read_parquet(ROOT / "data" / "processed" / "students.parquet")
    cd = V2.cohort_daily(vd, stu)
    for w in [7, 14, 30]:
        rd.pop_terms[f"cohort_w{w}"] = V2.pop_cohort_before(rd, cd, w)
    return rd


def pop_only_ranking(rd, key, seed=0):
    s = rd.pop_terms[key] + 1e-6 * np.random.default_rng(seed).random((rd.N, rd.T))
    P = build_paths(s, rd.universe, np.zeros_like(rd.universe), None, 5, max_rev=5)
    nd = float(np.nanmean(student_mean(ndcg_at_k(P, rd.relAB, 5), rd.sidx, rd.n_students)))
    rc = float(np.nanmean(student_mean(recall_at_k(P, rd.relAB, 5), rd.sidx, rd.n_students)))
    return nd, rc


def save(name, res):
    OUT.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(res.pop("history"))
    hist.to_csv(OUT / f"tuning_history_{name}.csv", index=False)
    json.dump(res, open(OUT / f"tuned_{name}.json", "w"), indent=2, default=float)


def main(part):
    cfg = load_config()
    seed_everything(cfg["seed"])
    PV2.mkdir(parents=True, exist_ok=True)
    if part in ("fix2", "fix3"):
        ks_train = train_knowledge_state(cfg)
        rd, _ = build_rd(cfg, "val")
        rd = attach(rd, cfg, ks_train)
        log.info("validation decision points %d", rd.N)
    if part == "fix2":
        for cid, conf in v2_configs().items():
            res = V2.tune_v2(rd, conf, cfg["recommender"], V2CFG, log)
            res["config"] = conf
            save(cid, res)
    elif part == "fix3":
        rows = []
        for key in V2CFG["pop_options"]:
            nd, rc = pop_only_ranking(rd, key, cfg["seed"])
            rows.append({"pop_option": key, "val_NDCG@5_pop_only": nd, "val_R@5_pop_only": rc})
        pt = pd.DataFrame(rows).sort_values(["val_NDCG@5_pop_only", "val_R@5_pop_only"], ascending=False)
        pt.to_csv(OUT / "pop_definition_selection.csv", index=False)
        key = pt.iloc[0]["pop_option"]
        log.info("Pop definition chosen on validation: %s\n%s", key, pt.to_string())
        for cid in ["A8", "A9"]:
            conf = v2_configs(key)[cid]
            res = V2.tune_v2(rd, conf, cfg["recommender"], V2CFG, log)
            res["config"] = conf
            save(cid, res)
    elif part == "freeze":
        tuned = {}
        for cid in ["A2", "A3", "A4", "A5", "A6", "A8", "A9"]:
            tuned[cid] = json.load(open(OUT / f"tuned_{cid}.json"))
        pop_key = tuned["A8"]["config"]["pop"]
        v1 = yaml.safe_load(open(ROOT / "config" / "final_config.yaml"))
        final = {
            "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
            "note": ("v2 (POST-HOC). All choices tuned on validation (DDD/FFF 2014B) only. Frozen before any holdout "
                     "or 2014J v2 scoring."),
            "knowledge_models": "unchanged from v1 (config/final_config.yaml); Fix 1 found no bug",
            "v2_filters": {k: V2CFG[k] for k in ["modes", "q_grid", "max_upcoming_removed", "min_active_share"]},
            "pop_definition": pop_key,
            "recommender": v1["recommender"],
            "ablations": {
                "A0": v1["ablations"]["A0"], "A1": {**v1["ablations"]["A1"], "note": "v1 A1, audited: no leakage"},
                **{c: {k: v for k, v in t.items()} for c, t in tuned.items()},
                "A7": {"identical_to": "A6", "note": "no data-derived edge qualifies at pass mark 40 (v1 #23)"},
            },
        }
        with open(ROOT / "config" / "final_config_v2.yaml", "w") as f:
            yaml.safe_dump(S4["to_plain"](final), f, sort_keys=False)
        log.info("wrote config/final_config_v2.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["fix2", "fix3", "freeze"])
    main(ap.parse_args().part)
