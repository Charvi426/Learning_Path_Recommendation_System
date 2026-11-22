"""Stage 4: recommender tuning on validation (METHODOLOGY section 6) and freezing config/final_config.yaml.

usage: python scripts/04_tune_recommender.py [--variant default]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.evaluation.ablation import ablation_configs, tune_config  # noqa: E402
from src.evaluation.relevance import relevance_table  # noqa: E402
from src.features.pipeline import variant_dir  # noqa: E402
from src.features.prerequisites import prereq_dict  # noqa: E402
from src.recommend.engine import RecData  # noqa: E402
from src.utils.io import ROOT, get_logger, load_config, path, seed_everything, write_parquet  # noqa: E402

log = get_logger("04_tune")


def load_rec_inputs(cfg, variant="default"):
    d = variant_dir(cfg, variant)
    proc = path(cfg, "processed")
    queries = pd.read_parquet(d / "queries.parquet")
    inter = pd.read_parquet(d / "interactions_variant.parquet")
    students = pd.read_parquet(proc / "students.parquet")
    edges = pd.read_csv(d / "prereq_edges.csv")
    graphs = {"chain": prereq_dict(edges[edges["source"] == "chain"]), "data": prereq_dict(edges)}
    has_data = bool((edges["source"] == "data").any())
    ks = pd.read_parquet(d / "knowledge_state.parquet")
    rel_path = d / "relevance.parquet"
    if rel_path.exists():
        rel = pd.read_parquet(rel_path)
    else:
        vd = pd.read_parquet(proc / "vle_daily.parquet", columns=["sid", "topic", "date", "clicks", "vol_clicks"])
        dp = queries[queries["kind"] == "decision"][["qid", "sid", "k", "t", "split", "code_module"]]
        rel = relevance_table(dp, inter, vd, int(students["sid"].max()) + 1, 6, cfg["recommender"]["rel_a_min_clicks"])
        write_parquet(rel, rel_path)
    # popularity (A1): mean training-cohort clicks per (module, k_now, topic) in the decision window
    tr = rel[rel["split"] == "train"]
    pop = []
    for j in range(1, 7):
        g = tr.groupby(["code_module", "k"])[f"wclicks_{j}"].mean().rename("pop_clicks").reset_index()
        g["j"] = j
        pop.append(g)
    pop = pd.concat(pop).rename(columns={"k": "k_now"})
    models = sorted({c[2:] for c in ks.columns if c.startswith("K_")})
    return {"queries": queries, "inter": inter, "students": students, "graphs": graphs, "has_data": has_data,
            "ks": ks, "rel": rel, "pop": pop, "models": models, "dir": d}


def recdata(inp, split):
    ks = inp["ks"][inp["ks"]["split"] == split]
    rel = inp["rel"][inp["rel"]["split"] == split]
    return RecData(ks, rel, inp["graphs"], inp["models"], inp["pop"])


def tune_all(cfg, variant="default", only=None):
    inp = load_rec_inputs(cfg, variant)
    rd = recdata(inp, "val")
    log.info("[%s] validation decision points: %d (students %d); groups %s", variant, rd.N, rd.n_students,
             pd.Series(rd.group).value_counts().to_dict())
    confs = ablation_configs(inp["has_data"])
    tuned, hist = {}, []
    for cid, conf in confs.items():
        if only and cid not in only:
            continue
        if conf["kind"] != "score":
            tuned[cid] = {"config": conf}
            continue
        if cid == "A7" and not inp["has_data"]:
            # no data-derived edges qualify: A7 is identical to A6 (DECISIONS.md #23)
            tuned[cid] = {"config": conf, "identical_to": "A6"}
            continue
        r = tune_config(rd, conf, cfg["recommender"], log)
        for h in r.pop("history"):
            hist.append({"config": cid, **h})
        tuned[cid] = {"config": conf, **r}
    if "A7" in tuned and tuned["A7"].get("identical_to") == "A6" and "A6" in tuned:
        a6 = {k: v for k, v in tuned["A6"].items() if k != "config"}
        tuned["A7"].update(a6)
        tuned["A7"]["config"] = {**confs["A7"], "model": "learnedI", "graph": "data"}
    return tuned, pd.DataFrame(hist), inp


def to_plain(o):
    if isinstance(o, dict):
        return {k: to_plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_plain(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="default")
    a = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    tuned, hist, inp = tune_all(cfg, a.variant)
    tabs = path(cfg, "tables")
    tag = "" if a.variant == "default" else f"_{a.variant}"
    hist.to_csv(tabs / f"recommender_tuning_val{tag}.csv", index=False)
    summ = [{"config": c, "label": t["config"]["label"], **t.get("thresholds", {}),
             **{f"w_{k}": v for k, v in t.get("weights", {}).items()},
             "val_NDCG@5": t.get("val_NDCG@5"), "val_R@5": t.get("val_R@5")} for c, t in tuned.items()]
    pd.DataFrame(summ).to_csv(tabs / f"recommender_tuned_val{tag}.csv", index=False)
    if a.variant == "default":
        models_summary = yaml.safe_load(open(ROOT / "models" / "train_summary.json"))
        final = {
            "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
            "note": "Frozen after validation tuning. Not to be changed after test-split evaluation.",
            "modules": cfg["modules"], "split": cfg["split"], "pass_mark": cfg["pass_mark"],
            "knowledge_models": {"fixed_S": models_summary["fixed_S"], "l2": models_summary["l2"],
                                 "das3h_windows": models_summary["das3h_windows"],
                                 "identification_narrow": models_summary["identification"]["narrow"],
                                 "main_forgetting_model": "learned (A5) / learnedI (A6)"},
            "recommender": {"K": cfg["recommender"]["K"], "max_revision": cfg["recommender"]["max_revision"],
                            "horizon": cfg["recommender"]["horizon"]},
            "ablations": {c: {k: v for k, v in t.items() if k != "history"} for c, t in tuned.items()},
        }
        with open(ROOT / "config" / "final_config.yaml", "w") as f:
            yaml.safe_dump(to_plain(final), f, sort_keys=False)
        log.info("wrote config/final_config.yaml")
    print(pd.DataFrame(summ).round(4).to_string())
