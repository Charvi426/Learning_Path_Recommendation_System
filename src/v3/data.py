"""v3 data access for a 'course set': DDD/FFF (v1 processed data) or one v2 holdout module."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..features.prerequisites import prereq_dict
from ..models.data import topic_index_map
from ..utils.io import ROOT
from . import attribution as A

HOLDOUT = ["AAA", "BBB", "CCC", "EEE", "GGG"]
ATTRIBUTIONS = ["direct", "coaccess", "shared_0.4", "shared_0.6", "shared_0.8"]


def dirs(cs: str) -> dict:
    if cs == "DDDFFF":
        return {"proc": ROOT / "data/processed", "svle": ROOT / "data/interim/student_vle_capped.parquet",
                "cache": ROOT / "data/processed/v3/DDDFFF"}
    return {"proc": ROOT / f"data/processed/v2_holdout/{cs}",
            "svle": ROOT / f"data/interim/v2_holdout/{cs}/student_vle_capped.parquet",
            "cache": ROOT / f"data/processed/v3/{cs}"}


def load(cs: str) -> dict:
    d = dirs(cs)
    p = d["proc"]
    data = {"cs": cs, "dirs": d,
            "queries": pd.read_parquet(p / "queries.parquet"),
            "inter": pd.read_parquet(p / "interactions_variant.parquet"),
            "topics": pd.read_parquet(p / "topics.parquet"),
            "students": pd.read_parquet(p / "students.parquet"),
            "tmap": pd.read_parquet(p / "topic_map.parquet"),
            "edges": pd.read_csv(p / "prereq_edges.csv"),
            "diff": pd.read_csv(p / "topic_difficulty.csv"),
            "state": {sp: pd.read_parquet(p / f"state_{sp}.parquet") for sp in ["train", "val", "test"]
                      if (p / f"state_{sp}.parquet").exists()}}
    data["tindex"] = topic_index_map(data["topics"][["code_module", "topic"]])
    chain = prereq_dict(data["edges"][data["edges"]["source"] == "chain"])
    data["graphs"] = {"chain": chain, "data": chain}
    data["T"] = int(data["topics"]["topic"].max())
    return data


def classes(cs: str, data: dict) -> pd.DataFrame:
    rc = pd.read_csv(ROOT / "results/v3/resource_classes.csv")
    mods = data["topics"]["code_module"].unique()
    return rc[rc["code_module"].isin(mods)]


def practice(cs: str, data: dict, attribution: str) -> pd.DataFrame:
    """Cached daily attributed practice (sid, topic, date, dose)."""
    d = dirs(cs)
    d["cache"].mkdir(parents=True, exist_ok=True)
    f = d["cache"] / f"practice_{attribution}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    sv = pd.read_parquet(d["svle"], columns=["code_module", "code_presentation", "id_student", "id_site", "date",
                                             "clicks"])
    rc = classes(cs, data)
    mode, H = (attribution.split("_") + [None])[:2]
    static, shared = A.site_weights(sv, data["topics"], data["tmap"], rc, mode, float(H) if H else 0.6)
    pr = A.practice_doses(sv, data["students"], data["topics"], static, shared, rc)
    pr.to_parquet(f, index=False)
    return pr
