"""v3 Step 1 (post-hoc): classify VLE resources into NAVIGATION and CONTENT (src/v3/attribution.py).
Writes results/v3/resource_classes.csv (all 7 modules; click timing only, no outcomes)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from src.v3.attribution import classify_resources  # noqa: E402

SOURCES = {"DDD_FFF": ROOT / "data/interim/student_vle_capped.parquet",
           **{m: ROOT / f"data/interim/v2_holdout/{m}/student_vle_capped.parquet" for m in ["AAA", "BBB", "CCC", "EEE", "GGG"]}}


def main():
    vle = pd.read_csv(ROOT / "data/raw/vle.csv")
    out = []
    for name, p in SOURCES.items():
        sv = pd.read_parquet(p, columns=["code_module", "code_presentation", "id_site", "date", "clicks"])
        out.append(classify_resources(sv, vle))
    rc = pd.concat(out, ignore_index=True).sort_values(["code_module", "code_presentation", "id_site"])
    (ROOT / "results/v3").mkdir(parents=True, exist_ok=True)
    rc.round(6).to_csv(ROOT / "results/v3/resource_classes.csv", index=False)
    s = rc.groupby(["code_module", "resource_class"]).agg(n_sites=("id_site", "size"), clicks=("cohort_clicks", "sum"))
    s["share_clicks"] = s["clicks"] / s.groupby(level=0)["clicks"].transform("sum")
    s.round(4).to_csv(ROOT / "results/v3/resource_classes_summary.csv")
    print(s.round(3).to_string())
    print(rc[rc.resource_class == "NAVIGATION"].groupby("activity_type").size())


if __name__ == "__main__":
    main()
