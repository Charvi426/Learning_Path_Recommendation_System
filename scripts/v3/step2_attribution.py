"""v3 Step 2 (post-hoc): attributed practice for DDD/FFF under each attribution rule, plus diagnostics on the
DDD VALIDATION exam-takers (where does revision-period practice go?). Writes results/v3/attribution_*.csv."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.v3 import attribution as A  # noqa: E402
from src.v3.data import ATTRIBUTIONS, classes, load, practice  # noqa: E402


def main():
    data = load("DDDFFF")
    sv = pd.read_parquet(ROOT / "data/interim/student_vle_capped.parquet",
                         columns=["code_module", "code_presentation", "id_student", "id_site", "date", "clicks"])
    rc = classes("DDDFFF", data)
    wrows = []
    for att in ATTRIBUTIONS:
        mode, H = (att.split("_") + [None])[:2]
        static, shared = A.site_weights(sv, data["topics"], data["tmap"], rc, mode, float(H) if H else 0.6)
        wsum = static.groupby("id_site")["w"].sum()
        wrows.append({"attribution": att, "n_static_sites": int(static["id_site"].nunique()),
                      "n_shared_sites": int(len(shared)), "min_weight": float(static["w"].min()),
                      "max_abs_sum_minus_1": float((wsum - 1).abs().max()),
                      "mean_topics_per_static_site": float(static.groupby("id_site").size().mean())})
    pd.DataFrame(wrows).to_csv(ROOT / "results/v3/attribution_weights_summary.csv", index=False)
    # revision-period diagnostics on DDD validation exam-takers
    stu = data["students"]
    ex = stu[(stu["split"] == "val") & stu["exam_score"].notna()][["sid", "code_module", "code_presentation",
                                                                    "t_exam"]]
    tp = data["topics"]
    rows = []
    for att in ATTRIBUTIONS:
        pr = practice("DDDFFF", data, att)
        p = pr.merge(ex, on="sid")
        last_due = tp.groupby(["code_module", "code_presentation"])["t_due"].max().rename("last_due").reset_index()
        p = p.merge(last_due, on=["code_module", "code_presentation"])
        rev = p[(p["date"] > p["last_due"]) & (p["date"] < p["t_exam"])]
        tot = rev.groupby("topic")["dose"].sum()
        for j in range(1, 7):
            stud = rev[rev["topic"] == j]["sid"].nunique()
            rows.append({"attribution": att, "topic": j, "share_of_revision_practice": float(tot.get(j, 0) / tot.sum()),
                         "share_examtakers_with_revision_practice": stud / len(ex),
                         "share_examtakers_with_dose_ge_log1p3": rev[(rev["topic"] == j) & (rev["dose"] >= np.log1p(3))]
                         ["sid"].nunique() / len(ex)})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/v3/attribution_revision_diagnostics_val.csv", index=False)
    print(pd.DataFrame(wrows).to_string())
    print(d.pivot_table(index="topic", columns="attribution", values="share_examtakers_with_dose_ge_log1p3").round(3))


if __name__ == "__main__":
    main()
