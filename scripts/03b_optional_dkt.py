"""Optional DKT baseline via pyKT (METHODOLOGY 5.4). Requires torch + pykt-toolkit (installed with --no-deps,
see README). Trains on train, early-stops on validation AUC, evaluates once on test, and re-scores the other
models on exactly the same rows (steps with >= 1 earlier response).

usage: python scripts/03b_optional_dkt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.model_metrics import all_metrics  # noqa: E402
from src.evaluation.stats import bootstrap_rows_by_student  # noqa: E402
from src.models.dkt_pykt import predict_dkt, train_dkt  # noqa: E402
from src.models.train import designs, load_variant  # noqa: E402
from src.utils.io import ROOT, load_config, path, seed_everything, write_json  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    seed_everything(cfg["seed"])
    data = load_variant(cfg)
    model, maxlen, info = train_dkt(data["inter"], data["tindex"], seed=cfg["seed"])
    print(info)
    fitted = joblib.load(ROOT / "models" / "fitted_models.joblib")
    D = designs(data, "chain")
    rows = []
    for split in ["val", "test"]:
        pd_ = predict_dkt(model, maxlen, data["inter"], data["tindex"], split)
        r = D[split].rows[["sid", "k", "y", "y_score"]].rename(columns={"k": "topic"}).reset_index()
        m = r.merge(pd_, on=["sid", "topic"], how="inner")
        idx = m["index"].to_numpy()
        y = m["y"].to_numpy()
        preds = {"dkt": m["p_dkt"].to_numpy()}
        for name in ["pfa", "das3h", "none", "learned", "learnedI"]:
            preds[name] = fitted[name].predict_proba(D[split])[idx]
        for name, p in preds.items():
            met = all_metrics(y, p)
            est, lo, hi = bootstrap_rows_by_student(roc_auc_score, m["sid"].to_numpy(), y, p,
                                                    n_boot=cfg["evaluation"]["bootstrap"], seed=cfg["seed"])
            rows.append({"split": split, "model": name, "rows": "steps with >= 1 earlier response",
                         "n_rows": len(y), **met, "AUC_lo": lo, "AUC_hi": hi})
    df = pd.DataFrame(rows)
    df.to_csv(path(cfg, "tables") / "dkt_comparison.csv", index=False)
    write_json(info, path(cfg, "tables") / "dkt_training.json")
    print(df.round(4).to_string())
