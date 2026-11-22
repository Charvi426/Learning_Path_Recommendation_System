"""Stage-3 orchestration: identification check, baselines, PFA, forgetting variants, DAS3H benchmark,
bootstrap intervals. Hyper-parameters are tuned on validation only; the test split is not touched here."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..evaluation.model_metrics import all_metrics, log_loss  # noqa: E402
from ..features.pipeline import variant_dir  # noqa: E402
from ..features.prerequisites import prereq_dict  # noqa: E402
from ..utils.io import get_logger, path, write_json  # noqa: E402
from .das3h import ForgettingPFA, Majority, SklearnLogit  # noqa: E402
from .data import build_design, topic_index_map, training_targets  # noqa: E402

log = get_logger("train")
C_GRID = [0.01, 0.1, 1.0, 10.0]
FORGET_VARIANTS = {  # name: (forgetting, interest)
    "none": ("none", False),
    "fixed": ("fixed", False),
    "learned": ("learned", False),
    "learnedI": ("learned", True),
}


def load_variant(cfg, variant="default"):
    d = variant_dir(cfg, variant)
    proc = path(cfg, "processed")
    data = {
        "queries": pd.read_parquet(d / "queries.parquet"),
        "inter": pd.read_parquet(d / "interactions_variant.parquet"),
        "edges": pd.read_csv(d / "prereq_edges.csv"),
        "topics": pd.read_parquet(proc / "topics.parquet"),
        "students": pd.read_parquet(proc / "students.parquet"),
        "diff": pd.read_csv(d / "topic_difficulty.csv"),
        "dir": d,
    }
    data["state"] = {sp: pd.read_parquet(d / f"state_{sp}.parquet") for sp in ["train", "val", "test"]}
    data["tindex"] = topic_index_map(data["topics"][["code_module", "topic"]])
    data["graphs"] = {"chain": prereq_dict(data["edges"][data["edges"]["source"] == "chain"]),
                      "data": prereq_dict(data["edges"])}
    return data


def designs(data, graph="chain", windows=(7, 14, 30, np.inf)):
    out = {}
    for sp in ["train", "val", "test"]:
        tg = training_targets(data["queries"], data["inter"], sp)
        out[sp] = build_design(tg, data["state"][sp], data["graphs"][graph], data["tindex"], windows)
    return out


# ---------------------------------------------------------------------------------------------
def identification_check(D_train, figdir: Path, tabdir: Path, tag="") -> dict:
    p = D_train.pairs[D_train.pairs["assessed"]].copy()
    p["k"] = D_train.rows["k"].to_numpy()[p["row"].to_numpy()]
    p["module"] = D_train.rows["code_module"].to_numpy()[p["row"].to_numpy()]
    p["revised"] = p["delta"] < (D_train.rows["t"].to_numpy()[p["row"].to_numpy()] - p["t_submit_j"])
    within = p["delta"] - p.groupby(["module", "k"])["delta"].transform("mean")
    per_topic = p.groupby(["module", "k"])["delta"].describe()[["count", "mean", "std", "25%", "50%", "75%"]]
    per_topic["iqr"] = per_topic["75%"] - per_topic["25%"]
    per_topic["share_revised"] = p.groupby(["module", "k"])["revised"].mean()
    per_topic.round(3).to_csv(tabdir / f"delta_by_topic{tag}.csv")
    stats = {
        "n_pairs": int(len(p)), "mean": float(p["delta"].mean()), "sd": float(p["delta"].std()),
        "q": {str(q): float(p["delta"].quantile(q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
        "within_topic_sd": float(within.std()),
        "share_variance_within_topic": float(within.var() / p["delta"].var()),
        "median_within_topic_iqr": float(per_topic["iqr"].median()),
        "share_pairs_revised_after_submission": float(p["revised"].mean()),
    }
    # decision rule (fixed before fitting): narrow if the median within-topic IQR < 7 days
    stats["narrow"] = bool(stats["median_within_topic_iqr"] < 7)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(p["delta"], bins=np.arange(0, p["delta"].max() + 3, 2), color="#4c72b0")
    axes[0].set_xlabel("Δ (days since last meaningful practice of prerequisite) at the due date")
    axes[0].set_ylabel("training pairs")
    axes[0].set_title(f"Δ distribution (n={len(p)}, median={stats['q']['0.5']:.0f}, SD={stats['sd']:.1f})")
    for (m, k), g in p.groupby(["module", "k"]):
        axes[1].hist(g["delta"], bins=np.arange(0, 120, 3), histtype="step", label=f"{m} k={k}")
    axes[1].set_xlabel("Δ (days)"); axes[1].set_ylabel("pairs")
    axes[1].set_title(f"by target topic (median within-topic IQR = {stats['median_within_topic_iqr']:.0f} d)")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / f"delta_hist{tag}.png", dpi=150)
    plt.close(fig)
    return stats


def tune_sklearn(kind, D, windows=(7, 30, np.inf)):
    best = None
    for C in C_GRID:
        m = SklearnLogit(kind, C=C, windows=windows).fit(D["train"])
        ll = log_loss(D["val"].rows["y"], m.predict_proba(D["val"]))
        if best is None or ll < best[0]:
            best = (ll, C, m)
    return best


def fit_forgetting(name, D, data, l2, S=30.0, init_from=None):
    forgetting, interest = FORGET_VARIANTS[name]
    m = ForgettingPFA(forgetting, interest, S=S, l2=l2, n_topics=len(data["tindex"]))
    init = None
    if init_from is not None:
        # initialise theta / beta from the no-forgetting fit (Methodology 5.1) + several phi0 starts
        m._fit_scalers(D["train"])
        P = m._prep(D["train"])
        m.sl_, npar = m._layout(P["X"].shape[1], P["Z"].shape[1])
        src = init_from
        w = np.zeros(npar)
        w[m.sl_["beta"]] = src.w_[src.sl_["beta"]]
        nt = min(src.sl_["theta"].stop - src.sl_["theta"].start, m.sl_["theta"].stop - m.sl_["theta"].start)
        w[m.sl_["theta"].start:m.sl_["theta"].start + nt] = src.w_[src.sl_["theta"].start:src.sl_["theta"].start + nt]
        w[m.sl_["thK"]] = src.w_[src.sl_["thK"]]
        starts = []
        for g0 in [np.log2(7), np.log2(30), np.log2(120)]:
            w2 = w.copy()
            if forgetting == "learned":
                w2[m.sl_["phi"].start] = g0
            starts.append(w2)
        best = None
        for w0 in starts:
            mm = ForgettingPFA(forgetting, interest, S=S, l2=l2, n_topics=len(data["tindex"])).fit(D["train"], init=w0)
            if best is None or mm.train_loss_ < best.train_loss_:
                best = mm
        return best
    return m.fit(D["train"], init=init)


def tune_forgetting(name, D, data, cfg, init_from=None):
    grid_S = cfg["model"]["fixed_S_grid"] if name == "fixed" else [30.0]
    res = []
    best = None
    for S in grid_S:
        for l2 in cfg["model"]["l2_grid"]:
            m = fit_forgetting(name, D, data, l2, S, init_from)
            ll = log_loss(D["val"].rows["y"], m.predict_proba(D["val"]))
            res.append({"model": name, "S": S if name == "fixed" else np.nan, "l2": l2, "val_logloss": ll,
                        "converged": m.converged_})
            if best is None or ll < best[0]:
                best = (ll, m)
    return best[1], pd.DataFrame(res)


def _boot_one(b, name, D_train, base: ForgettingPFA, sids, seed):
    rng = np.random.default_rng(seed + b)
    uniq = np.unique(sids)
    draw = rng.choice(uniq, size=len(uniq), replace=True)
    cnt = pd.Series(draw).value_counts()
    sw = pd.Series(sids).map(cnt).fillna(0).to_numpy(float)
    m = ForgettingPFA(base.forgetting, base.interest, S=base.S, l2=base.l2, n_topics=base.n_topics)
    m.fit(D_train, init=base.w_.copy(), sample_weight=sw, scalers_from=base)
    return m.w_


def bootstrap(name, D_train, base, n_boot, seed, n_jobs=6):
    sids = D_train.rows["sid"].to_numpy()
    W = Parallel(n_jobs=n_jobs)(delayed(_boot_one)(b, name, D_train, base, sids, seed) for b in range(n_boot))
    return np.vstack(W)


def decay_figure(models: dict, D_train, figpath: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    days = np.arange(0, 181)
    ax.plot(days, np.ones_like(days), color="grey", ls=":", label="no forgetting")
    fx = models["fixed"]
    ax.plot(days, np.exp(-days / fx.S), color="k", ls="--", label=f"fixed S = {fx.S:.0f} d")
    for name, col in [("learned", "#4c72b0"), ("learnedI", "#dd8452")]:
        m = models[name]
        pr = D_train.pairs[D_train.pairs["assessed"]]
        _, h = m.retention_state(pr)
        for q, ls in [(0.1, ":"), (0.5, "-"), (0.9, "-.")]:
            hq = np.quantile(h, q)
            ax.plot(days, 2.0 ** (-days / hq), color=col, ls=ls,
                    label=f"{name}: h at {int(q*100)}th pct = {hq:.0f} d")
    ax.set_xlabel("days since last meaningful practice (Δ)")
    ax.set_ylabel("retention R")
    ax.set_title("Fitted retention curves (training pairs)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figpath, dpi=150)
    plt.close(fig)


def train_all(cfg, variant="default", n_boot=None, graphs=("chain",), out_tag="", do_bootstrap=True,
              tune_windows=True):
    t0 = time.time()
    data = load_variant(cfg, variant)
    mdir = path(cfg, "models") / (variant if variant != "default" else "")
    mdir.mkdir(parents=True, exist_ok=True)
    tabs, figs = path(cfg, "tables"), path(cfg, "figures")
    tag = "" if variant == "default" else f"_{variant}"
    summary = {"variant": variant}
    val_rows = []
    fitted = {}
    tuning = []
    for graph in graphs:
        gsuf = "" if graph == "chain" else "_data"
        D = designs(data, graph)
        if graph == "chain":
            ident = identification_check(D["train"], figs, tabs, tag)
            summary["identification"] = ident
            write_json(ident, tabs / f"delta_identification{tag}.json")
            log.info("identification: %s", ident)
            # baselines
            maj = Majority().fit(D["train"])
            fitted["majority"] = maj
            for kind, name in [("irt", "topic_difficulty"), ("pfa", "pfa")]:
                ll, C, m = tune_sklearn(kind, D)
                fitted[name] = m
                tuning.append({"model": name, "C": C, "val_logloss": ll})
            # DAS3H windows: tune the window set on validation
            win_sets = {"7_30_inf": (7, 30, np.inf)}
            if tune_windows:
                win_sets["7_14_30_inf"] = (7, 14, 30, np.inf)
            best = None
            for wn, ws in win_sets.items():
                ll, C, m = tune_sklearn("das3h", D, ws)
                tuning.append({"model": f"das3h[{wn}]", "C": C, "val_logloss": ll})
                if best is None or ll < best[0]:
                    best = (ll, wn, m)
            fitted["das3h"] = best[2]
            summary["das3h_windows"] = best[1]
            # the default window set is also kept for the sensitivity table
            if tune_windows:
                fitted["das3h_7_30_inf"] = tune_sklearn("das3h", D, (7, 30, np.inf))[2]
                fitted["das3h_7_14_30_inf"] = tune_sklearn("das3h", D, (7, 14, 30, np.inf))[2]
        # forgetting variants
        base_none, tdf = tune_forgetting("none", D, data, cfg)
        tuning += tdf.to_dict("records")
        fitted["none" + gsuf] = base_none
        for name in ["fixed", "learned", "learnedI"]:
            m, tdf = tune_forgetting(name, D, data, cfg, init_from=base_none)
            tdf["graph"] = graph
            tuning += tdf.to_dict("records")
            fitted[name + gsuf] = m
        if graph == "chain":
            decay_figure(fitted, D["train"], figs / f"decay_curves{tag}.png")
        # validation metrics
        for name, m in fitted.items():
            if (graph == "chain" and name.endswith("_data")) or (graph == "data" and not name.endswith("_data")):
                continue
            p = m.predict_proba(D["val"])
            val_rows.append({"model": name, "graph": graph,
                             **all_metrics(D["val"].rows["y"], p, D["val"].rows["y_score"])})
        # bootstrap intervals
        if do_bootstrap and graph == "chain":
            nb = n_boot or cfg["model"]["bootstrap"]
            coef_rows = []
            topic_names = [f"{m}-{k}" for (m, k) in sorted(data["tindex"], key=data["tindex"].get)]
            for name in ["none", "fixed", "learned", "learnedI"]:
                tb = time.time()
                W = bootstrap(name, D["train"], fitted[name], nb, cfg["seed"])
                ct = fitted[name].coef_table(topic_names)
                ct["ci_lo"] = np.percentile(W, 2.5, axis=0)
                ct["ci_hi"] = np.percentile(W, 97.5, axis=0)
                ct["boot_sd"] = W.std(0)
                ct["excludes_0"] = (ct["ci_lo"] > 0) | (ct["ci_hi"] < 0)
                ct["model"] = name
                ct["n_boot"] = nb
                if name == "fixed":
                    ct["S"] = fitted[name].S
                coef_rows.append(ct)
                np.save(mdir / f"boot_{name}.npy", W)
                log.info("bootstrap %s: %d resamples in %.0fs", name, nb, time.time() - tb)
            coef = pd.concat(coef_rows, ignore_index=True)
            coef.to_csv(tabs / f"coefficients{tag}.csv", index=False)
    pd.DataFrame(tuning).to_csv(tabs / f"model_tuning_val{tag}.csv", index=False)
    pd.DataFrame(val_rows).to_csv(tabs / f"model_metrics_val{tag}.csv", index=False)
    joblib.dump(fitted, mdir / "fitted_models.joblib")
    summary["fixed_S"] = fitted["fixed"].S
    summary["l2"] = {n: fitted[n].l2 for n in FORGET_VARIANTS}
    summary["seconds"] = time.time() - t0
    write_json(summary, mdir / "train_summary.json")
    return fitted, summary
