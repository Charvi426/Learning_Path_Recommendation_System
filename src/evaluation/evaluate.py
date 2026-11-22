"""Stage-5 evaluation on the test split (METHODOLOGY section 7). Uses the frozen config/final_config.yaml."""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..features.build import wname  # noqa: E402
from ..models.data import build_design, training_targets  # noqa: E402
from ..recommend.engine import RecData, evaluate_paths, make_paths  # noqa: E402
from ..recommend.explain import explain_item  # noqa: E402
from ..recommend.path import labels  # noqa: E402
from ..recommend.scoring import TERMS  # noqa: E402
from .model_metrics import all_metrics, calibration_table, ece  # noqa: E402
from .stats import bootstrap_mean_ci, fast_student_bootstrap_indices, wilcoxon_holm  # noqa: E402

MODEL_LABELS = {
    "majority": "Majority", "topic_difficulty": "Topic difficulty (IRT-like)", "pfa": "PFA",
    "none": "Forgetting-aware PFA: no forgetting", "fixed": "Forgetting-aware PFA: fixed S",
    "learned": "Forgetting-aware PFA: learned half-life",
    "learnedI": "Forgetting-aware PFA: learned half-life + interest (full 5.1)",
    "das3h": "DAS3H windows (benchmark)", "das3h_7_30_inf": "DAS3H windows {7,30,inf}",
    "das3h_7_14_30_inf": "DAS3H windows {7,14,30,inf}",
}


# --------------------------------------------------------------------------------------------- 7.1
def model_metrics_test(fitted, D_test, n_boot, seed, figpath=None, order=None):
    y = D_test.rows["y"].to_numpy(float)
    ys = D_test.rows["y_score"].to_numpy(float)
    sid = D_test.rows["sid"].to_numpy()
    Wb = fast_student_bootstrap_indices(sid, n_boot, seed)
    rows, calib, preds = [], [], {}
    for name in order or fitted:
        if name not in fitted:
            continue
        p = fitted[name].predict_proba(D_test)
        preds[name] = p
        m = all_metrics(y, p, ys)
        m["ECE"] = ece(y, p)
        bs = {k: [] for k in ["AUC", "logloss", "Brier", "RMSE"]}
        for w in Wb:
            if name != "majority":
                bs["AUC"].append(roc_auc_score(y, p, sample_weight=w))
            ww = w / w.sum()
            pc = np.clip(p, 1e-12, 1 - 1e-12)
            bs["logloss"].append(-np.sum(ww * (y * np.log(pc) + (1 - y) * np.log(1 - pc))))
            bs["Brier"].append(np.sum(ww * (p - y) ** 2))
            bs["RMSE"].append(np.sqrt(np.sum(ww * (p - y) ** 2)))
        row = {"model": name, "label": MODEL_LABELS.get(name, name), "n_rows": len(y),
               "n_students": int(len(np.unique(sid)))}
        for k, v in m.items():
            row[k] = v
            if k in bs and len(bs[k]):
                row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(bs[k], 2.5), np.percentile(bs[k], 97.5)
        rows.append(row)
        c = calibration_table(y, p)
        c["model"] = name
        calib.append(c)
    df = pd.DataFrame(rows)
    cal = pd.concat(calib, ignore_index=True)
    if figpath is not None:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 1], [0, 1], color="grey", ls=":")
        for name in ["topic_difficulty", "pfa", "none", "fixed", "learned", "learnedI", "das3h"]:
            c = cal[cal["model"] == name]
            if len(c):
                ax.plot(c["mean_pred"], c["obs_rate"], marker="o", ms=3, label=MODEL_LABELS.get(name, name))
        ax.set_xlabel("mean predicted P(pass)")
        ax.set_ylabel("observed pass rate")
        ax.set_title("Calibration on the test split (10 equal-width bins)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(figpath, dpi=150)
        plt.close(fig)
    return df, cal, preds


def paired_model_differences(preds: dict, D_test, pairs, n_boot, seed):
    """Paired student-bootstrap CIs for AUC and log-loss differences between models (same resamples)."""
    y = D_test.rows["y"].to_numpy(float)
    Wb = fast_student_bootstrap_indices(D_test.rows["sid"].to_numpy(), n_boot, seed + 1)

    def ll(p, w):
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.sum(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))) / w.sum()

    rows = []
    for a, b in pairs:
        pa, pb = preds[a], preds[b]
        da = [roc_auc_score(y, pa, sample_weight=w) - roc_auc_score(y, pb, sample_weight=w) for w in Wb]
        dl = [ll(pa, w) - ll(pb, w) for w in Wb]
        one = np.ones_like(y)
        rows.append({"model_x": a, "model_y": b,
                     "AUC_diff": roc_auc_score(y, pa) - roc_auc_score(y, pb),
                     "AUC_diff_lo": np.percentile(da, 2.5), "AUC_diff_hi": np.percentile(da, 97.5),
                     "logloss_diff": ll(pa, one) - ll(pb, one),
                     "logloss_diff_lo": np.percentile(dl, 2.5), "logloss_diff_hi": np.percentile(dl, 97.5),
                     "n_boot": n_boot})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- 7.2
def config_paths(rd: RecData, final: dict, cid: str, seed: int, rcfg: dict):
    t = final["ablations"][cid]
    conf = t["config"]
    if conf["kind"] != "score":
        P, _ = make_paths(rd, conf, None, None, L=rcfg["K"], seed=seed)
        return P[0], None, conf, None
    W = np.array([[t["weights"][k] for k in TERMS]])
    P, aux = make_paths(rd, conf, W, t["thresholds"], L=rcfg["K"], max_rev=rcfg["max_revision"],
                        horizon=rcfg["horizon"], return_terms=True)
    return P[0], aux, conf, W[0]


def ranking_table(rd: RecData, final: dict, rcfg: dict, n_boot: int, seed: int, configs=None):
    rows, per_student, paths = [], {}, {}
    for cid in configs or final["ablations"]:
        P, aux, conf, W = config_paths(rd, final, cid, seed, rcfg)
        paths[cid] = (P, aux, conf, W)
        (_, ps), = evaluate_paths(rd, P[None], rels=("A", "B", "AB"))
        per_student[cid] = ps
        for metric, sm in ps.items():
            est, lo, hi = bootstrap_mean_ci(sm, n_boot, seed)
            base, rel = metric.split("[")
            rows.append({"config": cid, "label": conf["label"], "relevance": rel.rstrip("]"), "metric": base,
                         "value": est, "ci_lo": lo, "ci_hi": hi, "n_students": int(np.isfinite(sm).sum())})
    return pd.DataFrame(rows), per_student, paths


# --------------------------------------------------------------------------------------------- 7.3
def simulated_gain(rd: RecData, paths: dict, simulator, state_test, queries, graphs, tindex, med_clicks: dict,
                   n_boot: int, seed: int):
    """Predict P(pass k_now) with the DAS3H window model before/after following each path."""
    qmeta = queries.set_index("qid").loc[rd.qid]
    targets = pd.DataFrame({"qid": rd.qid, "sid": rd.sid, "k": rd.k_now, "t": rd.t, "code_module": rd.module})
    D = build_design(targets, state_test, graphs["chain"], tindex, windows=simulator.windows)
    p0 = simulator.predict_proba(D)
    rows = []
    per_student = {}
    med = np.array([med_clicks[m] for m in rd.module])
    pr = D.pairs.copy()
    t_row = D.rows["t"].to_numpy()[pr["row"].to_numpy()]
    for cid, (P, aux, conf, W) in paths.items():
        is_rev = aux["rev"] if aux is not None else rd.assessed
        R = D.rows.copy()
        inpath = np.zeros((rd.N, rd.T), bool)
        for pos in range(P.shape[1]):
            ok = P[:, pos] >= 0
            inpath[np.flatnonzero(ok), P[ok, pos]] = True
        rev_items = inpath & is_rev
        learn_k = inpath[np.arange(rd.N), rd.k_now - 1] & ~rd.assessed[np.arange(rd.N), rd.k_now - 1]
        # Learn/Practice of the target topic: add the median per-topic clicks to every window
        for w in simulator.windows:
            c = f"c_{wname(w)}"
            R[c] = np.where(learn_k, np.log1p(np.expm1(R[c]) + med), R[c])
        # Revise of a prerequisite: its outcome counts as recent (inside every window)
        revised_pair = rev_items[pr["row"].to_numpy(), pr["j"].to_numpy() - 1] & pr["assessed"].to_numpy()
        yj = pr["y_j"].to_numpy()
        for w in simulator.windows:
            already = (t_row - pr["t_submit_j"].to_numpy()) <= w
            add = revised_pair & ~already
            R[f"succ_{wname(w)}"] += np.bincount(pr["row"], weights=(add & (yj == 1)).astype(float), minlength=rd.N)
            R[f"fail_{wname(w)}"] += np.bincount(pr["row"], weights=(add & (yj == 0)).astype(float), minlength=rd.N)
        D2 = type(D)(rows=R, pairs=D.pairs, topic_index=D.topic_index)
        p1 = simulator.predict_proba(D2)
        gain = p1 - p0
        from .ranking_metrics import student_mean
        sm = student_mean(gain, rd.sidx, rd.n_students)
        per_student[cid] = sm
        est, lo, hi = bootstrap_mean_ci(sm, n_boot, seed)
        rows.append({"config": cid, "label": conf["label"], "mean_gain": est, "ci_lo": lo, "ci_hi": hi,
                     "share_paths_with_target": float(learn_k.mean()),
                     "share_paths_revising_prereq": float((np.bincount(pr["row"], weights=revised_pair.astype(float),
                                                                       minlength=rd.N) > 0).mean()),
                     "mean_p_nothing": float(p0.mean()), "mean_p_path": float(p1.mean()),
                     "simulator": "DAS3H windows " + "/".join(wname(w) for w in simulator.windows)})
    return pd.DataFrame(rows), per_student


# --------------------------------------------------------------------------------------------- 7.4
def exam_check(ks_exam: pd.DataFrame, students: pd.DataFrame, variants, n_boot, seed, pass_mark=40):
    g = ks_exam.groupby("sid")
    agg = pd.DataFrame({"sid": list(g.groups)})
    for v in variants:
        agg[f"meanM_{v}"] = g[f"M_{v}"].mean().to_numpy()
        agg[f"meanK_{v}"] = g[f"K_{v}"].mean().to_numpy()
    agg = agg.merge(students[["sid", "exam_score", "code_module"]], on="sid")
    agg = agg[agg["exam_score"].notna()]
    ex = agg["exam_score"].to_numpy(float)
    passed = (ex >= pass_mark).astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(agg), size=(n_boot, len(agg)))
    rows = []
    for v in variants:
        M, K = agg[f"meanM_{v}"].to_numpy(), agg[f"meanK_{v}"].to_numpy()
        rM, rK = sps.spearmanr(M, ex)[0], sps.spearmanr(K, ex)[0]
        aM, aK = roc_auc_score(passed, M), roc_auc_score(passed, K)
        d_r, d_a = [], []
        for ii in idx:
            d_r.append(sps.spearmanr(K[ii], ex[ii])[0] - sps.spearmanr(M[ii], ex[ii])[0])
            d_a.append(roc_auc_score(passed[ii], K[ii]) - roc_auc_score(passed[ii], M[ii]))
        rows.append({"variant": v, "n_students": len(agg), "spearman_M": rM, "spearman_K": rK,
                     "diff_spearman_K_minus_M": rK - rM, "diff_lo": np.percentile(d_r, 2.5),
                     "diff_hi": np.percentile(d_r, 97.5), "AUC_M": aM, "AUC_K": aK, "diff_AUC_K_minus_M": aK - aM,
                     "diffAUC_lo": np.percentile(d_a, 2.5), "diffAUC_hi": np.percentile(d_a, 97.5),
                     "exam_pass_rate": float(passed.mean())})
    return pd.DataFrame(rows), agg


# --------------------------------------------------------------------------------------------- 7.6
def fairness(rd: RecData, per_student: dict, students: pd.DataFrame, preds: dict, D_test, configs=("A3", "A6"),
             models=("none", "learnedI")):
    st = students.set_index("sid")
    sids = np.unique(rd.sid)
    rows = []
    attrs = {"gender": "gender", "imd_band": "imd_band", "disability": "disability"}
    for attr, col in attrs.items():
        grp = st.loc[sids, col].fillna("missing").astype(str).to_numpy()
        for cid in configs:
            v = per_student[cid]["NDCG@5[AB]"]
            for gval in sorted(set(grp)):
                m = (grp == gval) & np.isfinite(v)
                rows.append({"attribute": attr, "group": gval, "what": f"NDCG@5 {cid}", "value": float(np.mean(v[m]))
                             if m.any() else np.nan, "n": int(m.sum())})
        gr_rows = st.loc[D_test.rows["sid"], col].fillna("missing").astype(str).to_numpy()
        y = D_test.rows["y"].to_numpy()
        for mname in models:
            for gval in sorted(set(gr_rows)):
                m = gr_rows == gval
                a = roc_auc_score(y[m], preds[mname][m]) if len(np.unique(y[m])) == 2 else np.nan
                rows.append({"attribute": attr, "group": gval, "what": f"AUC {mname}", "value": a, "n": int(m.sum())})
    df = pd.DataFrame(rows)
    gaps = (df[df["n"] >= 30].groupby(["attribute", "what"])["value"]
            .agg(lambda s: s.max() - s.min()).rename("largest_gap").reset_index())
    return df, gaps


# --------------------------------------------------------------------------------------------- 6.5
def sample_paths(rd: RecData, paths: dict, cid: str, ks_test: pd.DataFrame, n=10, seed=0):
    P, aux, conf, W = paths[cid]
    model = conf["model"]
    m = rd.models[model]
    rng = np.random.default_rng(seed)
    ab = rd.ability
    q1, q2 = np.quantile(ab, [0.2, 0.8])
    has = (P >= 0).sum(1) >= 3
    picks = []
    for lab, mask, cnt in [("strong", ab >= q2, 3), ("average", (ab > q1) & (ab < q2), 4), ("struggling", ab <= q1, 3)]:
        cand = np.flatnonzero(mask & has)
        cand = cand[rng.permutation(len(cand))]
        chosen, seen = [], set()
        for c in cand:
            if rd.sid[c] in seen:
                continue
            chosen.append(c)
            seen.add(rd.sid[c])
            if len(chosen) == cnt:
                break
        picks += [(lab, c) for c in chosen]
    terms = aux["terms"]
    out = []
    graph = conf.get("graph")
    for lab, i in picks:
        name = lambda j: f"{rd.module[i]} topic {j + 1}"  # noqa: E731
        labs = labels(P[i], aux["rev"][i], rd.opened[i])
        items = []
        for pos, j in enumerate([x for x in P[i] if x >= 0]):
            pre = [pp for pp in range(rd.T) if rd.prereq[graph][i, j, pp]] if graph else []
            pk = None
            if pre:
                pk = min(pre, key=lambda pp: m["K"][i, pp])
            info = {"M": m["M"][i, j], "R": m["R"][i, j] if np.isfinite(m["R"][i, j]) else 1.0,
                    "K": m["K"][i, j], "delta": rd.delta[i, j] if np.isfinite(rd.delta[i, j]) else 0,
                    "p_hat": m["K"][i, j], "prereq_name": name(pk) if pk is not None else None,
                    "prereq_K": m["K"][i, pk] if pk is not None else None, "is_rev": bool(aux["rev"][i, j])}
            contrib = (W * terms[i, j]).round(3)
            items.append({"position": pos + 1, "topic": name(j), "action": labs[pos],
                          "score": float((W * terms[i, j]).sum()),
                          "terms": dict(zip(TERMS, terms[i, j].round(3).tolist())),
                          "weighted": dict(zip(TERMS, contrib.tolist())),
                          "explanation": explain_item(name(j), terms[i, j], W, info)})
        out.append({"student_group": lab, "student_sid": int(rd.sid[i]), "module": rd.module[i],
                    "presentation": rd.pres[i], "decision_day": float(rd.t[i]), "next_assessment_topic": int(rd.k_now[i]),
                    "ability": float(rd.ability[i]), "config": cid, "path": items,
                    "observed_outcome_group": {"A": "passed next assessment", "B": "failed next assessment",
                                               "none": "did not submit / no outcome"}[rd.group[i]]})
    return out


def sample_paths_markdown(sp: list, weights: np.ndarray, cid: str) -> str:
    lines = [f"# Sample learning paths ({cid}, test split 2014J)", "",
             "Frozen configuration from `config/final_config.yaml`. Weights (G, Q, F, D, I) = "
             + ", ".join(f"{t}={w:.1f}" for t, w in zip(TERMS, weights)) + ".",
             "Students are pseudonymous internal ids (`sid`), grouped by ability at the decision point "
             "(top 20% = strong, bottom 20% = struggling).", ""]
    for n, s in enumerate(sp, 1):
        lines.append(f"## {n}. {s['student_group'].capitalize()} student (sid {s['student_sid']}), "
                     f"{s['module']} {s['presentation']}, day {s['decision_day']:.0f}, next assessment = topic "
                     f"{s['next_assessment_topic']}")
        lines.append(f"ability = {s['ability']:.2f}; observed afterwards: {s['observed_outcome_group']}")
        lines.append("")
        lines.append("| # | Action | Topic | Score | Explanation |")
        lines.append("| --- | --- | --- | --- | --- |")
        for it in s["path"]:
            lines.append(f"| {it['position']} | {it['action']} | {it['topic']} | {it['score']:.3f} | {it['explanation']} |")
        lines.append("")
    return "\n".join(lines)
