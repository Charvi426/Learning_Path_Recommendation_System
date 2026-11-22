"""Stage-2 orchestration: factors for every query, written per variant.

variant = 'default'      pass mark 40, non-submissions excluded from targets
          'pass70'       pass mark 70 (sensitivity)
          'nonsub_fail'  a non-submission counts as a fail with score 0 at the due date (sensitivity)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..utils.io import ROOT, get_logger, path, write_json, write_parquet
from . import difficulty as DF
from . import prerequisites as PR
from .build import TopicArrays, compute_state, make_queries, total_clicks_before, wname
from .engagement import ClickCube
from .interest import add_interest, add_share, fit_zstats, raw_signal

log = get_logger("features")


def variant_dir(cfg: dict, variant: str) -> Path:
    p = path(cfg, "processed")
    if variant != "default":
        p = p / f"sens_{variant}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def transform_interactions(inter: pd.DataFrame, variant: str, pass_mark: float) -> pd.DataFrame:
    inter = inter.copy()
    if variant == "pass70":
        inter["y"] = np.where(inter["is_submitted"] == 1, (inter["score"] >= 70).astype(float), np.nan)
    elif variant == "nonsub_fail":
        ns = inter["is_submitted"] == 0
        inter.loc[ns, "score"] = 0.0
        inter.loc[ns, "y_score"] = 0.0
        inter.loc[ns, "t_submit"] = inter.loc[ns, "t_due"]
        inter.loc[ns, "is_submitted"] = 1
        inter["y"] = (inter["score"] >= pass_mark).astype(float)
    return inter


def build_state_tables(cfg: dict, inter: pd.DataFrame, students: pd.DataFrame, topics: pd.DataFrame,
                       vd: pd.DataFrame, cmy: dict, day_mask: str | None = None):
    n_topics = int(topics["topic"].max())
    ta = TopicArrays(inter, topics, students, n_topics)
    cube = ClickCube(vd, ta.n_sid, n_topics, cfg["features"]["revision_min_clicks"], day_mask=day_mask)
    queries = make_queries(inter, students, ta)
    state = compute_state(cube, ta, queries, cfg["features"]["windows_extended"], cmy,
                          cfg["features"]["ability_shrink"])
    del cube
    return ta, queries, state


def interest_validity(cfg, inter, students, topics, vd, cmy, queries, state, zstats, total_daily, use_vol):
    """Methodology 4.4 checks on training 'due' queries; rows = opened (student, query, topic)."""
    rows = []
    trq = queries[(queries["split"] == "train") & (queries["kind"] == "due")]["qid"]
    st = state[state["qid"].isin(trq) & state["opened"]]
    ass = st[st["assessed"]]

    def corr(a, b, name, n_target=None, target=None, passed=None):
        m = np.isfinite(a) & np.isfinite(b)
        r = sps.pearsonr(a[m], b[m])[0]
        rho = sps.spearmanr(a[m], b[m])[0]
        rows.append({"check": name, "pearson_r": r, "spearman_rho": rho, "n": int(m.sum()),
                     "target": target or "", "passes": passed(r) if passed else ""})

    pos = lambda r: bool(r > 0)  # noqa: E731
    corr(st["vol"].to_numpy(), st["early"].to_numpy(), "signals: vol vs early (opened topics)", target="> 0", passed=pos)
    corr(ass["vol"].to_numpy(), ass["post"].to_numpy(), "signals: vol vs post (assessed topics)", target="> 0", passed=pos)
    corr(ass["early"].to_numpy(), ass["post"].to_numpy(), "signals: early vs post (assessed topics)", target="> 0", passed=pos)

    # same correlations within (module, topic, stage): removes the mechanical trade-off that early topics
    # have long post periods and late topics long early periods
    def dm(d, c):
        return (d[c] - d.groupby(["code_module", "j", "stage"])[c].transform("mean")).to_numpy()
    corr(dm(st, "vol"), dm(st, "early"), "signals within (module,topic,stage): vol vs early", target="> 0", passed=pos)
    corr(dm(ass, "vol"), dm(ass, "post"), "signals within (module,topic,stage): vol vs post", target="> 0", passed=pos)
    corr(dm(ass, "early"), dm(ass, "post"), "signals within (module,topic,stage): early vs post", target="> 0", passed=pos)
    weak = lambda r: bool(abs(r) < 0.3)  # noqa: E731
    tot = pd.Series(total_clicks_before(total_daily, queries), index=queries["qid"])
    corr(st["I"].to_numpy(), tot.reindex(st["qid"]).to_numpy(), "I vs student total clicks (all resources, log1p)",
         target="|r| < 0.3", passed=weak)
    corr(st["I"].to_numpy(), st["c_inf"].to_numpy(), "I vs clicks on the same topic (c_inf)",
         target="|r| < 0.3", passed=weak)
    # split-half: I from odd vs even days
    halves = {}
    for mask in ["odd", "even"]:
        _, q2, s2 = build_state_tables(cfg, inter, students, topics, vd, cmy, day_mask=mask)
        s2 = add_share(s2, raw_signal(s2, cfg["features"]["interest_lambda_early"],
                                      cfg["features"]["interest_lambda_post"], use_vol))
        z2 = fit_zstats(s2[s2["qid"].isin(trq)])
        s2 = add_interest(s2, z2)
        halves[mask] = s2[["qid", "j", "I", "share"]]
    h = halves["odd"].merge(halves["even"], on=["qid", "j"], suffixes=("_odd", "_even"))
    h = h[h["qid"].isin(trq)]
    both = h["share_odd"].notna() & h["share_even"].notna()
    a, b = h.loc[both, "I_odd"].to_numpy(), h.loc[both, "I_even"].to_numpy()
    r = sps.pearsonr(a, b)[0]
    rows.append({"check": "split-half: I(odd days) vs I(even days), both halves active",
                 "pearson_r": r, "spearman_rho": sps.spearmanr(a, b)[0], "n": int(both.sum()),
                 "target": "> 0", "passes": bool(r > 0)})
    rows.append({"check": "split-half reliability, Spearman-Brown corrected", "pearson_r": 2 * r / (1 + r),
                 "spearman_rho": np.nan, "n": int(both.sum()), "target": "> 0", "passes": bool(r > 0)})
    return pd.DataFrame(rows)


def build_features(cfg: dict, variant: str = "default", run_validity: bool = True) -> dict:
    proc = path(cfg, "processed")
    out = variant_dir(cfg, variant)
    tabs = path(cfg, "tables")
    topics = pd.read_parquet(proc / "topics.parquet")
    students = pd.read_parquet(proc / "students.parquet")
    inter = pd.read_parquet(proc / "interactions.parquet")
    vd = pd.read_parquet(proc / "vle_daily.parquet", columns=["sid", "topic", "date", "clicks", "vol_clicks"])
    inter = transform_interactions(inter, variant, cfg["pass_mark"])
    train_inter = inter[inter["split"] == "train"]

    cmy = DF.cohort_mean_y(train_inter)
    diff = DF.topic_difficulty(train_inter)
    ch = PR.chain_edges(topics)
    de, de_stats = PR.data_edges(train_inter, topics, cfg["prerequisites"]["data_edge_min_diff"],
                                 cfg["prerequisites"]["data_edge_min_group"])
    edges = PR.combine(ch, de)
    edges.to_csv(out / "prereq_edges.csv", index=False)
    de_stats.to_csv(out / "prereq_data_edge_stats.csv", index=False)
    diff.to_csv(out / "topic_difficulty.csv", index=False)
    write_json({"cohort_mean_y": cmy}, out / "cohort_stats.json")
    if variant == "default":
        diff.to_csv(tabs / "topic_difficulty.csv", index=False)
        de_stats.to_csv(tabs / "prereq_data_edge_stats.csv", index=False)

    log.info("[%s] building click cube and state ...", variant)
    ta, queries, state = build_state_tables(cfg, inter, students, topics, vd, cmy)
    state["code_module"] = state["code_module"].astype(str)

    # interest proxy
    lam1, lam2 = cfg["features"]["interest_lambda_early"], cfg["features"]["interest_lambda_post"]
    use_vol = True
    trq = queries[(queries["split"] == "train") & (queries["kind"] == "due")]["qid"]
    state = add_share(state, raw_signal(state, lam1, lam2, use_vol))
    zstats = fit_zstats(state[state["qid"].isin(trq)])
    state = add_interest(state, zstats)
    info = {"variant": variant, "interest_uses_vol": True}
    if variant == "default" and run_validity:
        total_daily = pd.read_parquet(path(cfg, "interim") / "total_daily_clicks.parquet")
        val = interest_validity(cfg, inter, students, topics, vd, cmy, queries, state, zstats, total_daily, True)
        r_tot = val.loc[val["check"].str.startswith("I vs student total"), "pearson_r"].iloc[0]
        info["r_I_total_clicks"] = float(r_tot)
        if abs(r_tot) >= 0.3:
            log.warning("interest proxy |r| with total clicks = %.3f >= 0.3: fallback, dropping vol", r_tot)
            use_vol = False
            state = add_share(state, raw_signal(state, lam1, lam2, use_vol))
            zstats = fit_zstats(state[state["qid"].isin(trq)])
            state = add_interest(state, zstats)
            val2 = interest_validity(cfg, inter, students, topics, vd, cmy, queries, state, zstats, total_daily, False)
            val["proxy_version"] = "vol+early+post (rejected)"
            val2["proxy_version"] = "early+post (fallback, used)"
            val = pd.concat([val, val2], ignore_index=True)
        else:
            val["proxy_version"] = "vol+early+post (used)"
        val.to_csv(tabs / "interest_validity.csv", index=False)
        info["interest_uses_vol"] = use_vol
        write_json(info, out / "interest_info.json")
    elif variant != "default":
        # sensitivity variants reuse the default decision about the vol signal
        import json
        dinfo = json.load(open(variant_dir(cfg, "default") / "interest_info.json"))
        if not dinfo["interest_uses_vol"]:
            use_vol = False
            state = add_share(state, raw_signal(state, lam1, lam2, use_vol))
            zstats = fit_zstats(state[state["qid"].isin(trq)])
            state = add_interest(state, zstats)
        info["interest_uses_vol"] = use_vol
        write_json(info, out / "interest_info.json")

    zstats.to_csv(out / "interest_zstats.csv", index=False)
    write_parquet(queries, out / "queries.parquet")
    write_parquet(inter, out / "interactions_variant.parquet")
    for sp in ["train", "val", "test"]:
        qs = queries[queries["split"] == sp]["qid"]
        write_parquet(state[state["qid"].isin(qs)], out / f"state_{sp}.parquet")
        # wide training-row table (target topic + chain prerequisite) for inspection / reporting
        q = queries[(queries["split"] == sp) & (queries["kind"] == "due")]
        own = state[state["qid"].isin(q["qid"])]
        w = q.merge(own, left_on=["qid", "k"], right_on=["qid", "j"], how="left", suffixes=("", "_own"))
        pr = own[["qid", "j", "M", "delta", "assessed", "y_j", "I"]].copy()
        pr["k"] = pr["j"] + 1
        w = w.merge(pr.drop(columns="j").add_prefix("prev_").rename(columns={"prev_qid": "qid", "prev_k": "k"}),
                    on=["qid", "k"], how="left")
        w = w.merge(inter[["sid", "topic", "y", "y_score", "is_submitted"]].rename(columns={"topic": "k"}),
                    on=["sid", "k"], how="left")
        write_parquet(w, out / f"features_{sp}.parquet")
    log.info("[%s] queries %d, state rows %d", variant, len(queries), len(state))
    return info
