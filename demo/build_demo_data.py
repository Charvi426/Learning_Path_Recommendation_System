"""Precompute everything the demo reads (demo/data/*.parquet). The app itself never fits or scores anything.

  python demo/build_demo_data.py

Nothing is tuned and nothing under results/ or config/ is written. Coefficients are reproduced with the frozen
hyper-parameters (the fitted objects are not in git) and checked against the recorded ones. The build stops if they
differ, or if the recorded A5 ranking numbers cannot be reproduced on the scale they were scored on (v2: M*R,
v3: raw K, see DECISIONS.md DR-7).

The paths and plans written for display follow config/final_config_v3.yaml (K = M*R). They are never compared with
outcomes here.
"""
from __future__ import annotations

import copy
import json
import runpy
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import results_catalog as RC  # noqa: E402
from src.evaluation.relevance import relevance_table  # noqa: E402
from src.features.build import TopicArrays, compute_state  # noqa: E402
from src.features.engagement import ClickCube  # noqa: E402
from src.features.pipeline import build_features  # noqa: E402
from src.models.das3h import ForgettingPFA  # noqa: E402
from src.models.knowledge_state import knowledge_state  # noqa: E402
from src.models.train import fit_forgetting  # noqa: E402
from src.recommend import v2 as V2  # noqa: E402
from src.recommend.explain import explain_item  # noqa: E402
from src.recommend.path import labels  # noqa: E402
from src.utils.io import load_config, seed_everything  # noqa: E402
from src.v3 import common as C  # noqa: E402
from src.v3 import pipeline as PL  # noqa: E402
from src.v3 import planner as PN  # noqa: E402
from src.v3.data import HOLDOUT, load, practice  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "demo" / "data"
WORK = ROOT / "demo" / ".build"          # side outputs of the preprocessing stages, so results/ is never written
SETS = {"DDDFFF": ["DDD", "FFF"], **{m: [m] for m in HOLDOUT}}
N_PER_MODULE, MIN_EDGE_CASES = 300, 8
REVISION_DOSE = float(np.log1p(3))


def log(msg):
    print(f"[demo build] {msg}", flush=True)


# ------------------------------------------------------------------ intermediates (clean clone: raw CSVs only)
def ensure_intermediates(cfg):
    need = ["state_train", "state_test", "queries", "topic_map", "vle_daily", "students"]

    def ready(proc: Path, interim: Path):
        return all((proc / f"{n}.parquet").exists() for n in need) and (interim / "student_vle_capped.parquet").exists()

    missing_raw = not (ROOT / "data/raw/studentVle.csv").exists()
    todo = []
    for cs in SETS:
        sub = "" if cs == "DDDFFF" else f"/v2_holdout/{cs}"
        if not ready(ROOT / f"data/processed{sub}", ROOT / f"data/interim{sub}"):
            todo.append(cs)
    if not todo:
        return
    if missing_raw:
        raise SystemExit("Put the seven OULAD CSVs in data/raw/ first (see demo/README.md).")
    s1 = runpy.run_path(str(ROOT / "scripts/01_preprocess.py"), run_name="s1")["main"]
    module_cfg = runpy.run_path(str(ROOT / "scripts/v2/fix4_holdout.py"), run_name="f4")["module_cfg"]
    for cs in todo:
        c = copy.deepcopy(cfg) if cs == "DDDFFF" else module_cfg(cfg, cs)
        c["paths"] = {**c["paths"], **{k: f"demo/.build/{cs}/{k}" for k in ("tables", "figures", "models", "sample_paths")}}
        log(f"preprocessing {cs}")
        s1(c)
        build_features(c, "default", run_validity=False)


# ------------------------------------------------------------------ frozen models
def load_frozen():
    read = lambda p: yaml.safe_load((ROOT / p).read_text(encoding="utf-8"))  # noqa: E731
    return {"v1": read("config/final_config.yaml"), "v2": read("config/final_config_v2.yaml"),
            "v3": read("config/final_config_v3.yaml"), "sel": C.selected()}


def fit_course(cs, frozen):
    """Reproduce the frozen v2 (learned half-life) and v3 (ACT-R) fits for one course set."""
    data = load(cs)
    D = PL.due_design(data, "train")
    km = frozen["v1"]["knowledge_models"]
    none = ForgettingPFA("none", False, l2=float(km["l2"]["none"]), n_topics=len(data["tindex"])).fit(D)
    learned = fit_forgetting("learned", {"train": D}, data, float(km["l2"]["learned"]), float(km["fixed_S"]), init_from=none)
    var, _ = C.build_variants(frozen["sel"], cs, data, ["train", "test"])
    var.fit(data, D)
    return {"cs": cs, "data": data, "learned": learned, "var": var}


def check_coefficients(b):
    def close(a, ref, what, tol):
        if abs(a - ref) > tol:
            raise SystemExit(f"{b['cs']}: {what} = {a} differs from the recorded {ref}; refusing to build")

    if b["cs"] == "DDDFFF":
        rec = pd.read_csv(ROOT / "results/tables/coefficients.csv")
        for i, ref in enumerate(rec[rec["model"] == "learned"]["estimate"]):
            close(b["learned"].w_[i], ref, f"v2 coefficient {i}", 1e-6)
        want = json.loads((ROOT / "results/v3/coefficients_selected_variant_train.json").read_text())["v3"]
    else:
        want = json.loads((ROOT / "results/v3/holdout/coefficients_by_module.json").read_text())[b["cs"]]
    got = b["var"].m.coef()
    for k, ref in want.items():
        close(got[k], ref, f"v3 {k}", 2e-4)          # recorded with 4 decimals


def recommender_data(b, cfg):
    data, T = b["data"], b["data"]["T"]
    ks = knowledge_state(data["queries"], data["state"]["test"], {"learned": b["learned"]}, data["graphs"],
                         data["tindex"], data["diff"])
    ks["code_module"] = ks["code_module"].astype(str)
    q = data["queries"]
    dp = q[(q["kind"] == "decision") & (q["split"] == "test")][["qid", "sid", "k", "t", "split", "code_module"]]
    vd = pd.read_parquet(data["dirs"]["proc"] / "vle_daily.parquet", columns=["sid", "topic", "date", "clicks", "vol_clicks"])
    rel = relevance_table(dp, data["inter"], vd, int(data["students"]["sid"].max()) + 1, T,
                          cfg["recommender"]["rel_a_min_clicks"])
    rd = V2.RecDataV2(ks[ks["kind"] == "decision"], rel, data["graphs"], ["learned"], None, data["topics"], T)
    gone = ~rd.exists                                   # same masking as scripts/v2/fix4_holdout.py
    rd.assessed &= rd.exists
    for k in ("M", "K", "p"):
        rd.models["learned"][k] = np.where(gone, 1.0, rd.models["learned"][k])
    rd.I = np.where(gone, 0.5, rd.I)
    for a in ("relA", "relB", "relAB"):
        setattr(rd, a, np.where(gone, 0.0, getattr(rd, a)))
    b.update(rd=rd, vd=vd, kv=C.knowledge_for_decisions(b["var"], data, "test"))
    return b


def check_recorded_scores(bundles, frozen, cfg):
    """Reproduces numbers that already exist; it does not evaluate the configuration that the demo displays."""
    rc = cfg["recommender"]
    per = {}
    for cs, b in bundles.items():
        rd = C.inject(b["rd"], b["kv"], "v3", "K")       # raw K: the scale the recorded numbers were scored on
        paths = {**C.v2_paths(rd, frozen["v2"], ["A5"], rc), **C.v3_paths(rd, frozen["v3"], ["A5"], rc)}
        per[cs] = {k: v["NDCG@5[AB]"] for k, v in C.per_student(rd, paths).items()}
    rows = []
    for name, sets, f in (("holdout", HOLDOUT, "results/v3/holdout/recommender_holdout.csv"),
                          ("DDD/FFF 2014J", ["DDDFFF"], "results/v3/posthoc_2014J/recommender_2014J.csv")):
        for cfg_id in ("v2:A5", "v3:A5"):
            got = float(np.nanmean(np.concatenate([per[s][cfg_id] for s in sets])))
            ref = float(RC.read(RC.cell(f, "value", config=cfg_id, relevance="AB", metric="NDCG@5")))
            if abs(got - ref) > 1e-9:
                raise SystemExit(f"cannot reproduce recorded {name} {cfg_id}: {got} vs {ref}; refusing to build")
            rows.append({"set": name, "config": cfg_id, "recorded": ref,
                         "reproduced_on_scale": "raw K" if cfg_id == "v3:A5" else "M*R", "difference": got - ref})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ students
def choose_students(b, module, seed):
    d = b["data"]
    s = d["students"]
    s = s[(s["split"] == "test") & (s["code_module"] == module)].copy()
    clicks = b["vd"].groupby("sid")["clicks"].sum()
    s["n_clicks"] = s["sid"].map(clicks).fillna(0.0)
    s["withdrawn"] = s["date_unregistration"].notna()
    rng = np.random.default_rng(seed)
    order = s.iloc[rng.permutation(len(s))]
    forced = pd.concat([order[order["n_clicks"] == 0].head(MIN_EDGE_CASES), order[order["withdrawn"]].head(MIN_EDGE_CASES),
                        order[~order["withdrawn"]].head(MIN_EDGE_CASES)])
    s = pd.concat([forced, order]).drop_duplicates("sid").head(N_PER_MODULE)
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)           # the id must not reveal why it was chosen
    s["student"] = [f"{module}-{i + 1:03d}" for i in range(len(s))]
    return s


def exam_day_sat():
    raw = ROOT / "data/raw"
    a = pd.read_csv(raw / "assessments.csv")
    x = a[a["assessment_type"] == "Exam"].merge(pd.read_csv(raw / "studentAssessment.csv"), on="id_assessment")
    x = x[(x["is_banked"] == 0) & x["score"].notna()]
    return x[["code_module", "code_presentation", "id_student", "date_submitted"]].rename(
        columns={"date_submitted": "day_sat"})


# ------------------------------------------------------------------ per-student frames
def topic_name(module, j):
    return f"{module} topic {j + 1}"


def decision_frames(b, sel, frozen, cfg):
    rd, rc = b["rd"], cfg["recommender"]
    C.inject(rd, b["kv"], "v3", frozen["v3"]["knowledge_scale"]["kcol"])      # frozen scale: M*R
    anon = dict(zip(sel["sid"], sel["student"]))
    keep = [i for i, s in enumerate(rd.sid) if s in anon]
    c = rd.k_now[keep].astype(int)
    dec = pd.DataFrame({"student": [anon[rd.sid[i]] for i in keep], "decision": [int(rd.qid[i]) for i in keep],
                        "day": rd.t[keep].astype(int), "next_topic": c})
    dec["week"] = np.maximum(dec["day"] // 7 + 1, 0)

    clicks = b["data"]["state"]["test"]
    clicks = clicks[clicks["qid"].isin(rd.qid)].sort_values(["qid", "j"])["clicks_raw_inf"].to_numpy().reshape(rd.N, rd.T)
    states, paths = [], []
    for ver, model in (("v2", "learned"), ("v3", "v3")):
        t = frozen[ver]["ablations"]["A5"]
        W = np.array([[t["weights"][k] for k in V2.TERMS6]])
        P, aux = V2.paths_v2(rd, t["config"], W, t["filters"], rc["K"], rc["max_revision"], rc["horizon"])
        P, m, w = P[0], rd.models[model], W[0][:5]
        for i in keep:
            who, mod, did = anon[rd.sid[i]], rd.module[i], int(rd.qid[i])
            for j in range(rd.T):
                if not rd.exists[i, j]:
                    continue
                done = bool(rd.assessed[i, j])
                states.append((who, ver, did, j + 1, m["M"][i, j], not done, m["R"][i, j] if done else np.nan,
                               m["K"][i, j], rd.I[i, j],
                               ("Fading" if aux["rev"][i, j] else "Learned") if done else "Not started", clicks[i, j]))
            items = [j for j in P[i] if j >= 0]
            for pos, (j, action) in enumerate(zip(items, labels(P[i], aux["rev"][i], rd.opened[i]))):
                pre = np.flatnonzero(rd.prereq["chain"][i, j])
                pk = min(pre, key=lambda p: m["K"][i, p]) if len(pre) else None
                info = {"M": m["M"][i, j], "R": m["R"][i, j] if np.isfinite(m["R"][i, j]) else 1.0, "K": m["K"][i, j],
                        "delta": rd.delta[i, j] if np.isfinite(rd.delta[i, j]) else 0, "p_hat": m["K"][i, j],
                        "prereq_name": topic_name(mod, pk) if pk is not None else None,
                        "prereq_K": m["K"][i, pk] if pk is not None else None, "is_rev": bool(aux["rev"][i, j])}
                terms = aux["terms"][i, j][:5]
                paths.append((who, ver, did, pos + 1, j + 1, action, float(terms @ w),
                              explain_item(topic_name(mod, j), terms, w, info)))
    st = pd.DataFrame(states, columns=["student", "model", "decision", "topic", "M", "M_is_predicted", "R", "K", "I",
                                       "status", "clicks"])
    pa = pd.DataFrame(paths, columns=["student", "model", "decision", "position", "topic", "action", "score",
                                      "explanation"])
    return dec, st, pa


def curve_frames(b, sel, length):
    """Daily K of every assessed topic (v2: M*R from the half-life, v3: M*R from activation) and practice events."""
    d, var = b["data"], b["var"]
    it = d["inter"]
    it = it[(it["is_submitted"] == 1) & it["sid"].isin(sel["sid"])]
    local = sel[["sid", "student", "code_module", "code_presentation", "prev_attempts"]].reset_index(drop=True)
    local["lsid"] = np.arange(len(local))
    last = (sel["date_unregistration"].fillna(length)).clip(upper=length).astype(int)
    last = dict(zip(sel["sid"], last))
    q = [(s, day) for s, first in it.groupby("sid")["t_submit"].min().items()
         for day in range(int(first) + 1, last[s] + 1)]
    vd = b["vd"][b["vd"]["sid"].isin(sel["sid"])]
    sub = it[["sid", "topic", "t_submit"]].rename(columns={"topic": "j", "t_submit": "ts"})
    ev = [sub.assign(day=sub["ts"], model="both", kind="Submission", value=1.0)]

    def events(frame, value, threshold, model):
        e = frame.rename(columns={"topic": "j", "date": "day", value: "value"})[["sid", "j", "day", "value"]]
        e = e[e["value"] > 0].merge(sub, on=["sid", "j"], how="left")
        rev = (e["day"] > e["ts"]) & (e["value"] >= threshold)
        return e.assign(kind=np.where(rev, "Revision", "Practice"), model=model)

    ev.append(events(vd, "clicks", 3, "v2"))
    pr = practice(b["cs"], d, b["var_attribution"])
    ev.append(events(pr[pr["sid"].isin(sel["sid"])], "dose", REVISION_DOSE, "v3"))
    ev = pd.concat(ev, ignore_index=True).merge(local[["sid", "student"]], on="sid")
    ev = ev[["student", "j", "day", "model", "kind", "value"]].rename(columns={"j": "topic"})
    ev = ev.astype({"topic": "int8", "day": "int16", "value": "float32"})
    if not q:
        return pd.DataFrame(columns=["student", "topic", "day", "v2", "v3"]), ev

    q = pd.DataFrame(q, columns=["sid", "t"]).merge(local[["sid", "lsid"]], on="sid")
    q["qid"] = np.arange(len(q))
    m = local[["sid", "lsid"]]
    relabel = lambda f: f.merge(m, on="sid").drop(columns="sid").rename(columns={"lsid": "sid"})  # noqa: E731
    ta = TopicArrays(relabel(it), d["topics"], relabel(local.drop(columns="lsid")), d["T"])
    cube = ClickCube(relabel(vd), len(local), d["T"], 3)
    st = compute_state(cube, ta, q[["qid", "lsid", "t"]].rename(columns={"lsid": "sid"}), [np.inf], {})
    f = st[st["assessed"]].merge(q[["qid", "sid", "t"]], on="qid")
    t = f["t"].to_numpy(float)
    R2, _ = b["learned"].retention_state(f)
    _, R3 = var.K_assessed(f, t, t)
    out = pd.DataFrame({"student": f["sid"].map(dict(zip(local["sid"], local["student"]))), "topic": f["j"],
                        "day": f["t"], "v2": f["M"] * R2, "v3": f["M"] * R3})
    return out.astype({"topic": "int8", "day": "int16", "v2": "float32", "v3": "float32"}), ev


def plan_frame(b, sel, frozen, sat):
    d = b["data"]
    s = d["students"]
    if not ((s["split"] == "test") & s["exam_score"].notna()).any():
        return None
    p = PN.plans(b["var"], d, "test", scale=frozen["v3"]["knowledge_scale"]["scale"])
    p = p.merge(sel[["sid", "student", "id_student"]], on="sid")
    p = p.merge(sat, on=["code_module", "code_presentation", "id_student"], how="left")
    p["after_exam"] = p["t"] >= p["day_sat"]
    return p.rename(columns={"j": "topic", "t": "day"})[
        ["student", "weeks_before", "day", "topic", "rank", "flagged", "M", "K_exam", "priority", "explanation",
         "day_sat", "t_exam", "after_exam"]]


# ------------------------------------------------------------------ static pages
def static_frames(frozen):
    weights = []
    for ver in ("v2", "v3"):
        for cid in ("A3", "A5", "A6", "A8", "A9"):
            t = frozen[ver]["ablations"][cid]
            weights.append({"version": ver, "config": cid, "label": t["config"]["label"], **t["weights"],
                            "p_star": t["filters"]["p_star"], "tau_q": t["filters"]["tau_q"],
                            "q_rev": t["filters"]["q_rev"], "q_new": t["filters"]["q_new"]})
    index, cells = [], []
    for order, (key, t) in enumerate(RC.tables().items()):
        index.append({"order": order, "key": key, "title": t.title, "stage": t.stage, "evidence": t.evidence,
                      "validity": t.validity, "sources": ", ".join(t.sources)})
        for i, r in RC.frame(t).iterrows():
            for j, col in enumerate(t.columns):
                cells.append({"key": key, "row": i, "col_no": j, "col": col, "text": str(r[col])})
    crit = [{"id": c["id"], "name": c["name"], "v3": c["v3"].text(), "v2": c["v2"].text(), "effect": c["effect"].text(),
             "met": bool(RC.read(c["met"])), "evidence": c["evidence"], "reliable": c["reliable"]} for c in RC.criteria()]
    return {"weights": pd.DataFrame(weights), "result_index": pd.DataFrame(index), "result_cells": pd.DataFrame(cells),
            "criteria": pd.DataFrame(crit),
            "defects": pd.DataFrame(RC.DEFECTS, columns=["id", "name", "what", "affects"])}


def main():
    t0 = time.time()
    cfg = load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    DATA.mkdir(parents=True, exist_ok=True)
    ensure_intermediates(cfg)
    frozen = load_frozen()
    bundles = {}
    for cs in SETS:
        log(f"reproducing frozen fits: {cs}")
        b = fit_course(cs, frozen)
        check_coefficients(b)
        b["var_attribution"] = frozen["sel"]["attribution"]
        bundles[cs] = recommender_data(b, cfg)
    log("checking that the recorded A5 scores are reproduced")
    out = static_frames(frozen)
    out["integrity_checks"] = check_recorded_scores(bundles, frozen, cfg)

    lengths = pd.read_csv(ROOT / "data/raw/courses.csv").set_index(["code_module", "code_presentation"])[
        "module_presentation_length"]
    sat = exam_day_sat()
    parts = {k: [] for k in ("students", "decisions", "states", "paths", "plans")}
    for cs, mods in SETS.items():
        b = bundles[cs]
        chosen = {m: choose_students(b, m, seed) for m in mods}
        sel_all = pd.concat(chosen.values(), ignore_index=True)
        dec, st, pa = decision_frames(b, sel_all, frozen, cfg)
        for name, f in (("decisions", dec), ("states", st), ("paths", pa)):
            parts[name].append(f.assign(module=f["student"].str[:3]))
        plans = plan_frame(b, sel_all, frozen, sat)
        if plans is not None:
            parts["plans"].append(plans.assign(module=plans["student"].str[:3]))
        for m, sel in chosen.items():
            log(f"curves: {m} ({len(sel)} students)")
            cu, ev = curve_frames(b, sel, int(lengths[(m, "2014J")]))
            cu.to_parquet(DATA / f"curves_{m}.parquet", index=False, compression="zstd")
            ev.to_parquet(DATA / f"events_{m}.parquet", index=False, compression="zstd")
            has_exam = bool(b["data"]["students"].query("code_module == @m")["exam_score"].notna().any())
            # no demographics, outcomes or original ids are written
            parts["students"].append(pd.DataFrame({
                "student": sel["student"], "module": m, "presentation": "2014J", "n_clicks": sel["n_clicks"],
                "withdrawn": sel["withdrawn"], "withdrawn_day": sel["date_unregistration"],
                "module_has_exam": has_exam, "sat_exam": sel["exam_score"].notna()}))
    for name, fs in parts.items():
        out[name] = pd.concat(fs, ignore_index=True)

    # the default student: first id (by number) in DDD who stayed, sat the exam and has the most decision points
    s = out["students"]
    s = s[(s["module"] == "DDD") & s["sat_exam"] & ~s["withdrawn"]][["student"]].merge(
        out["decisions"].groupby("student").size().rename("n").reset_index())
    default = s.sort_values(["n", "student"], ascending=[False, True])["student"].iloc[0]
    out["meta"] = pd.DataFrame([{"default_student": default, "seed": seed, "n_per_module": N_PER_MODULE,
                                 "v3_scale": frozen["v3"]["knowledge_scale"]["kcol"]}])
    for name, f in out.items():
        f.to_parquet(DATA / f"{name}.parquet", index=False, compression="zstd")
    size = sum(p.stat().st_size for p in DATA.glob("*.parquet")) / 1e6
    log(f"wrote {len(list(DATA.glob('*.parquet')))} files, {size:.1f} MB, in {time.time() - t0:.0f} s")
    print(out["integrity_checks"].to_string(index=False))


if __name__ == "__main__":
    main()
