"""Train the prototype's forecast and prepare its 2014J course data.

  python demo/build_prototype.py        (needs the seven OULAD CSVs in data/raw/)

A new model for the demo only. It does not read or write results/, config/ or models/, and it is not a correction of
the v1-v3 experiments.

  question   on day t, will this student submit and pass topic k (not yet submitted)?
  rows       every 7 days while registered, up to the earlier of the submission and the due date
  inputs     prototype_engine.features: events before day t only
  label      1 = submitted with at least 40; 0 = failed or never submitted
  training   presentations before 2014J only. 2014J outcomes are never used as labels, for fitting or for scoring.
  validation latest earlier presentation, on students who are not also in its training data. AAA and CCC have one
             earlier presentation, so they use 5-fold cross-validation by student.
  tuning     none (C = 1, fixed in advance)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import prototype_engine as E  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = HERE / "data" / "prototype"
MODEL = HERE / "prototype" / "model.json"
SERVE = "2014J"
EVERY_DAYS = 7
N_BOOT = 200
SEED = 42


def log(msg):
    print(f"[prototype] {msg}", flush=True)


def training_presentations(presentations) -> list[str]:
    return sorted(p for p in presentations if p < SERVE)       # 2013B < 2013J < 2014B < 2014J


def student_rows(history: E.History, course: E.Course, outcome: dict, excluded: set, registered: float,
                 withdrawn: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, is_next) for one student. outcome = {topic: (completion day, score)}."""
    topics = np.array([k for k in range(1, course.n + 1) if k not in excluded], int)
    if not len(topics):
        return np.empty((0, len(E.FEATURES))), np.empty(0), np.empty(0, bool)
    done = np.array([outcome.get(k, (np.inf, 0))[0] for k in topics])
    y = np.array([float(k in outcome and outcome[k][1] >= E.PASS_MARK) for k in topics])
    until = np.minimum(done, course.due[topics - 1])           # t <= submission: the label is never an input
    X, Y, nxt = [], [], []
    for t in range(0, int(course.due[-1]) + 1, EVERY_DAYS):
        if t < registered:
            continue
        if t >= withdrawn:
            break
        at_risk = t <= until
        if at_risk.any():
            X.append(E.features(history, course, t, topics[at_risk]))
            Y.append(y[at_risk])
            nxt.append(np.arange(at_risk.sum()) == 0)
    if not X:
        return np.empty((0, len(E.FEATURES))), np.empty(0), np.empty(0, bool)
    return np.vstack(X), np.concatenate(Y), np.concatenate(nxt)


def read_clicks():
    """Daily totals for every presentation; site-level rows for 2014J only (what the app replays)."""
    key = ["code_module", "code_presentation"]
    daily, serve = [], []
    dtype = {"code_module": str, "code_presentation": str, "id_student": "int32", "id_site": "int32",
             "date": "int16", "sum_click": "int32"}
    for ch in pd.read_csv(RAW / "studentVle.csv", dtype=dtype, chunksize=1_500_000):
        daily.append(ch.groupby(key + ["id_student", "date"], as_index=False)["sum_click"].sum())
        s = ch[ch["code_presentation"] == SERVE]
        if len(s):
            serve.append(s.groupby(key + ["id_student", "id_site", "date"], as_index=False)["sum_click"].sum())
    d = pd.concat(daily, ignore_index=True).groupby(key + ["id_student", "date"], as_index=False)["sum_click"].sum()
    s = pd.concat(serve, ignore_index=True).groupby(key + ["id_student", "id_site", "date"], as_index=False)[
        "sum_click"].sum()
    return d.rename(columns={"date": "day", "sum_click": "clicks"}), s.rename(columns={"date": "day", "sum_click": "clicks"})


def presentation_rows(course, students, records, daily) -> dict:
    recs = dict(tuple(records.groupby("id_student")))
    clicks = {s: (g["day"].to_numpy(), g["clicks"].to_numpy()) for s, g in daily.groupby("id_student")}
    X, y, nxt, who = [], [], [], []
    none = records.iloc[0:0]
    for r in students.itertuples():
        mine = recs.get(r.id_student, none)
        res = E.topic_results(mine, course)
        history = E.History(res, *clicks.get(r.id_student, ([], [])))
        unusable = mine[(mine["is_banked"] == 1) | mine["score"].isna()]
        excluded = set(course.parts[course.parts["id_assessment"].isin(unusable["id_assessment"])]["topic"])
        a, b, c = student_rows(history, course, {int(k): (d, s) for k, d, s in res.itertuples(index=False)}, excluded,
                               -np.inf if pd.isna(r.date_registration) else r.date_registration,
                               np.inf if pd.isna(r.date_unregistration) else r.date_unregistration)
        X.append(a.astype(np.float32))
        y.append(b)
        nxt.append(c)
        who.append(np.full(len(b), r.id_student))
    return {"X": np.vstack(X), "y": np.concatenate(y), "next": np.concatenate(nxt).astype(bool),
            "student": np.concatenate(who)}


def fit(X, y, columns=slice(None)):
    sc = StandardScaler().fit(X[:, columns].astype(float))
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[:, columns].astype(float)), y)
    return sc, lr


def score(model, X, columns=slice(None)):
    sc, lr = model
    return lr.predict_proba(sc.transform(X[:, columns].astype(float)))[:, 1]


def validate(data: dict) -> dict:
    pres = sorted(data)
    base = list(range(len(E.SCHEDULE_FEATURES)))
    ys, ps, bs, who, next_ = [], [], [], [], []
    if len(pres) > 1:
        tr = [data[p] for p in pres[:-1]]
        X, y = np.vstack([d["X"] for d in tr]), np.concatenate([d["y"] for d in tr])
        va = data[pres[-1]]
        keep = ~np.isin(va["student"], np.concatenate([d["student"] for d in tr]))       # no student in both
        folds = [(X, y, va["X"][keep], va["y"][keep], va["student"][keep], va["next"][keep])]
        method = f"trained on {', '.join(pres[:-1])}; validated on {pres[-1]}"
    else:
        d = data[pres[0]]
        folds = [(d["X"][a], d["y"][a], d["X"][b], d["y"][b], d["student"][b], d["next"][b])
                 for a, b in GroupKFold(5).split(d["X"], d["y"], d["student"])]
        method = f"5-fold cross-validation by student within {pres[0]}"
    for X, y, Xv, yv, sv, nv in folds:
        ps.append(score(fit(X, y), Xv))
        bs.append(score(fit(X, y, base), Xv, base))
        ys.append(yv)
        who.append(sv)
        next_.append(nv)
    y, p, b, who, next_ = map(np.concatenate, (ys, ps, bs, who, next_))

    uniq, inv = np.unique(who, return_inverse=True)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        boot.append(roc_auc_score(y, p, sample_weight=w))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == i].sum() - y[bins == i].sum()) for i in range(10)) / len(y)
    return {"method": method, "n_students": int(len(uniq)), "n_rows": int(len(y)), "pass_rate": float(y.mean()),
            "auc": float(roc_auc_score(y, p)), "auc_lo": float(np.percentile(boot, 2.5)),
            "auc_hi": float(np.percentile(boot, 97.5)), "auc_next_topic": float(roc_auc_score(y[next_], p[next_])),
            "auc_calendar_only": float(roc_auc_score(y, b)), "brier": float(np.mean((p - y) ** 2)),
            "brier_calendar_only": float(np.mean((b - y) ** 2)), "calibration_error": float(ece)}


def main():
    if not (RAW / "studentVle.csv").exists():
        raise SystemExit("Put the seven OULAD CSVs in data/raw/ first.")
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    read = lambda n: pd.read_csv(RAW / f"{n}.csv", na_values=["?"])  # noqa: E731
    courses, assess, vle, reg = read("courses"), read("assessments"), read("vle"), read("studentRegistration")
    records = read("studentAssessment").rename(columns={"date_submitted": "day"})
    records = records.merge(assess[assess["assessment_type"] != "Exam"][
        ["id_assessment", "code_module", "code_presentation"]], on="id_assessment")
    log("reading studentVle in chunks")
    daily, serve = read_clicks()
    of = lambda df, m, p: df[(df["code_module"] == m) & (df["code_presentation"] == p)]  # noqa: E731

    models, topics, parts = {}, [], []
    for m, cs in courses.groupby("code_module"):
        length = dict(zip(cs["code_presentation"], cs["module_presentation_length"]))
        train = training_presentations(length)
        data = {}
        for p in train:
            data[p] = presentation_rows(E.make_course(m, p, length[p], assess), of(reg, m, p), of(records, m, p),
                                        of(daily, m, p))
        val = validate(data)
        X, y = np.vstack([data[p]["X"] for p in train]), np.concatenate([data[p]["y"] for p in train])
        sc, lr = fit(X, y)
        models[m] = {"features": E.FEATURES, "trained_on": train, "n_rows": int(len(y)),
                     "n_students": int(len(np.unique(np.concatenate([data[p]["student"] for p in train])))),
                     "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist(),
                     "intercept": float(lr.intercept_[0]), "validation": val}
        log(f"{m}: AUC {val['auc']:.3f} (calendar only {val['auc_calendar_only']:.3f}), {val['method']}")

        c = E.make_course(m, SERVE, length[SERVE], assess)
        topics.append(c.topics.assign(module=m, length=c.length))
        parts.append(c.parts.assign(module=m))
        sc2 = of(serve, m, SERVE)
        sc2[["id_student", "id_site", "day", "clicks"]].sort_values("id_student").to_parquet(
            OUT / f"clicks_{m}.parquet", index=False, compression="zstd")
        sc2.groupby(["id_site", "day"], as_index=False)["clicks"].sum().to_parquet(
            OUT / f"cohort_{m}.parquet", index=False, compression="zstd")
        of(records, m, SERVE)[["id_student", "id_assessment", "day", "score"]].to_parquet(
            OUT / f"records_{m}.parquet", index=False, compression="zstd")
        of(reg, m, SERVE)[["id_student", "date_unregistration"]].to_parquet(
            OUT / f"students_{m}.parquet", index=False, compression="zstd")
        of(vle, m, SERVE)[["id_site", "activity_type", "week_from"]].to_parquet(
            OUT / f"sites_{m}.parquet", index=False, compression="zstd")

    pd.concat(topics).to_parquet(OUT / "topics.parquet", index=False)
    pd.concat(parts).to_parquet(OUT / "parts.parquet", index=False)
    MODEL.write_text(json.dumps({"serves": SERVE, "every_days": EVERY_DAYS, "models": models}, indent=1) + "\n",
                     encoding="utf-8", newline="\n")
    log(f"saved {MODEL.relative_to(ROOT).as_posix()} and {OUT.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
