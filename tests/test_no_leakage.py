"""No-leakage tests: every feature for a prediction at time t must depend only on events strictly before t,
and cohort statistics must come from the training split only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.build import TopicArrays, compute_state  # noqa: E402
from src.features.engagement import ClickCube  # noqa: E402
from src.features.interest import add_interest, add_share, fit_zstats, raw_signal  # noqa: E402

WINDOWS = [7, 14, 30, np.inf]
SKIP = {"qid", "j"}


def _toy(seed=0, n_students=6):
    rng = np.random.default_rng(seed)
    topics = pd.DataFrame({"code_module": "X", "code_presentation": "2013J", "topic": [1, 2, 3, 4],
                           "t_open": [-11.0, 20, 50, 90], "t_due": [20.0, 50, 90, 130]})
    students = pd.DataFrame({"sid": np.arange(n_students), "code_module": "X", "code_presentation": "2013J",
                             "prev_attempts": 0})
    inter = []
    for s in range(n_students):
        for k, due in zip([1, 2, 3, 4], [20, 50, 90, 130]):
            sub = rng.random() < 0.85
            inter.append({"sid": s, "topic": k, "t_due": due, "is_submitted": int(sub),
                          "t_submit": due - rng.integers(0, 5) if sub else np.nan,
                          "y_score": rng.random() if sub else np.nan})
    inter = pd.DataFrame(inter)
    inter["y"] = np.where(inter["is_submitted"] == 1, (inter["y_score"] >= 0.4).astype(float), np.nan)
    vd = []
    for s in range(n_students):
        for _ in range(150):
            d = int(rng.integers(-10, 140))
            c = float(rng.integers(1, 8))
            vd.append({"sid": s, "topic": int(rng.integers(1, 5)), "date": d, "clicks": c,
                       "vol_clicks": c if rng.random() < 0.4 else 0.0})
    vd = pd.DataFrame(vd).groupby(["sid", "topic", "date"], as_index=False).sum()
    return topics, students, inter, vd


def _state(topics, students, inter, vd, queries, dmin=None):
    ta = TopicArrays(inter, topics, students, 4)
    if dmin is not None:  # keep the same day origin after truncation
        pad = pd.DataFrame({"sid": [0, 0], "topic": [1, 1], "date": [dmin[0], dmin[1]], "clicks": [0.0, 0.0],
                            "vol_clicks": [0.0, 0.0]})
        vd = pd.concat([vd, pad], ignore_index=True)
    cube = ClickCube(vd, ta.n_sid, 4, 3)
    st = compute_state(cube, ta, queries, WINDOWS, {"X": 0.9})
    st = add_share(st, raw_signal(st))
    return st


def _assert_same(a, b):
    cols = [c for c in a.columns if c not in SKIP]
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind in "fc":
            assert np.allclose(x, y, equal_nan=True), f"feature {c} changed by future events"
        else:
            assert (x == y).all(), f"feature {c} changed by future events"


@pytest.mark.parametrize("t", [15.0, 21.0, 51.0, 75.0, 131.0])
def test_future_events_do_not_change_features(t):
    topics, students, inter, vd = _toy()
    q = pd.DataFrame({"qid": np.arange(len(students)), "sid": students["sid"], "t": t})
    base = _state(topics, students, inter, vd, q, dmin=(int(vd.date.min()), int(vd.date.max()) + 50))
    zs = fit_zstats(base.assign(code_module="X"))
    base = add_interest(base, zs)

    # perturb: drop every click on or after t, add random clicks after t, and change / hide every
    # outcome submitted on or after t
    rng = np.random.default_rng(1)
    vd2 = vd[vd["date"] < t].copy()
    extra = pd.DataFrame({"sid": rng.integers(0, len(students), 200), "topic": rng.integers(1, 5, 200),
                          "date": rng.integers(int(t), int(t) + 50, 200), "clicks": 50.0, "vol_clicks": 50.0})
    vd2 = pd.concat([vd2, extra]).groupby(["sid", "topic", "date"], as_index=False).sum()
    inter2 = inter.copy()
    fut = inter2["t_submit"] >= t
    inter2.loc[fut, "y_score"] = 1 - inter2.loc[fut, "y_score"]
    inter2.loc[fut, "y"] = 1 - inter2.loc[fut, "y"]
    inter2.loc[fut, "t_submit"] = inter2.loc[fut, "t_submit"] - 1  # still >= t - 1 ... keep >= t below
    inter2.loc[fut & (inter2["t_submit"] < t), "t_submit"] = t
    pert = _state(topics, students, inter2, vd2, q, dmin=(int(vd.date.min()), int(vd.date.max()) + 50))
    pert = add_interest(pert, zs)
    _assert_same(base, pert)


def test_strictly_before_t_boundary():
    """A click exactly on day t must not count for a prediction at t; a click on t - 1 must."""
    topics = pd.DataFrame({"code_module": "X", "code_presentation": "2013J", "topic": [1, 2, 3, 4],
                           "t_open": [-11.0, 20, 50, 90], "t_due": [20.0, 50, 90, 130]})
    students = pd.DataFrame({"sid": [0], "code_module": "X", "code_presentation": "2013J", "prev_attempts": 0})
    inter = pd.DataFrame({"sid": 0, "topic": [1], "t_due": [20.0], "is_submitted": [1], "t_submit": [30.0],
                          "y_score": [0.9], "y": [1.0]})
    vd = pd.DataFrame({"sid": 0, "topic": [2, 2, 2], "date": [-10, 29, 30], "clicks": [0.0, 5.0, 7.0],
                       "vol_clicks": 0.0})
    q = pd.DataFrame({"qid": [0, 1], "sid": [0, 0], "t": [30.0, 31.0]})
    ta = TopicArrays(inter, topics, students, 4)
    st = compute_state(ClickCube(vd, 1, 4, 3), ta, q, WINDOWS, {"X": 0.9})
    t2 = st[st["j"] == 2].set_index("qid")
    assert t2.loc[0, "clicks_raw_inf"] == 5.0 and t2.loc[1, "clicks_raw_inf"] == 12.0
    t1 = st[st["j"] == 1].set_index("qid")
    assert not t1.loc[0, "assessed"] and t1.loc[1, "assessed"]   # submitted on day 30
    assert t1.loc[1, "delta"] == 1.0


@pytest.mark.skipif(not (ROOT / "data/processed/state_val.parquet").exists(), reason="run scripts/01-02 first")
def test_real_data_truncation_sample():
    """On real data: recompute the state of sampled validation queries from data truncated at t."""
    P = ROOT / "data/processed"
    topics = pd.read_parquet(P / "topics.parquet")
    students = pd.read_parquet(P / "students.parquet")
    inter = pd.read_parquet(P / "interactions.parquet")
    vd = pd.read_parquet(P / "vle_daily.parquet", columns=["sid", "topic", "date", "clicks", "vol_clicks"])
    q = pd.read_parquet(P / "queries.parquet")
    full = pd.read_parquet(P / "state_val.parquet")
    rng = np.random.default_rng(0)
    sample = q[(q["split"] == "val")].sample(25, random_state=0)
    for _, r in sample.iterrows():
        sid, t = int(r["sid"]), float(r["t"])
        stu = students[students["sid"] == sid].assign(sid=0)
        it = inter[inter["sid"] == sid].assign(sid=0)
        fut = it["t_submit"] >= t
        it.loc[fut, ["t_submit", "y", "y_score"]] = np.nan
        it.loc[fut, "is_submitted"] = 0
        v = vd[(vd["sid"] == sid) & (vd["date"] < t)].assign(sid=0)
        pad = pd.DataFrame({"sid": 0, "topic": 1, "date": [vd.date.min(), vd.date.max()], "clicks": 0.0,
                            "vol_clicks": 0.0})
        v = pd.concat([v, pad], ignore_index=True)
        tp = topics.merge(stu[["code_module", "code_presentation"]], on=["code_module", "code_presentation"])
        ta = TopicArrays(it, tp, stu, 6)
        cmy = {stu["code_module"].iloc[0]: float(full["ability"].max())}  # value irrelevant for comparison below
        st = compute_state(ClickCube(v, 1, 6, 3), ta, pd.DataFrame({"qid": [0], "sid": [0], "t": [t]}),
                           [7, 14, 30, np.inf], cmy)
        ref = full[full["qid"] == r["qid"]].sort_values("j")
        for c in ["c_7", "c_30", "c_inf", "d_inf", "vol", "early", "post", "M", "delta", "dt_last", "n_assessed"]:
            assert np.allclose(st[c].to_numpy(float), ref[c].to_numpy(float), equal_nan=True), (c, sid, t)


@pytest.mark.skipif(not (ROOT / "data/processed/topic_difficulty.csv").exists(), reason="run scripts/01-02 first")
def test_cohort_statistics_use_training_split_only():
    from src.features.difficulty import topic_difficulty
    P = ROOT / "data/processed"
    inter = pd.read_parquet(P / "interactions.parquet")
    saved = pd.read_csv(P / "topic_difficulty.csv")
    recomputed = topic_difficulty(inter[inter["split"] == "train"])
    assert np.allclose(saved["diff"], recomputed["diff"])
    # perturbing val/test outcomes must not change the cohort statistics
    inter2 = inter.copy()
    m = inter2["split"] != "train"
    inter2.loc[m, "y"] = 1 - inter2.loc[m, "y"]
    assert np.allclose(topic_difficulty(inter2[inter2["split"] == "train"])["diff"], saved["diff"])
    zs = pd.read_csv(P / "interest_zstats.csv")
    q = pd.read_parquet(P / "queries.parquet")
    st = pd.read_parquet(P / "state_train.parquet")
    trq = q[(q["split"] == "train") & (q["kind"] == "due")]["qid"]
    z2 = fit_zstats(st[st["qid"].isin(trq)])
    assert np.allclose(z2["mean"].to_numpy(), zs["mean"].to_numpy())
