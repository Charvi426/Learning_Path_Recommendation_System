"""v2 Fixes 2 and 3: Pop uses no events at or after t; relative candidate filters behave as specified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recommend import v2 as V2  # noqa: E402


class _RD:
    """Minimal stand-in for RecDataV2 (only the fields the functions read)."""

    def __init__(self, N=4, T=4):
        self.N, self.T = N, T
        self.module = np.array(["X"] * N)
        self.pres = np.array(["2013J"] * N)
        self.t = np.array([20.0, 40.0, 60.0, 80.0])[:N]
        self.exists = np.ones((N, T), bool)
        self.topic_no = np.arange(1, T + 1)


def _cohort_daily(rng, max_day=120):
    rows = [{"code_module": "X", "code_presentation": "2013J", "topic": int(rng.integers(1, 5)),
             "date": int(rng.integers(-10, max_day)), "clicks": float(rng.integers(1, 20))} for _ in range(400)]
    return pd.DataFrame(rows).groupby(["code_module", "code_presentation", "topic", "date"], as_index=False).sum()


def test_pop_cohort_uses_no_events_at_or_after_t():
    rng = np.random.default_rng(0)
    cd = _cohort_daily(rng)
    rd = _RD()
    for w in [7, 14, 30]:
        base = V2.pop_cohort_before(rd, cd, w)
        for i, t in enumerate(rd.t):
            # change every event on or after t (and keep the day range identical): Pop for row i must not move
            cd2 = cd.copy()
            fut = cd2["date"] >= t
            cd2.loc[fut, "clicks"] = cd2.loc[fut, "clicks"] * 100 + 7
            pert = V2.pop_cohort_before(rd, cd2, w)
            assert np.allclose(base[i], pert[i]), (w, t)


def test_pop_cohort_matches_manual_window_sum():
    rng = np.random.default_rng(1)
    cd = _cohort_daily(rng)
    rd = _RD()
    w = 14
    got = V2.pop_cohort_before(rd, cd, w)
    for i, t in enumerate(rd.t):
        g = cd[(cd["date"] >= t - w) & (cd["date"] <= t - 1)].groupby("topic")["clicks"].sum()
        raw = np.array([g.get(k, 0.0) for k in range(1, 5)])
        exp = raw / raw.max() if raw.max() > 0 else raw
        assert np.allclose(got[i], exp)
        assert got[i].max() <= 1 and got[i].min() >= 0


def test_pop_train_window_only_reads_training_statistics():
    # train_window is the row-max normalisation of rd.pop, which is built from training-split windows only
    # (scripts/04_tune_recommender.load_rec_inputs: rel[rel.split == 'train']); check the normalisation itself
    rd = _RD()
    rd.pop = np.array([[0, 5, 10, 0], [1, 1, 1, 1], [0, 0, 0, 0], [3, 0, 0, 6.0]])
    out = V2.pop_train_window(rd)
    assert np.allclose(out[0], [0, 0.5, 1, 0]) and np.allclose(out[2], 0) and out.max() <= 1
    src = (ROOT / "scripts" / "04_tune_recommender.py").read_text()
    assert 'rel[rel["split"] == "train"]' in src


def _filter_rd():
    rd = _RD(N=2, T=5)
    rd.assessed = np.array([[1, 1, 1, 1, 0], [1, 0, 0, 0, 0]], bool)
    rd.k_now = np.array([5, 2])
    return rd


def test_within_student_revision_flags_bottom_q_percent():
    rd = _filter_rd()
    K = np.array([[0.9, 0.2, 0.5, 0.7, 0.8], [0.4, 0.9, 0.8, 0.7, 0.6]])
    M = K.copy()
    Qn = np.ones_like(K)
    f = {"mode": "within", "q_rev": 50, "q_new": 10, "tau_q": 0.4}
    new, rev, excl = V2.relative_candidates(rd, {"M": M, "K": K}, Qn, f, horizon=2)
    assert rev[0].tolist() == [False, True, True, False, False]     # 2 lowest of 4 assessed (ceil(0.5*4) = 2)
    assert rev[1].tolist() == [True, False, False, False, False]    # ceil(0.5*1) = 1
    f["q_rev"] = 10
    _, rev, _ = V2.relative_candidates(rd, {"M": M, "K": K}, Qn, f, horizon=2)
    assert rev[0].tolist() == [False, True, False, False, False]     # ceil(0.1*4) = 1


def test_within_student_new_filter_excludes_top_mastery_only():
    rd = _filter_rd()
    K = np.array([[0.9, 0.2, 0.5, 0.7, 0.8], [0.4, 0.95, 0.8, 0.7, 0.6]])
    Qn = np.ones_like(K)
    f = {"mode": "within", "q_rev": 10, "q_new": 50, "tau_q": 0.4}
    new, rev, excl = V2.relative_candidates(rd, {"M": K, "K": K}, Qn, f, horizon=2)
    # student 1: unassessed within horizon (k <= 4) = topics 2, 3, 4; floor(0.5 * 3) = 1 excluded (highest = topic 2)
    assert excl[1].tolist() == [False, True, False, False, False]
    assert new[1].tolist() == [False, False, True, True, False]
    # student 0: pool = {topic 5}; floor(0.5 * 1) = 0 -> nothing excluded
    assert not excl[0].any() and new[0].tolist() == [False, False, False, False, True]


def test_cohort_quantile_filters_use_training_thresholds():
    rd = _filter_rd()
    K = np.array([[0.9, 0.2, 0.5, 0.7, 0.8], [0.4, 0.95, 0.8, 0.7, 0.6]])
    rd.cohort_thr = {"m": {("rev", 20): np.full((2, 5), 0.45), ("new", 20): np.full((2, 5), 0.75)}}
    f = {"mode": "cohort", "q_rev": 20, "q_new": 20, "tau_q": 0.4}
    new, rev, excl = V2.relative_candidates(rd, {"M": K, "K": K}, np.ones_like(K), f, horizon=2, model="m")
    assert rev[0].tolist() == [False, True, False, False, False] and rev[1].tolist() == [True] + [False] * 4
    assert excl[1].tolist() == [False, True, True, False, False]      # 0.95 and 0.8 above the 0.75 threshold
    assert excl[0].tolist() == [False, False, False, False, True]


def test_six_term_grid_and_admissibility():
    W = V2.weight_grid6(["G", "Q", "F", "D", "I", "Pop"])
    assert len(W) == 2211 and np.allclose(W.sum(1), 1) and (W[:, 4] <= 0.2 + 1e-9).all()
    assert len(V2.weight_grid6(["G", "Q", "F", "D", "Pop"])) == 1001
    ok = {"share_upcoming_removed_by_mastery_filter": 0.05, "share_dp_with_revision_flag": 0.5,
          "share_dp_mastery_filter_active": 0.2}
    assert V2.admissible(ok)
    assert not V2.admissible({**ok, "share_upcoming_removed_by_mastery_filter": 0.11})
    assert not V2.admissible({**ok, "share_dp_with_revision_flag": 0.0})
    assert not V2.admissible({**ok, "share_dp_mastery_filter_active": 0.0})
