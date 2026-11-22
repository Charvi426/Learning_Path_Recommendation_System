"""v3 Step 3: ACT-R activation behaves as specified and never uses events at or after the cut-off."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v3.memory import Series  # noqa: E402

FIXED = {"rule": "fixed", "d": 0.5, "x_sub": 2.0}
SPACING = {"rule": "spacing", "a": 0.177, "c": 0.217, "x_sub": 2.0}


def _series(days, doses=None, t_submit=None):
    doses = doses or [1.0] * len(days)
    pr = pd.DataFrame({"sid": 0, "topic": 1, "date": days, "dose": doses})
    inter = pd.DataFrame({"sid": [0], "topic": [1], "t_submit": [t_submit if t_submit is not None else -1000.0],
                          "is_submitted": [1 if t_submit is not None else 0]})
    return Series(pr, inter)


def _m(s, t_cut, t_eval, params):
    return float(s.activation(np.array([0]), np.array([t_cut], float), np.array([t_eval], float), params)[0])


def test_activation_falls_monotonically_without_practice():
    for params in [FIXED, SPACING]:
        s = _series([0, 10, 20], t_submit=25.0)
        m = [_m(s, 30, t, params) for t in range(30, 200, 10)]
        assert all(b < a for a, b in zip(m, m[1:]))


def test_activation_rises_after_a_practice_event():
    for params in [FIXED, SPACING]:
        s = _series([0, 10, 60], t_submit=25.0)
        before = _m(s, 60, 60, params)          # day-60 event not yet counted (strictly before t_cut)
        after = _m(s, 61, 61, params)
        assert after > before


def test_spacing_rule_rewards_spaced_practice_at_a_later_date():
    # same last event (day 50), first event either long before (spaced) or the day before (massed)
    spaced, massed = _series([0, 50]), _series([49, 50])
    for t in [150, 300]:
        assert _m(spaced, t, t, SPACING) > _m(massed, t, t, SPACING)


def test_no_event_at_or_after_cutoff_is_used():
    base = _series([0, 10, 20, 30], [1, 2, 1, 3], t_submit=15.0)
    pert = _series([0, 10, 20, 30, 40, 41, 45], [1, 2, 1, 3, 50, 50, 50], t_submit=15.0)
    for params in [FIXED, SPACING]:
        for t_cut, t_eval in [(31, 31), (31, 90), (40, 40), (40, 120)]:
            assert np.isclose(_m(base, t_cut, t_eval, params), _m(pert, t_cut, t_eval, params))
    # the submission itself is only counted strictly after its day
    s = _series([0], t_submit=15.0)
    assert _m(s, 15, 15, FIXED) < _m(s, 16, 16, FIXED)


def test_attributed_half_life_features_use_only_past_practice():
    s = _series([0, 10, 30, 31], [3.0, 3.0, 0.5, 3.0], t_submit=12.0)
    de, c, d = s.attributed_hl_features(np.array([0]), np.array([31.0]), np.array([12.0]))
    assert de[0] == 31 - 12           # day 30 dose < log1p(3); day 31 is not before t
    assert np.isclose(c[0], np.log1p(3.0 + 3.0 + 0.5)) and d[0] == 3
