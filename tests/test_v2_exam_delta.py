"""v2 Fix 1: Δ at the exam date (Methodology 5.1). A later day with >= 3 clicks on j resets Δ, including days in
the revision period between the last assessment and the exam; days on or after the exam date are not used."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.build import TopicArrays, compute_state  # noqa: E402
from src.features.engagement import ClickCube  # noqa: E402

TOPICS = pd.DataFrame({"code_module": "X", "code_presentation": "2013J", "topic": [1, 2, 3],
                       "t_open": [-11.0, 30, 60], "t_due": [30.0, 60, 90]})
STU = pd.DataFrame({"sid": [0], "code_module": "X", "code_presentation": "2013J", "prev_attempts": 0})
INTER = pd.DataFrame({"sid": 0, "topic": [1, 2, 3], "t_due": [30.0, 60, 90], "is_submitted": 1,
                      "t_submit": [28.0, 58, 88], "y_score": [0.8, 0.7, 0.9], "y": 1.0})
T_EXAM = 130.0


def _delta(clicks_by_day: dict, topic=1, t=T_EXAM):
    days = sorted(clicks_by_day)
    vd = pd.DataFrame({"sid": 0, "topic": topic, "date": days + [-11, 200],
                       "clicks": [float(clicks_by_day[d]) for d in days] + [0.0, 0.0], "vol_clicks": 0.0})
    ta = TopicArrays(INTER, TOPICS, STU, 3)
    st = compute_state(ClickCube(vd, 1, 3, 3), ta, pd.DataFrame({"qid": [0], "sid": [0], "t": [t]}),
                       [7, 30, np.inf], {"X": 0.9})
    return float(st.loc[st["j"] == topic, "delta"].iloc[0])


def test_no_revision_delta_counts_from_submission():
    assert _delta({}) == T_EXAM - 28


def test_revision_period_day_with_3_clicks_resets_delta():
    # revision period = after the last due date (90) and before the exam (130)
    assert _delta({100: 3}) == T_EXAM - 100
    assert _delta({100: 3, 120: 5}) == T_EXAM - 120


def test_fewer_than_3_clicks_do_not_reset():
    assert _delta({120: 2}) == T_EXAM - 28


def test_exam_day_and_later_are_not_used():
    assert _delta({130: 9}) == T_EXAM - 28          # on the exam day: excluded (strictly before t)
    assert _delta({150: 9}) == T_EXAM - 28          # after the exam: excluded
    assert _delta({129: 3}) == 1.0                  # the day before the exam: counted


def test_clicks_before_submission_do_not_reset():
    assert _delta({20: 9}) == T_EXAM - 28
