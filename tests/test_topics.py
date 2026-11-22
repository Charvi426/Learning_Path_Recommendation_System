import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.topics import build_topics, topic_of_day, map_sites  # noqa: E402


def _assess():
    return pd.DataFrame({
        "code_module": ["X"] * 5, "code_presentation": ["2013J"] * 5,
        "id_assessment": [1, 2, 3, 4, 5], "assessment_type": ["TMA", "CMA", "TMA", "TMA", "Exam"],
        "date": [20.0, 20.0, 50.0, 90.0, 120.0], "weight": [10, 0, 20, 30, 100]})


def test_same_day_assessments_merge_and_exam_excluded():
    t = build_topics(_assess(), pd.Series({("X", "2013J"): -10}))
    assert list(t["topic"]) == [1, 2, 3]
    assert list(t["t_due"]) == [20, 50, 90]
    assert list(t["t_open"]) == [-11, 20, 50]           # first window starts at first VLE day
    assert t.loc[0, "n_assessments"] == 2 and sorted(t.loc[0, "assessment_ids"]) == [1, 2]


def test_topic_of_day_windows_are_left_open_right_closed():
    t_open = np.array([-11, 20, 50.0]); t_due = np.array([20, 50, 90.0])
    days = np.array([-11, 0, 20, 21, 50, 51, 90, 200])
    assert list(topic_of_day(days, t_open, t_due)) == [1, 1, 1, 2, 2, 3, 3, 3]


def test_site_mapping_week_from_then_cohort_clicks():
    t = build_topics(_assess(), pd.Series({("X", "2013J"): -10}))
    vle = pd.DataFrame({"id_site": [10, 11, 12, 13], "code_module": "X", "code_presentation": "2013J",
                        "activity_type": ["resource", "oucontent", "homepage", "forumng"],
                        "week_from": [4.0, np.nan, np.nan, np.nan], "week_to": [5.0, np.nan, np.nan, np.nan]})
    svle = pd.DataFrame({"code_module": "X", "code_presentation": "2013J", "id_student": 1,
                         "id_site": [11, 11, 11, 12, 13], "date": [5, 60, 61, 1, 30],
                         "clicks": [1.0, 5.0, 5.0, 9.0, 2.0]})
    m = map_sites(vle, t, svle, ["homepage"], ["forumng"]).set_index("id_site")
    assert m.loc[10, "topic"] == 2 and m.loc[10, "map_method"] == "week_from"   # day 28 -> (20, 50]
    assert m.loc[11, "topic"] == 3 and m.loc[11, "map_method"] == "cohort_clicks"
    assert pd.isna(m.loc[12, "topic"])                                            # navigation excluded
    assert m.loc[13, "topic"] == 2 and bool(m.loc[13, "voluntary"])


@pytest.mark.skipif(not (ROOT / "data/processed/topics.parquet").exists(), reason="run scripts/01 first")
def test_processed_topics_have_at_least_five_windows_and_increasing_dates():
    t = pd.read_parquet(ROOT / "data/processed/topics.parquet")
    for _, g in t.groupby(["code_module", "code_presentation"]):
        assert len(g) >= 5
        assert (np.diff(g.sort_values("topic")["t_due"]) > 0).all()
        assert (g["t_open"] < g["t_due"]).all()
