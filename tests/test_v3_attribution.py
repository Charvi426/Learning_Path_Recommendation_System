"""v3 Steps 1-2: attribution weights >= 0, sum to 1 per site, use only cohort pre-assessment (window) clicks;
navigation resources never contribute practice."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v3 import attribution as A  # noqa: E402

TOPICS = pd.DataFrame({"code_module": "X", "code_presentation": "2013J", "topic": [1, 2, 3],
                       "t_open": [-11.0, 30, 60], "t_due": [30.0, 60, 90]})
VLE = pd.DataFrame({"id_site": [1, 2, 3, 4, 5], "code_module": "X", "code_presentation": "2013J",
                    "activity_type": ["oucontent", "resource", "homepage", "oucontent", "oucontent"]})
TMAP = pd.DataFrame({"id_site": [1, 2, 3, 4, 5], "topic": [1, 2, np.nan, 3, 2]})
STU = pd.DataFrame({"code_module": "X", "code_presentation": "2013J", "id_student": [100, 200], "sid": [0, 1]})


def _svle():
    rows = []
    # site 1: clicked in windows 1 and 2 (pre-assessment) and heavily in the revision period (day 100)
    rows += [(100, 1, 10, 4), (100, 1, 40, 4), (200, 1, 100, 50)]
    rows += [(100, 2, 45, 3), (200, 2, 50, 3)]                   # site 2 only in window 2
    rows += [(100, 3, d, 5) for d in range(0, 100, 3)]            # homepage, every few days
    rows += [(100, 4, 70, 2), (200, 4, 80, 2)]                    # site 4: window 3 = final window
    rows += [(100, 5, 10, 1), (100, 5, 40, 1), (200, 5, 70, 1)]   # site 5: spread over all windows
    return pd.DataFrame([{"code_module": "X", "code_presentation": "2013J", "id_student": s, "id_site": i,
                          "date": d, "clicks": float(c)} for s, i, d, c in rows])


def _classes():
    return A.classify_resources(_svle(), VLE)


def test_homepage_is_navigation_and_content_types_are_not():
    rc = _classes().set_index("id_site")
    assert rc.loc[3, "resource_class"] == "NAVIGATION"
    assert (rc.loc[[1, 2, 4, 5], "resource_class"] == "CONTENT").all()


@pytest.mark.parametrize("mode", ["direct", "coaccess", "shared"])
def test_weights_nonnegative_and_sum_to_one(mode):
    static, shared = A.site_weights(_svle(), TOPICS, TMAP, _classes(), mode, 0.6)
    assert (static["w"] >= 0).all()
    assert np.allclose(static.groupby("id_site")["w"].sum(), 1.0)
    assert 3 not in set(static["id_site"]) and 3 not in set(shared["id_site"])     # navigation never weighted


def test_coaccess_uses_only_pre_assessment_window_clicks():
    sv = _svle()
    static, _ = A.site_weights(sv, TOPICS, TMAP, _classes(), "coaccess")
    w1 = static[static["id_site"] == 1].set_index("topic")["w"]
    # site 1: 4 clicks in window 1, 4 in window 2; the 50 revision-period clicks (day 100) must be ignored
    assert np.isclose(w1.get(1), 0.5) and np.isclose(w1.get(2), 0.5) and 3 not in w1.index
    # changing revision-period (after last due) clicks does not change any weight
    sv2 = sv.copy()
    sv2.loc[sv2["date"] > 90, "clicks"] *= 100
    static2, _ = A.site_weights(sv2, TOPICS, TMAP, _classes(), "coaccess")
    m = static.merge(static2, on=["id_site", "topic"])
    assert np.allclose(m["w_x"], m["w_y"]) and len(m) == len(static)


def test_shared_sites_spread_over_assessed_topics():
    static, shared = A.site_weights(_svle(), TOPICS, TMAP, _classes(), "shared", 0.6)
    assert 4 in set(shared["id_site"])        # final-window site
    assert 5 in set(shared["id_site"])        # spread co-access (entropy over 3 topics = 1)
    U = A.eligible_matrix(TOPICS, 0, 120)
    assert np.allclose(U.sum(1), 1)
    assert np.allclose(U[100 - 0], [1 / 3, 1 / 3, 1 / 3])   # after the last due date: all assessed topics
    assert np.allclose(U[45 - 0], [0.5, 0.5, 0])            # day 45: topic 1 assessed + window topic 2


def test_navigation_never_contributes_practice():
    rc = _classes()
    for mode in ["direct", "coaccess", "shared"]:
        static, shared = A.site_weights(_svle(), TOPICS, TMAP, rc, mode, 0.6)
        pr = A.practice_doses(_svle(), STU, TOPICS, static, shared, rc)
        sv = _svle()
        sv2 = sv.copy()
        sv2.loc[sv2["id_site"] == 3, "clicks"] *= 1000      # only navigation clicks change
        pr2 = A.practice_doses(sv2, STU, TOPICS, static, shared, rc)
        assert np.allclose(pr["dose"].to_numpy(), pr2["dose"].to_numpy())
        # student 100's homepage-only days carry no dose
        nav_days = set(range(0, 100, 3)) - {10, 40, 45, 70}
        assert pr[(pr["sid"] == 0) & pr["date"].isin(nav_days)].empty
