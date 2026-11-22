"""The working prototype: no leakage (including the kind that broke v1-v3), valid paths, robust inputs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
import build_prototype as B  # noqa: E402
import prototype_data as D  # noqa: E402
import prototype_engine as E  # noqa: E402

built = pytest.mark.skipif(not (D.DATA / "topics.parquet").exists(), reason="run python demo/build_prototype.py first")
MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
NO_RESOURCES = pd.DataFrame({"id_site": pd.Series(dtype=int), "activity_type": pd.Series(dtype=str),
                             "topic": pd.Series(dtype=int)})
NO_ACTIVITY = pd.DataFrame(columns=["day", "clicks", "topic", "id_site"], dtype=float)
FLAT_MODEL = {"features": E.FEATURES, "mean": [0] * len(E.FEATURES), "scale": [1] * len(E.FEATURES),
              "coef": [0] * len(E.FEATURES), "intercept": 0.0}


def toy_course():
    topics = pd.DataFrame({"topic": [1, 2, 3, 4, 5], "t_open": [-26.0, 20, 50, 80, 110],
                           "t_due": [20.0, 50, 80, 110, 140], "kind": "TMA"})
    parts = pd.DataFrame({"id_assessment": [11, 12, 13, 14, 15, 16], "topic": [1, 2, 3, 4, 5, 5],
                          "weight": [10.0, 20, 20, 20, 15, 15]})
    return E.Course("TOY", "2014J", 160, topics, parts)


# ------------------------------------------------------------------ leakage
@pytest.mark.parametrize("target", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("early_by", [0, 1, 9])
def test_label_is_never_in_its_own_features_even_when_submitted_early(target, early_by):
    """The test that v1-v3 lacked (DR-1): change y_k, and no input of any row for k may change."""
    c = toy_course()
    day = {k: float(c.due[k - 1] - (early_by if k == target else 4)) for k in range(1, 6)}

    def rows(score):
        scores = {k: (score if k == target else 70.0) for k in day}
        res = pd.DataFrame({"topic": list(day), "day": list(day.values()), "score": list(scores.values())})
        h = E.History(res, [1, 10, 30, 60, 90, 120], [3, 4, 5, 6, 7, 8])
        X, y, _ = B.student_rows(h, c, {k: (day[k], scores[k]) for k in day}, set(), -50, np.inf)
        of_target = np.isclose(X[:, E.FEATURES.index("topic_position")], target / c.n)
        return X[of_target], y[of_target]

    (Xp, yp), (Xf, yf) = rows(95.0), rows(5.0)
    assert len(yp) and (yp == 1).all() and (yf == 0).all()
    np.testing.assert_array_equal(Xp, Xf)


def test_features_ignore_everything_on_or_after_today():
    c = toy_course()
    res = pd.DataFrame({"topic": [1, 2, 3], "day": [18.0, 48, 60], "score": [80.0, 60, 90]})
    changed = res.copy()
    changed.loc[2, "score"] = 1
    a = E.features(E.History(res, [5, 30, 59, 60, 70], [1, 2, 3, 4, 5]), c, 60, [3, 4, 5])
    b = E.features(E.History(changed, [5, 30, 59, 60, 70, 71], [1, 2, 3, 999, 999, 999]), c, 60, [3, 4, 5])
    np.testing.assert_array_equal(a, b)


def test_bundle_score_is_unknown_until_complete_or_due():
    c = toy_course()
    one = pd.DataFrame({"id_assessment": [15], "day": [120], "score": [80]})
    assert E.topic_results(one, c, 130).empty and E.topic_results(one, c, 140).empty
    assert E.topic_results(one, c, 141)["day"].tolist() == [140.0]
    both = pd.DataFrame({"id_assessment": [15, 16], "day": [120, 125], "score": [80, 60]})
    assert E.topic_results(both, c, 126)[["day", "score"]].values.tolist() == [[125.0, 70.0]]
    assert E.topic_results(both, c, 125).empty


def test_resource_map_uses_only_cohort_clicks_before_today():
    c = toy_course()
    sites = pd.DataFrame({"id_site": [1, 2, 3], "activity_type": ["oucontent", "homepage", "quiz"],
                          "week_from": [np.nan, np.nan, 4.0]})
    cohort = pd.DataFrame({"id_site": [1, 1, 2], "day": [10, 60, 10], "clicks": [5, 500, 50]})
    at40 = E.map_resources(c, sites, cohort, 40).set_index("id_site")["topic"].to_dict()
    assert at40 == {1: 1, 3: 2}
    assert E.map_resources(c, sites, cohort, 61).set_index("id_site")["topic"].to_dict() == {1: 3, 3: 2}
    assert 1 not in set(E.map_resources(c, sites, cohort, 10)["id_site"])


def test_2014j_is_never_used_for_training():
    assert B.training_presentations(["2013B", "2013J", "2014B", "2014J"]) == ["2013B", "2013J", "2014B"]
    models = json.loads(D.MODEL.read_text(encoding="utf-8"))["models"]
    assert sorted(models) == MODULES
    for m in models.values():
        assert "2014J" not in m["trained_on"] and "2014J" not in m["validation"]["method"]


@built
@pytest.mark.parametrize("module", MODULES)
def test_replay_hides_the_future(module):
    for day in (0, 45, 120):
        h = D.replay(module, D.example_student(module), day)
        for name in ("records", "results", "activity"):
            assert (h[name]["day"] < day).all(), name
        assert h["withdrew"] is None or h["withdrew"] < day


# ------------------------------------------------------------------ the path
def test_path_is_limited_to_selection_and_respects_prerequisites():
    c, rng = toy_course(), np.random.default_rng(0)
    for _ in range(3000):
        t = int(rng.integers(0, 150))
        done = [k for k in range(1, 6) if rng.random() < 0.5]
        res = pd.DataFrame({"topic": done, "day": [float(rng.integers(-5, max(t, 1))) for _ in done],
                            "score": [float(rng.integers(0, 101)) for _ in done]}, dtype=float)
        selected = [k for k in range(1, 6) if rng.random() < 0.5]
        st = E.topic_state(c, FLAT_MODEL, res, NO_ACTIVITY, NO_RESOURCES, t)
        needs = set(st[st["status"] != "Secure"]["topic"])
        for add in (False, True):
            steps, fine = E.learning_path(st, selected, t, NO_RESOURCES, NO_ACTIVITY, add)
            order = [s["topic"] for s in steps]
            assert len(order) == len(set(order)) and set(order) <= needs
            assert set(selected) == (set(order) & set(selected)) | set(fine)
            if not add:
                assert set(order) <= set(selected)
            for i, s in enumerate(steps):
                k = s["topic"]
                if s["action"] != "Revise":
                    if k - 1 in order:
                        assert order.index(k - 1) < i, (t, selected, order)
                    elif add and k > 1:
                        assert k - 1 not in needs, (t, selected, order)


def test_revision_resets_forgetting_and_changes_the_path():
    c = toy_course()
    res = pd.DataFrame({"topic": [1.0], "day": [15.0], "score": [85.0]})
    old = E.topic_state(c, FLAT_MODEL, res, NO_ACTIVITY, NO_RESOURCES, 100)
    assert old.loc[0, "status"] == "Fading"
    assert E.learning_path(old, [1], 100, NO_RESOURCES, NO_ACTIVITY)[0][0]["action"] == "Revise"
    act = pd.DataFrame({"day": [98.0], "clicks": [5.0], "topic": [1.0], "id_site": [np.nan]})
    new = E.topic_state(c, FLAT_MODEL, res, act, NO_RESOURCES, 100)
    assert new.loc[0, "status"] == "Secure"
    assert E.learning_path(new, [1], 100, NO_RESOURCES, act) == ([], [1])


def test_urgent_topic_comes_before_other_revision():
    c = toy_course()
    res = pd.DataFrame({"topic": [1.0, 2, 3], "day": [5.0, 10, 74], "score": [90.0, 90, 90]})
    st = E.topic_state(c, FLAT_MODEL, res, NO_ACTIVITY, NO_RESOURCES, 106)
    assert E.learning_path(st, [1, 2, 4], 106, NO_RESOURCES, NO_ACTIVITY)[0][0]["topic"] == 4


@built
@pytest.mark.parametrize("module", MODULES)
def test_predictions_respond_sensibly_to_scores_and_activity(module):
    c, model = D.course(module), D.models()[module]
    t = int(c.due[1]) + 3
    open_topics = list(range(3, c.n + 1))

    def p(score, days):
        res = pd.DataFrame({"topic": [1, 2], "day": c.due[:2] - 1, "score": float(score)})
        return E.predict(model, E.features(E.History(res, days, [20] * len(days)), c, t, open_topics))

    busy = list(range(t - 28, t, 2))
    assert (p(90, busy) > p(45, busy)).all()
    assert (p(70, busy) > p(70, [])).all()


@built
@pytest.mark.parametrize("module", MODULES)
def test_end_to_end_for_real_students_at_several_days(module):
    c, model, rng = D.course(module), D.models()[module], np.random.default_rng(1)
    for who in rng.choice(D.student_ids(module), 8, replace=False):
        for day in (0, 30, 100, 200, c.length):
            h, res = D.replay(module, int(who), day), D.resources(module, day)
            st = E.topic_state(c, model, h["results"], h["activity"], res, day)
            assert len(st) == c.n and st["p_pass"].dropna().between(0, 1).all()
            steps, fine = E.learning_path(st, range(1, c.n + 1), day, res, h["activity"])
            assert len(steps) + len(fine) == c.n and all(s["why"] for s in steps)


# ------------------------------------------------------------------ inputs
@built
def test_uploaded_files_give_the_same_result_as_the_replay():
    for m in MODULES:
        day, who = 110, D.example_student(m)
        h = D.replay(m, who, day)
        v = h["activity"].rename(columns={"day": "date", "clicks": "sum_click"}).to_csv(index=False).encode()
        s = h["records"].rename(columns={"day": "date_submitted"}).to_csv(index=False).encode()
        u = D.uploaded_history(D.read_vle_file(v, m), D.read_score_file(s, m), m, who, day)
        res = D.resources(m, day)
        a = E.topic_state(D.course(m), D.models()[m], h["results"], h["activity"], res, day)
        b = E.topic_state(D.course(m), D.models()[m], u["results"], u["activity"], res, day)
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("row,message", [
    ({"topic": 1, "day": 5, "score": 101}, "between 0 and 100"),
    ({"topic": 1, "day": np.nan, "score": 50}, "needs both"),
    ({"topic": 9, "day": 5, "score": 50}, "topics 1 to 5"),
    ({"topic": 1, "day": 5.5, "score": 50}, "whole numbers"),
])
def test_bad_scores_get_a_clear_message(row, message):
    with pytest.raises(E.InputError, match=message):
        E.check_results(pd.DataFrame([row], dtype=float), toy_course())


@pytest.mark.parametrize("row,message", [
    ({"topic": 1, "day": 5, "clicks": -1}, "negative"),
    ({"topic": 1, "day": np.nan, "clicks": 4}, "needs a day"),
    ({"topic": 7, "day": 5, "clicks": 4}, "topics 1 to 5"),
])
def test_bad_activity_gets_a_clear_message(row, message):
    with pytest.raises(E.InputError, match=message):
        E.check_activity(pd.DataFrame([row], dtype=float), toy_course())


@built
@pytest.mark.parametrize("content,message", [
    (b"", "could not be read"),
    (b"a,b\n1,2\n", "needs the columns"),
    (b"id_site,date,sum_click\n", "no rows"),
    (b"id_site,date,sum_click\n1,x,3\n", "must contain numbers"),
    (b"code_module,id_site,date,sum_click\nAAA,1,2,3\n", "no rows for DDD"),
])
def test_bad_files_get_a_clear_message(content, message):
    with pytest.raises(E.InputError, match=message):
        D.read_vle_file(content, "DDD")


@built
@pytest.mark.parametrize("source", ["OULAD student ID", "Type in my data", "Upload my files"])
@pytest.mark.parametrize("module", MODULES)
def test_page_renders_for_every_module_and_source(module, source):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(f"import sys\nsys.path.insert(0, {str(ROOT / 'demo')!r})\n"
                             "import prototype_page\nprototype_page.page()\n", default_timeout=60)
    at.session_state["proto-module"] = module
    at.session_state["proto-source"] = source
    at.run()
    assert not at.exception and not at.error
    if source != "Upload my files":
        assert any("learning path" in h.value for h in at.subheader)
