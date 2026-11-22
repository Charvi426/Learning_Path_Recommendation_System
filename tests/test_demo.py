"""The demo's data functions and pages work on 20 random students and on the awkward cases."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
import data_access as da  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "demo/data/meta.parquet").exists(),
                                reason="run python demo/build_demo_data.py first")


def _random_students(n=20):
    ids = sorted(da.table("students")["student"])
    return list(np.random.default_rng(20).choice(ids, n, replace=False))


def _edge_cases():
    s = da.table("students")
    with_decision = set(da.table("decisions")["student"])
    return {"no_clicks": s[s["n_clicks"] == 0]["student"].iloc[0],
            "withdrawn": s[s["withdrawn"]]["student"].iloc[0],
            "no_decision": s[~s["student"].isin(with_decision)]["student"].iloc[0],
            "no_exam_module": s[~s["module_has_exam"]]["student"].iloc[0],
            "no_exam_result": s[s["module_has_exam"] & ~s["sat_exam"]]["student"].iloc[0]}


def test_about_300_students_for_each_of_the_seven_modules():
    n = da.table("students").groupby("module").size()
    assert sorted(n.index) == ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
    assert (n == 300).all() and da.table("students")["student"].is_unique


def test_no_original_ids_demographics_or_outcomes_are_shipped():
    banned = {"sid", "id_student", "gender", "region", "imd_band", "disability", "age_band", "final_result",
              "exam_score", "y"}
    for name in ("students", "decisions", "states", "paths", "plans"):
        assert not banned & set(da.table(name).columns), name


@pytest.mark.parametrize("sid", _random_students() if (ROOT / "demo/data/meta.parquet").exists() else [])
def test_data_functions_for_random_students(sid):
    info = da.student(sid)
    assert sid in da.students(info["module"])
    assert isinstance(da.notices(info), list)
    for model in ("v2", "v3"):
        c = da.curve(sid, model)
        assert c["K"].between(0, 1).all()
        da.events(sid, model)
        for _, d in da.decisions(sid).iterrows():
            did = int(d["decision"])
            s = da.knowledge_state(sid, model, did)
            assert len(s) >= 4 and set(s["status"]) <= set(da.STATUS_ORDER)
            assert s["K"].between(0, 1).all() and s["M"].between(0, 1).all()
            p = da.path(sid, model, did)
            assert len(p) <= 5 and p["topic"].is_unique and (p["action"] == "Revise").sum() <= 3
            assert set(p["action"]) <= {"Revise", "Practice", "Learn"} and p["explanation"].str.len().gt(0).all()
            assert set(p["topic"]) <= set(s["topic"])
            # the chart and the state table must agree on K at the decision day
            done = s[~s["M_is_predicted"]][["topic", "K"]].merge(c[c["day"] == d["day"]][["topic", "K"]], on="topic")
            assert np.allclose(done["K_x"], done["K_y"], atol=1e-5)
            table, text = da.what_changed(sid, did)
            assert text and table["change"].notna().all()
    plan, message = da.plan(sid)
    assert (message is None) != plan.empty


def test_awkward_students_get_a_message_not_a_crash():
    e = _edge_cases()
    assert any("no clicks" in m for m in da.notices(da.student(e["no_clicks"])))
    assert any("withdrew" in m for m in da.notices(da.student(e["withdrawn"])))
    assert any("no decision point" in m for m in da.notices(da.student(e["no_decision"])))
    assert "no exam results" in da.plan(e["no_exam_module"])[1]
    assert "Plans exist only" in da.plan(e["no_exam_result"])[1]


def test_plans_have_at_most_three_topics_and_mark_weeks_after_the_exam():
    p = da.table("plans")
    assert p[p["flagged"]].groupby(["student", "weeks_before"]).size().max() <= 3
    ok = p.dropna(subset=["day_sat"])
    assert (ok["after_exam"] == (ok["day"] >= ok["day_sat"])).all() and ok["after_exam"].any()


def test_results_page_states_status_and_validity():
    c = da.table("criteria")
    assert c["met"].tolist() == [True, True, False]
    assert c["reliable"].str.startswith("No").all()
    for t in da.result_tables():
        assert t["stage"] and t["evidence"] and t["validity"] and len(t["table"])


@pytest.mark.parametrize("case", ["default", "no_clicks", "withdrawn", "no_decision", "no_exam_module", "no_exam_result"])
@pytest.mark.parametrize("page", ["student_page", "planner_page"])
def test_pages_render(case, page):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    sid = da.default_student() if case == "default" else _edge_cases()[case]
    source = (ROOT / "demo/app.py").read_text(encoding="utf-8").split("\ntry:\n    da.table")[0]
    source = source.replace("Path(__file__).resolve().parent", f"Path({str(ROOT / 'demo')!r})")
    at = AppTest.from_string(f"{source}\n{page}()\n", default_timeout=60)
    at.session_state["module_value"] = sid[:3]
    at.session_state["student_value"] = sid
    at.run()
    assert not at.exception


@pytest.mark.parametrize("page", ["how_page", "results_page"])
def test_static_pages_render(page):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    source = (ROOT / "demo/app.py").read_text(encoding="utf-8").split("\ntry:\n    da.table")[0]
    source = source.replace("Path(__file__).resolve().parent", f"Path({str(ROOT / 'demo')!r})")
    at = AppTest.from_string(f"{source}\n{page}()\n", default_timeout=60)
    at.run()
    assert not at.exception
