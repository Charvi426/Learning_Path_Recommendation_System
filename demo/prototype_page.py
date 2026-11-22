"""'My learning path': the working prototype's page. The path is computed from what is on the screen."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

import prototype_data as pdata
import prototype_engine as E

REPLAY, TYPED, UPLOAD = "OULAD student ID", "Type in my data", "Upload my files"


def _history(source: str, module: str, day: int, who_box) -> tuple[dict, str] | None:
    """The past data for the chosen source, and a key that changes when the editable tables must be refilled."""
    empty = {"results": pd.DataFrame(columns=["topic", "day", "score"], dtype=float),
             "activity": pd.DataFrame(columns=["day", "clicks", "topic", "id_site"], dtype=float)}
    if source == TYPED:
        st.caption("Fill in both tables below. Days are counted from the start of the course.")
        return empty, f"typed-{module}"

    if source == REPLAY:
        ids = pdata.student_ids(module)
        who = who_box.selectbox("Student ID (type to search)", ids, index=ids.index(pdata.example_student(module)),
                                   key=f"proto-student-{module}")
        h = pdata.replay(module, who, day)
        if h["withdrew"] is not None:
            st.info(f"This student withdrew on day {h['withdrew']}. The path shows what returning would involve.")
        with st.sidebar.expander("Download this history as files"):
            st.caption("The format that 'Upload my files' accepts.")
            st.download_button("VLE clicks (CSV)", h["activity"].rename(columns={"day": "date", "clicks": "sum_click"})
                               .to_csv(index=False), f"vle_{module}_{who}_day{day}.csv", "text/csv")
            st.download_button("Scores (CSV)", h["records"].rename(columns={"day": "date_submitted"})
                               .to_csv(index=False), f"scores_{module}_{who}_day{day}.csv", "text/csv")
        return h, f"replay-{module}-{who}-{day}"

    left, right = st.columns(2)
    vf = left.file_uploader("VLE clicks CSV", type="csv", key=f"proto-vle-{module}",
                            help="id_site, date, sum_click (as in OULAD studentVle), or topic, day, clicks")
    sf = right.file_uploader("Scores CSV", type="csv", key=f"proto-scores-{module}",
                             help="id_assessment, date_submitted, score (as in OULAD studentAssessment), or topic, day, score")
    if vf is None and sf is None:
        st.info("Upload a VLE file, a score file or both. Files are read in memory on this computer and are not saved.")
        return None
    vb, sb = (f.getvalue() if f is not None else b"" for f in (vf, sf))
    vle = pdata.read_vle_file(vb, module) if vf is not None else None
    scores = pdata.read_score_file(sb, module) if sf is not None else None
    ids = pdata.students_in(vle, scores)
    who = ids[0] if len(ids) == 1 else None
    if len(ids) > 1:
        who = st.selectbox("The files hold several students. Which one are you?", ids, index=None)
        if who is None:
            return None
    h = pdata.uploaded_history(vle, scores, module, who, day)
    if h["unknown_sites"]:
        st.warning(f"{h['unknown_sites']} click rows use resources that are not in {module} 2014J. They count as "
                   "general activity, but not towards a topic.")
    # refill the score table when the files or the student change, or when another uploaded result becomes known
    known = len(h["results"][h["results"]["day"] < day])
    return h, f"upload-{module}-{who}-{known}-{hashlib.sha1(vb + b'|' + sb).hexdigest()[:10]}"


def _editors(course: E.Course, h: dict, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = E.check_results(h["results"], course).set_index("topic")
    base = pd.DataFrame({"topic": course.topics["topic"].astype(int), "due": course.topics["t_due"].astype(int)})
    base["score"] = base["topic"].map(r["score"]).astype(float)
    base["day"] = base["topic"].map(r["day"]).astype(float)
    past = h["activity"]
    typing = key.startswith("typed")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Your scores**")
        scores = st.data_editor(
            base, key=f"scores-{key}", hide_index=True, use_container_width=True, disabled=["topic", "due"],
            column_config={"topic": st.column_config.NumberColumn("Topic"),
                           "due": st.column_config.NumberColumn("Due day"),
                           "score": st.column_config.NumberColumn("Score (0-100)", min_value=0.0, max_value=100.0),
                           "day": st.column_config.NumberColumn("Day submitted", step=1, format="%d")})
    with right:
        st.markdown("**Your VLE activity**")
        if len(past):
            st.caption(f"Loaded: {past['clicks'].sum():,.0f} clicks on {past['day'].nunique()} days. "
                       "Add rows to include more.")
        else:
            st.caption("One row for each day and topic you worked on." if typing else "No clicks loaded. You can add rows.")
        blank = pd.DataFrame({c: pd.Series(np.nan, index=range(4), dtype=float) for c in ("day", "topic", "clicks")})
        extra = st.data_editor(
            blank, key=f"activity-{key}", hide_index=True, use_container_width=True, num_rows="dynamic",
            column_config={"day": st.column_config.NumberColumn("Day", step=1, format="%d"),
                           "topic": st.column_config.NumberColumn("Topic", min_value=1, max_value=course.n, step=1,
                                                                  format="%d"),
                           "clicks": st.column_config.NumberColumn("Clicks", min_value=0, step=1, format="%d")})
    # a half-typed row is normal while typing: say so and leave it out, rather than failing
    half = scores["score"].isna() != scores["day"].isna()
    if half.any():
        left.caption(":orange[Add both the score and the day for topic "
                     + ", ".join(str(int(k)) for k in scores.loc[half, "topic"]) + ". It is left out until then.]")
        scores = scores[~half]
    extra = extra.dropna(how="all")
    half = extra["day"].isna() | extra["clicks"].isna()
    if half.any():
        right.caption(":orange[An activity row needs a day and a number of clicks. It is left out until then.]")
        extra = extra[~half]
    frames = [f.reindex(columns=["day", "clicks", "topic", "id_site"]).astype(float) for f in (past, extra) if len(f)]
    activity = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["day", "clicks", "topic", "id_site"], dtype=float)
    return scores[["topic", "day", "score"]], activity


def _show_path(steps: list[dict], fine: list[int], selected: list[int]):
    st.subheader("3. Your learning path")
    if not selected:
        st.info("Select at least one topic.")
        return
    if not steps:
        st.success("Every selected topic is secure. Nothing needs doing today.")
    for s in steps:
        with st.container(border=True):
            st.markdown(f"**{s['step']}. {s['action']}: topic {s['topic']}**  \n{s['why']}")
            if s["resources"]:
                st.caption(f"Start with: {s['resources']}")
    if steps and fine:
        st.caption("Nothing needed now for topic " + ", ".join(map(str, fine)) + ".")
    if steps:
        st.download_button("Download my path (CSV)", pd.DataFrame(steps).to_csv(index=False), "learning_path.csv",
                           "text/csv")


def _show_state(state: pd.DataFrame):
    st.subheader("Where you stand")
    pct = lambda v: "–" if pd.isna(v) else f"{v:.0%}"  # noqa: E731
    chance = lambda v: "<1%" if v < 0.01 else ">99%" if v > 0.99 else pct(v)  # noqa: E731
    st.table(pd.DataFrame({
        "Topic": state["topic"], "Status": state["status"], "Score": state["M"].map(pct),
        "Still known": state["K"].map(pct), "Chance to pass": state["p_pass"].map(chance),
    }).set_index("Topic"))


def _about(module: str, model: dict, half_life: int):
    v = model["validation"]
    with st.expander("How this works, and how far to trust it"):
        st.markdown(
            f"**Chance to pass (learned).** A logistic regression for {module}, trained on "
            f"{', '.join(model['trained_on'])} ({model['n_students']:,} students). It uses only scores, clicks and "
            "deadlines from before 'today', and predicts whether you will submit and score at least "
            f"{E.PASS_MARK}.\n\n"
            f"**Validation** ({v['method']}; {v['n_students']:,} students): AUC **{v['auc']:.3f}** "
            f"[{v['auc_lo']:.3f}, {v['auc_hi']:.3f}], against {v['auc_calendar_only']:.3f} from the calendar alone. "
            f"Calibration error {v['calibration_error']:.3f}. It has not been tested on 2014J.\n\n"
            f"**Still known (assumed).** Your score, halved every {half_life} days since you submitted or last revised "
            f"(a day with at least {E.REVISION_CLICKS} clicks on that topic). Forgetting is not measured in OULAD, so "
            "this is not validated.\n\n"
            "**Order (rules).** Anything due within a week and what it builds on, then revision from the weakest "
            "topic, then the other topics in course order. Nobody has tested whether following this order improves "
            "results.\n\n"
            "**Topics** are assessment periods. Resources are assigned to them from the course plan, or from what "
            "the cohort had used before today."
        )


def page():
    st.title("My learning path")
    st.caption("A working prototype: enter or load your past scores and VLE activity, choose topics, and get a path. "
               "It is decision support, not a grade prediction to act on alone.")
    try:
        mods = pdata.modules()
        models = pdata.models()
    except pdata.PrototypeDataMissing as e:
        st.error(str(e))
        return

    sb = st.sidebar
    source = sb.radio("Where is your data?", [REPLAY, TYPED, UPLOAD], key="proto-source")
    module = sb.selectbox("Module (2014J)", mods, key="proto-module")
    course = pdata.course(module)
    who_box = sb.container()
    day = int(sb.number_input("Today is day", min_value=0, max_value=course.length, value=min(100, course.length),
                              step=1, key=f"proto-day-{module}", help="Only what happened before this day is used."))
    with sb.expander("Settings"):
        half_life = st.slider("Assumed days until half is forgotten", 15, 180, E.HALF_LIFE, 5, key="proto-half-life")

    try:
        st.subheader("1. Your past scores and activity")
        got = _history(source, module, day, who_box)
        if got is None:
            return
        history, key = got
        results, activity = _editors(course, history, key)
        dated = pd.concat([results.dropna(subset=["score"])["day"], activity["day"]])
        later = int((dated >= day).sum())
        if later:
            st.warning(f"{later} entries are dated day {day} or later and are ignored, because today is day {day}.")
        if not (dated < day).any():
            st.info("No past data yet, so the forecast can only use the course calendar and the lack of activity.")

        st.subheader("2. Choose your topics")
        topic_key = f"proto-topics-{module}"
        if topic_key not in st.session_state:
            upcoming = int(np.searchsorted(course.due, day, side="left")) + 1
            st.session_state[topic_key] = list(range(1, min(upcoming, course.n) + 1))
        kinds = dict(zip(course.topics["topic"], course.topics["kind"]))
        selected = st.multiselect("Topics", list(range(1, course.n + 1)), key=topic_key,
                                  format_func=lambda k: f"Topic {k} ({kinds[k]}, due day {int(course.due[k - 1])})")
        add = st.checkbox("Also add earlier topics that these depend on", value=True, key="proto-prereq")

        res = pdata.resources(module, day)
        state = E.topic_state(course, models[module], results, activity, res, day, half_life)
        steps, fine = E.learning_path(state, selected, day, res, activity, add)
    except E.InputError as e:
        st.error(str(e))
        return

    left, right = st.columns([6, 5], gap="large")
    with left:
        _show_path(steps, fine, selected)
    with right:
        _show_state(state)
    _about(module, models[module], half_life)
