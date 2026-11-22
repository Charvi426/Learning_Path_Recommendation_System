"""Learning-path demo.    streamlit run demo/app.py

My learning path is the working prototype (demo/prototype_page.py, python demo/build_prototype.py): it computes
a path live from the student's input. The other pages replay the recorded v2/v3 analysis from demo/data/
(python demo/build_demo_data.py). The app never fits a model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import data_access as da  # noqa: E402
import prototype_page  # noqa: E402

st.set_page_config(page_title="Forgetting-aware learning paths", layout="wide")

TOPIC_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#8C6D31", "#555555"]
MODEL_NAMES = {"v2": "v2: learned half-life", "v3": "v3: practice history (ACT-R)"}


def validity_banner():
    st.warning(
        "**Research prototype with known defects.** The recorded accuracy figures include a label leak, the exam day "
        "used is too late, and v3 was not scored as preregistered. The paths show how the system works; they are not "
        "validated advice. The Results page has the details."
    )


def sidebar(with_decision: bool):
    default = da.default_student()
    mods = da.modules()
    ss = st.session_state
    ss.setdefault("module_value", default[:3])
    ss.setdefault("student_value", default)
    sb = st.sidebar
    module = sb.selectbox("Module (2014J test students)", mods, index=mods.index(ss["module_value"]))
    ids = da.students(module)
    student = sb.selectbox("Student (anonymised id)", ids,
                           index=ids.index(ss["student_value"]) if ss["student_value"] in ids else 0)
    ss["module_value"], ss["student_value"] = module, student
    decision = None
    if with_decision:
        dec = da.decisions(student)
        if len(dec):
            names = [da.decision_label(r) for _, r in dec.iterrows()]
            i = sb.selectbox("Decision week", range(len(dec)), index=len(dec) - 1, format_func=lambda k: names[k])
            decision = dec.iloc[i]
        else:
            sb.caption("No decision week exists for this student.")
        model = sb.radio("Knowledge model", list(MODEL_NAMES), index=1, format_func=MODEL_NAMES.get)
        return student, decision, model
    return student, None, None


def forgetting_chart(student: str, model: str, day: int | None):
    c = da.curve(student, model)
    if c.empty:
        st.info("No assessed topic yet, so there is no knowledge to track.")
        return
    topics = sorted(int(t) for t in c["topic"].unique())
    shown = st.multiselect("Topics to show", topics, default=topics)
    if not shown:
        st.info("Select at least one topic.")
        return
    c = c[c["topic"].isin(shown)].copy()
    ev = da.events(student, model)
    ev = ev[ev["topic"].isin(shown)].copy()
    c["Topic"] = "Topic " + c["topic"].astype(str)
    ev["Topic"] = "Topic " + ev["topic"].astype(str)
    # an event shows in K from the following day: mark it on the first point that includes it
    on_line = ev[ev["kind"] == "Revision"].assign(shown_day=lambda e: e["day"] + 1).merge(
        c[["topic", "day", "K"]].rename(columns={"day": "shown_day"}), on=["topic", "shown_day"])

    colour = alt.Color("Topic:N", scale=alt.Scale(domain=[f"Topic {t}" for t in topics],
                                                  range=TOPIC_COLOURS[: len(topics)]),
                       legend=alt.Legend(orient="top", title=None))
    lo, hi = int(min(c["day"].min(), ev["day"].min())), int(c["day"].max())
    x = alt.X("day:Q", title="Day of the course", scale=alt.Scale(domain=[lo, hi], nice=False))
    lines = alt.Chart(c).mark_line().encode(
        x=x, y=alt.Y("K:Q", title="Knowledge K = M x R", scale=alt.Scale(domain=[0, 1]),
                                 axis=alt.Axis(minExtent=75, maxExtent=75)), color=colour,
        tooltip=["Topic:N", "day:Q", alt.Tooltip("K:Q", format=".2f")])
    layers = [lines]
    if len(on_line):
        layers.append(alt.Chart(on_line).mark_point(shape="triangle-up", filled=True, size=45).encode(
            x=alt.X("shown_day:Q"), y="K:Q", color=colour, tooltip=["Topic:N", "day:Q", "kind:N"]))
    if day is not None:
        layers.append(alt.Chart(pd.DataFrame({"day": [day]})).mark_rule(strokeDash=[5, 3], color="grey").encode(x="day:Q"))
    top = alt.layer(*layers).properties(height=270, width="container")
    st.altair_chart(top, use_container_width=True)

    strip = alt.Chart(ev).mark_point(filled=True, opacity=0.75).encode(
        x=x, y=alt.Y("Topic:N", title="Events", axis=alt.Axis(minExtent=75, maxExtent=75)),
        color=alt.Color("kind:N", title=None, legend=alt.Legend(orient="top"), scale=alt.Scale(domain=["Practice", "Submission", "Revision"],
                                                                range=["#B8B8B8", "#222222", "#D55E00"])),
        shape=alt.Shape("kind:N", title=None, legend=alt.Legend(orient="top"), scale=alt.Scale(domain=["Practice", "Submission", "Revision"],
                                                                range=["circle", "diamond", "triangle-up"])),
        tooltip=["Topic:N", "day:Q", "kind:N"],
    ).properties(height=40 * len(shown) + 70, width="container")
    st.altair_chart(strip, use_container_width=True)
    if model == "v2":
        st.caption("In v2 a revision is a day after submission with at least 3 clicks on the topic's own resources. "
                   "It resets the clock, so K jumps back up and then decays. The dashed line is the chosen week.")
    else:
        st.caption("In v3 every day of attributed practice adds to the topic's activation; triangles mark the larger "
                   "doses after submission. One click on a shared resource counts for several topics. The dashed line "
                   "is the chosen week.")


def student_page():
    student, decision, model = sidebar(with_decision=True)
    info = da.student(student)
    st.title(f"Student {student}")
    validity_banner()
    for message in da.notices(info):
        st.info(message)

    if decision is not None:
        did = int(decision["decision"])
        left, right = st.columns([7, 5], gap="medium")
        with left:
            st.subheader("Knowledge state")
            s = da.knowledge_state(student, model, did)
            predicted = s["M_is_predicted"].to_numpy(bool)
            view = pd.DataFrame({
                "Topic": s["topic"],
                "Status": s["status"],
                "Mastery M": [f"{v:.0%}" + (" *" if p else "") for v, p in zip(s["M"], predicted)],
                "Retention R": ["–" if pd.isna(v) else f"{v:.0%}" for v in s["R"]],
                "Knowledge K": [f"{v:.0%}" for v in s["K"]],
                "Interest I": [f"{v:.2f}" for v in s["I"]],
            })
            st.table(view.set_index("Topic"))
            st.caption("Fading = flagged for revision: the weaker half of this student's assessed topics, by K. "
                       "Not started = not assessed yet (* M and K are then the predicted chance of passing).")
        with right:
            st.subheader("Recommended path")
            p = da.path(student, model, did)
            if p.empty:
                st.info("No topic passes the candidate filters at this decision point.")
            for _, r in p.iterrows():
                st.markdown(f"**{r['position']}. {r['action']}: topic {r['topic']}**  \n{r['explanation']}")
            if 0 < len(p) < 5:
                st.caption(f"{len(p)} of 5 places filled: only these topics pass the filters.")
            st.caption("For a revision item, 'predicted success' in the explanation is K, not a calibrated probability.")

        st.subheader("What changed: v2 against v3")
        m, text = da.what_changed(student, did)
        st.write(text)
        if len(m):
            pos = lambda v: "–" if pd.isna(v) else str(int(v))  # noqa: E731
            st.dataframe(pd.DataFrame({"Topic": m["topic"], "v2 position": m["position_v2"].map(pos),
                                       "v2 action": m["action_v2"].fillna("–"),
                                       "v3 position": m["position_v3"].map(pos),
                                       "v3 action": m["action_v3"].fillna("–"), "Change": m["change"]}),
                         hide_index=True, use_container_width=True)
        st.caption("v3 paths follow the frozen configuration (K = M x R). The recorded v3 ranking scores used raw K, "
                   "so they do not evaluate these paths.")

    st.subheader("Forgetting over time")
    forgetting_chart(student, model, None if decision is None else int(decision["day"]))


def planner_page():
    student, _, _ = sidebar(with_decision=False)
    st.title(f"Pre-exam revision plan: {student}")
    validity_banner()
    p, message = da.plan(student)
    if message:
        st.info(message)
        return
    used, sat = int(p["t_exam"].iloc[0]), p["day_sat"].iloc[0]
    if pd.notna(sat):
        st.error(f"The analysis took the exam to be on day {used}. This student's recorded sitting was day {int(sat)}. "
                 "Weeks on or after that day are marked and could not have helped.")
    st.caption("v3 model only. Each week lists the 3 topics with the largest expected loss M − K at the assumed exam "
               "day, with no further practice. The percentages are uncalibrated and the planner has not been shown to "
               "improve exam results.")
    for weeks, g in p.groupby("weeks_before", sort=False):
        late = bool(g["after_exam"].iloc[0])
        with st.container(border=True):
            st.subheader(f"{weeks} week{'s' if weeks > 1 else ''} before the assumed exam day (day {int(g['day'].iloc[0])})")
            if late:
                st.markdown(":red[**On or after the real exam: not a usable plan.**]")
            cols = st.columns(3)
            for col, (_, r) in zip(cols, g.iterrows()):
                col.metric(f"{r['rank']}. Topic {r['topic']}: predicted knowledge", f"{r['K_exam']:.0%}",
                           help=f"Score on its assessment: {r['M']:.0%}")
                col.write(r["explanation"])


def how_page():
    st.title("How it works")
    validity_banner()
    st.info("This page describes the recorded v2/v3 research pipeline. The working prototype uses a separate "
            "model, explained at the bottom of My learning path.")
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF3F8", fontname="Arial"];
  data [label="OULAD\\nscores and clicks"]; topics [label="Topics\\n(assessment windows)"];
  factors [label="Factors\\nengagement, ability,\\ninterest, prerequisites"];
  model [label="Knowledge model\\nM, R, K"]; cand [label="Candidates\\nnew + fading topics"];
  score [label="Weighted score"]; path [label="Ordered path\\n+ explanations"]; plan [label="Pre-exam planner"];
  data -> topics -> factors -> model -> cand -> score -> path; model -> plan; }""", use_container_width=True)

    st.subheader("The score")
    st.latex(r"S = \alpha G + \beta Q + \gamma F + \delta D + \varepsilon I + \zeta\,\mathrm{Pop}")
    w = da.table("weights")
    w = w[w["version"] == "v3"]
    a5 = w[w["config"] == "A5"].iloc[0]
    used = " + ".join(f"{a5[k]:.1f} {k}" for k in ["G", "Q", "F", "D", "I", "Pop"] if a5[k] > 0)
    st.markdown(f"The demo uses configuration **A5**. Its tuned v3 score is **S = {used}**, "
                f"with target success p* = {a5['p_star']:.1f}. All weights below are from config/final_config_v3.yaml.")
    st.dataframe(w[["config", "label", "G", "Q", "F", "D", "I", "Pop", "p_star"]], hide_index=True,
                 use_container_width=True)
    if a5["D"] == 1:
        st.info("With all weight on D, revision topics closest to p* come first. These are usually the best-retained "
                "ones, not the most forgotten.")

    st.subheader("What each factor means")
    st.markdown(
        "- **M, mastery:** the score on that topic's assessment. Before the assessment, the predicted chance of passing.\n"
        "- **R, retention:** how much of it is estimated to remain. v2 uses time since the last revision; v3 uses the "
        "whole practice history.\n"
        "- **K, knowledge:** M x R.\n"
        "- **G, gap:** 1 − M, how much is still to learn.\n"
        "- **Q, readiness:** whether the previous topic is still known well enough.\n"
        "- **F, forgetting:** M − K, what was learned and then lost.\n"
        "- **D, difficulty fit:** how close the chance of success is to the target p*.\n"
        "- **I, interest:** a behavioural proxy from optional, early and after-deadline activity. It is not measured "
        "interest, and its tuned weight is zero.\n"
        "- **Pop, popularity:** how much earlier cohorts used the topic at the same point of the course."
    )


def results_page():
    st.title("Results, with their status")
    st.error("**No model comparison on this page is established.** The numbers are exactly as recorded and have not "
             "been corrected. Each table says what limits it.")
    st.subheader("Known defects")
    for r in da.table("defects").to_dict("records"):
        st.markdown(f"**{r['id']}. {r['name']}.** {r['what']} *Affects:* {r['affects']}")

    st.subheader("Preregistered v3 criteria")
    cols = st.columns(3)
    for col, r in zip(cols, da.table("criteria").to_dict("records")):
        with col.container(border=True):
            st.markdown(f"**({r['id']}) {r['name']}**")
            st.markdown("### :green[Met]" if r["met"] else "### :red[Not met]")
            st.markdown(f"Recorded, {r['evidence']}  \nv3 {r['v3']}  \nv2 {r['v2']}  \nv3 minus v2 {r['effect']}")
            st.markdown(f"**Can it be relied on?** {r['reliable']}")

    for t in da.result_tables():
        if t["key"] == "criteria":
            continue
        st.subheader(t["title"])
        st.markdown(f"Stage: **{t['stage']}** · Evidence: **{t['evidence']}**")
        st.markdown(f":red[{t['validity']}]")
        st.dataframe(t["table"], hide_index=True, use_container_width=True)
        st.caption(f"Source: {t['sources']}")


try:
    da.table("meta")
    recorded = [st.Page(student_page, title="Student view", url_path="student"),
                st.Page(planner_page, title="Pre-exam planner", url_path="planner"),
                st.Page(how_page, title="How it works", url_path="how"),
                st.Page(results_page, title="Results", url_path="results")]
except da.DemoDataMissing:
    recorded = []       # the prototype does not need the recorded analysis

pages = {"Prototype": [st.Page(prototype_page.page, title="My learning path", url_path="my-path", default=True)]}
if recorded:
    pages["Past analysis (has bugs)"] = recorded
st.navigation(pages).run()
