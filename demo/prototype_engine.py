"""Engine of the working prototype: past VLE clicks + scores -> learning path for the selected topics.

Independent of the frozen v1-v3 pipeline and of its known defects (DECISIONS.md DR-1 to DR-7).

The one rule: on day t only events with day < t exist. That holds for scores, clicks and the cohort clicks used to
place resources into topics. demo/build_prototype.py trains with the same `features` function that the app calls.

What is learned and what is not:
  - P(submit and pass) of an unassessed topic: a logistic regression per module, trained on earlier presentations.
  - retention R = 2^(-days / half-life) is an ASSUMPTION (adjustable), not a learned or validated quantity.
  - the order of the path follows the stated rules in `learning_path`, not a tuned score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PASS_MARK = 40
COURSE_START = -25              # the VLE opens a few weeks before day 0
REVISION_CLICKS = 3             # clicks on a topic in one day that count as revising it
READY = 0.6                     # K below this means revise (the starting threshold in Methodology 6.1)
HALF_LIFE = 60                  # days; an assumption
URGENT_DAYS = 7
NAVIGATION = ("homepage", "forumng", "subpage", "glossary")
TYPE_ORDER = ("oucontent", "quiz", "externalquiz", "resource", "page", "url")

FEATURES = [
    "course_progress", "topic_position", "log_days_to_due", "unfinished_before",
    "has_score", "mean_score", "last_score", "share_due_missed",
    "log_clicks_per_week", "log_clicks_7d", "log_clicks_28d", "active_days_7d", "active_days_28d",
    "log_days_inactive",
]
SCHEDULE_FEATURES = FEATURES[:3]      # what the baseline may use: the calendar, nothing about the student


class InputError(ValueError):
    """A problem with what the student entered, worded for the student."""


@dataclass(frozen=True)
class Course:
    module: str
    presentation: str
    length: int
    topics: pd.DataFrame        # topic, t_open, t_due, kind
    parts: pd.DataFrame         # id_assessment, topic, weight

    @property
    def n(self) -> int:
        return len(self.topics)

    @property
    def due(self) -> np.ndarray:
        return self.topics["t_due"].to_numpy(float)


def make_course(module: str, presentation: str, length: int, assessments: pd.DataFrame) -> Course:
    """Topics = windows between consecutive non-exam due dates; same-day assessments form one topic."""
    a = assessments[(assessments["code_module"] == module) & (assessments["code_presentation"] == presentation)
                    & (assessments["assessment_type"] != "Exam")].copy()
    if (module, presentation) == ("DDD", "2013B"):
        a = a[a["assessment_type"] != "CMA"]                # the only DDD presentation with quizzes (DECISIONS #5)
    due = np.sort(a["date"].unique())
    a["topic"] = np.searchsorted(due, a["date"]) + 1
    topics = pd.DataFrame({"topic": np.arange(1, len(due) + 1), "t_open": np.r_[COURSE_START - 1, due[:-1]],
                           "t_due": due, "kind": a.groupby("topic")["assessment_type"].first().to_numpy()})
    return Course(module, presentation, int(length), topics,
                  a[["id_assessment", "topic", "weight"]].reset_index(drop=True))


def topic_results(records: pd.DataFrame, course: Course, today: float = np.inf) -> pd.DataFrame:
    """Assessment-level records (id_assessment, day, score) -> topic results (topic, day, score) known before today.

    A topic with several assessments is complete when all are in. If some are missing it completes at the due date,
    with the ones submitted by then.
    """
    r = records[records["day"] < today].merge(course.parts, on="id_assessment").dropna(subset=["score"])
    out = []
    for k, g in r.groupby("topic"):
        day = g["day"].max()
        if g["id_assessment"].nunique() < (course.parts["topic"] == k).sum():
            day = max(day, course.due[int(k) - 1])
            if day >= today:
                continue
        w = g["weight"].to_numpy(float)
        out.append((int(k), float(day), float(np.average(g["score"], weights=w if w.sum() > 0 else None))))
    return pd.DataFrame(out, columns=["topic", "day", "score"]).astype({"topic": int, "day": float, "score": float})


class History:
    """One student's topic results and daily click totals, queried 'as known on day t'."""

    def __init__(self, results: pd.DataFrame, days, clicks):
        r = results.sort_values(["day", "topic"], kind="stable")
        self.topic = r["topic"].to_numpy(int)
        self.day = r["day"].to_numpy(float)
        self.score = r["score"].to_numpy(float)
        days, clicks = np.asarray(days, float), np.asarray(clicks, float)
        clicks, days = clicks[clicks > 0], days[clicks > 0]
        self.active, inv = np.unique(days, return_inverse=True)
        self.cum = np.r_[0.0, np.cumsum(np.bincount(inv, weights=clicks, minlength=len(self.active)))]


def features(h: History, course: Course, t: float, targets) -> np.ndarray:
    """One row per target topic, from events strictly before day t."""
    targets = np.asarray(targets, int)
    known = h.day < t
    done = np.zeros(course.n, bool)
    done[h.topic[known] - 1] = True
    scores = h.score[known] / 100
    overdue = course.due < t

    i = np.searchsorted(h.active, t, side="left")           # active days before t
    i7, i28 = np.searchsorted(h.active, [t - 7, t - 28], side="left")
    weeks = max((t - COURSE_START) / 7, 1.0)
    inactive = min(t - h.active[i - 1], 60) if i else 60

    student = [
        float(known.any()),
        scores.mean() if len(scores) else 0.0,
        scores[-1] if len(scores) else 0.0,
        (overdue & ~done).sum() / overdue.sum() if overdue.any() else 0.0,
        np.log1p(h.cum[i] / weeks),
        np.log1p(h.cum[i] - h.cum[i7]),
        np.log1p(h.cum[i] - h.cum[i28]),
        i - i7,
        i - i28,
        np.log1p(inactive),
    ]
    before = np.r_[0, np.cumsum(~done)]
    X = np.empty((len(targets), len(FEATURES)))
    X[:, 0] = t / course.length
    X[:, 1] = targets / course.n
    X[:, 2] = np.log1p(np.maximum(course.due[targets - 1] - t, 0))
    X[:, 3] = before[targets - 1] / course.n
    X[:, 4:] = student
    return X


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    if model["features"] != FEATURES:
        raise RuntimeError("The saved model does not match the engine. Run python demo/build_prototype.py")
    z = model["intercept"] + ((X - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(model["coef"])
    return 1 / (1 + np.exp(-z))


def map_resources(course: Course, sites: pd.DataFrame, cohort: pd.DataFrame, t: float) -> pd.DataFrame:
    """Resource -> topic as it could be known on day t: the planned week if the course gives one, otherwise the
    window in which the cohort has clicked it most *so far*. Navigation pages belong to no topic."""
    s = sites[~sites["activity_type"].isin(NAVIGATION)].copy()
    due = course.due
    window = lambda day: np.minimum(np.searchsorted(due, day, side="left") + 1, course.n)  # noqa: E731
    c = cohort[(cohort["day"] < t) & (cohort["day"] <= due[-1])]
    c = c.assign(topic=window(c["day"].to_numpy(float)))
    c = c.groupby(["id_site", "topic"], as_index=False)["clicks"].sum().sort_values(
        ["id_site", "clicks", "topic"], ascending=[True, False, True]).drop_duplicates("id_site")
    topic = s["id_site"].map(c.set_index("id_site")["topic"]).astype(float)
    planned = s["week_from"].notna()
    topic[planned] = window(7 * s.loc[planned, "week_from"].to_numpy(float))
    s["topic"] = topic
    return s.dropna(subset=["topic"]).astype({"topic": int})[["id_site", "activity_type", "topic"]]


def check_results(results: pd.DataFrame, course: Course) -> pd.DataFrame:
    r = results.copy()
    for c in ("topic", "day", "score"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r[r["score"].notna() | r["day"].notna()]
    if r.empty:
        return r.astype(float).astype({"topic": int})
    if r[["topic", "day", "score"]].isna().any().any():
        raise InputError("Each result needs both a score and the day it was submitted.")
    if not r["topic"].isin(course.topics["topic"]).all():
        raise InputError(f"{course.module} has topics 1 to {course.n}.")
    if r["topic"].duplicated().any():
        raise InputError("Give one score for each topic.")
    if not r["score"].between(0, 100).all():
        raise InputError("Scores must be between 0 and 100.")
    if (r["day"] % 1 != 0).any() or not r["day"].between(-400, course.length + 400).all():
        raise InputError("Submission days must be whole numbers, counted from the start of the course.")
    return r[["topic", "day", "score"]].astype({"topic": int, "day": float, "score": float})


def check_activity(activity: pd.DataFrame, course: Course) -> pd.DataFrame:
    a = activity.reindex(columns=["day", "clicks", "topic", "id_site"]).copy()
    for c in a:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a[a["day"].notna() | a["clicks"].notna()]
    if a[["day", "clicks"]].isna().any().any():
        raise InputError("Each activity row needs a day and a number of clicks.")
    if (a["clicks"] < 0).any():
        raise InputError("Clicks cannot be negative.")
    if (a["day"] % 1 != 0).any() or not a["day"].between(-400, course.length + 400).all():
        raise InputError("Activity days must be whole numbers, counted from the start of the course.")
    given = a["topic"].dropna()
    if not given.isin(course.topics["topic"]).all():
        raise InputError(f"{course.module} has topics 1 to {course.n}.")
    return a.astype(float)


def topic_state(course: Course, model: dict, results: pd.DataFrame, activity: pd.DataFrame, resources: pd.DataFrame,
                t: float, half_life: float = HALF_LIFE) -> pd.DataFrame:
    """The student's state on day t for every topic. Anything dated t or later is ignored."""
    res = check_results(results, course)
    act = check_activity(activity, course)
    res, act = res[res["day"] < t], act[act["day"] < t].copy()
    by_site = act["id_site"].map(resources.set_index("id_site")["topic"])
    act["topic"] = act["topic"].fillna(by_site)

    open_topics = [k for k, d in zip(course.topics["topic"], course.due) if k not in set(res["topic"]) and d >= t]
    prob = {}
    if open_topics:
        h = History(res, act["day"], act["clicks"])
        prob = dict(zip(open_topics, predict(model, features(h, course, t, open_topics))))

    done = res.set_index("topic")
    rows = []
    for r in course.topics.itertuples():
        k = int(r.topic)
        mine = act[act["topic"] == k]
        row = {"topic": k, "kind": r.kind, "due": int(r.t_due), "clicks": float(mine["clicks"].sum()),
               "M": np.nan, "R": np.nan, "K": np.nan, "p_pass": np.nan, "days_since": np.nan}
        if k in done.index:
            submitted = float(done.at[k, "day"])
            daily = mine[mine["day"] > submitted].groupby("day")["clicks"].sum()
            revised = daily[daily >= REVISION_CLICKS]
            since = t - (revised.index.max() if len(revised) else submitted)
            row["M"] = float(done.at[k, "score"]) / 100
            row["R"] = float(2 ** (-since / half_life))
            row["K"] = row["M"] * row["R"]
            row["days_since"] = since
            row["status"] = "Fading" if row["K"] < READY else "Secure"
            if row["M"] < PASS_MARK / 100:
                row["status"] = "Weak"
        elif r.t_due < t:
            row["status"] = "Missed"
        else:
            row["status"] = "In progress" if row["clicks"] > 0 else "Not started"
            row["p_pass"] = float(prob[k])
        rows.append(row)
    return pd.DataFrame(rows)


def _steps(resources: pd.DataFrame, activity: pd.DataFrame, topic: int, t: float, revise: bool, n: int = 3) -> str:
    r = resources[resources["topic"] == topic].copy()
    if r.empty:
        return ""
    seen = activity[activity["day"] < t].groupby("id_site")["day"].max()
    r["last"] = r["id_site"].map(seen)
    r["rank"] = r["activity_type"].map({a: i for i, a in enumerate(TYPE_ORDER)}).fillna(len(TYPE_ORDER))
    r["opened"] = r["last"].notna()
    # revising: what was opened longest ago; learning: what has not been opened yet
    by = ["opened", "rank", "last", "id_site"] if revise else ["opened", "rank", "id_site"]
    r = r.sort_values(by, ascending=[not revise] + [True] * (len(by) - 1)).head(n)
    return "; ".join(f"{a} {int(i)} ({'opened before' if o else 'not opened yet'})"
                     for a, i, o in zip(r["activity_type"], r["id_site"], r["opened"]))


def learning_path(state: pd.DataFrame, selected, t: float, resources: pd.DataFrame, activity: pd.DataFrame,
                  add_prerequisites: bool = True) -> tuple[list[dict], list[int]]:
    """Returns (ordered steps, selected topics that need nothing now).

    1. A selected topic needs a step unless it is assessed and secure.
    2. An unassessed topic needs its previous topic: if that one is fading, weak or unassessed it is added
       (recursively while unassessed).
    3. Order: anything due within 7 days, preceded by what it needs; then revision, weakest K first; then the
       remaining topics in course order. A topic therefore never precedes a step it depends on.
    """
    s = state.set_index("topic")
    selected = sorted({int(k) for k in selected})
    unknown = set(selected) - set(s.index)
    if unknown:
        raise InputError(f"Unknown topics: {sorted(unknown)}")
    assessed = s["M"].notna().to_dict()
    needs = (s["status"] != "Secure").to_dict()

    def prerequisites(k) -> list[int]:
        out, j = [], k - 1
        while not assessed[k] and j >= 1 and needs[j]:
            out.append(j)
            if assessed[j]:
                break
            j -= 1
        return out

    wanted = {k for k in selected if needs[k]}
    nothing_to_do = [k for k in selected if not needs[k]]
    added = {}
    if add_prerequisites:
        for k in sorted(wanted):
            for j in prerequisites(k):
                if j not in wanted:
                    added.setdefault(j, k)
    items = wanted | set(added)

    first = set()
    for k in items:
        if not assessed[k] and 0 <= s.at[k, "due"] - t <= URGENT_DAYS:
            first |= {k, *prerequisites(k)} & items
    later = items - first
    order = sorted(first)
    order += sorted((k for k in later if assessed[k]), key=lambda k: (s.at[k, "K"], k))
    order += sorted(k for k in later if not assessed[k])

    act = activity.reindex(columns=["day", "clicks", "topic", "id_site"]).astype(float)
    steps = []
    for k in order:
        r = s.loc[k]
        days = int(r["due"] - t)
        if assessed[k]:
            action = "Revise"
            if r["status"] == "Weak":
                why = f"You scored {r['M']:.0%}, below the {PASS_MARK}% pass mark."
            elif r["M"] < READY:
                why = f"You passed with {r['M']:.0%}, which is still a weak base for what follows."
            else:
                why = (f"You scored {r['M']:.0%}. After {r['days_since']:.0f} days without revision, an estimated "
                       f"{r['K']:.0%} remains.")
        elif r["status"] == "Missed":
            action = "Catch up"
            why = f"Its assessment was due {-days} days ago and there is no score for it."
        else:
            action = "Practise" if r["clicks"] > 0 else "Learn"
            when = "today" if days == 0 else f"in {days} day{'s' if days != 1 else ''}"
            why = f"Due {when}. Your predicted chance of submitting and passing is {max(min(r['p_pass'], 0.99), 0.01):.0%}"
            why += ", so give it priority." if r["p_pass"] < 0.5 else "."
        if k in added:
            why += f" Added because topic {added[k]} builds on it."
        steps.append({"step": len(steps) + 1, "topic": int(k), "action": action, "why": why,
                      "resources": _steps(resources, act, k, t, bool(assessed[k]))})
    return steps, nothing_to_do
