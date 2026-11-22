"""All data functions of the demo. They read demo/data/ only: no model, no raw data, no results files."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
STATUS_ORDER = ["Learned", "Fading", "Not started"]


class DemoDataMissing(RuntimeError):
    pass


@lru_cache(maxsize=None)
def table(name: str) -> pd.DataFrame:
    p = DATA / f"{name}.parquet"
    if not p.exists():
        raise DemoDataMissing("Demo data not found. Build it with:  python demo/build_demo_data.py")
    return pd.read_parquet(p)


@lru_cache(maxsize=None)
def _by_student(name: str) -> dict:
    return dict(tuple(table(name).groupby("student", sort=False)))


def _rows(name: str, student: str) -> pd.DataFrame:
    got = _by_student(name).get(student)
    return table(name).iloc[0:0] if got is None else got


def default_student() -> str:
    return str(table("meta")["default_student"].iloc[0])


def modules() -> list[str]:
    found = list(table("students")["module"].unique())
    return [m for m in ("DDD", "FFF") if m in found] + sorted(set(found) - {"DDD", "FFF"})


def students(module: str) -> list[str]:
    s = table("students")
    return sorted(s.loc[s["module"] == module, "student"])


def student(student_id: str) -> dict:
    r = _rows("students", student_id)
    if r.empty:
        raise KeyError(student_id)
    return r.iloc[0].to_dict()


def notices(info: dict) -> list[str]:
    """Plain messages for the cases in which part of a page cannot be shown."""
    out = []
    if info["n_clicks"] == 0:
        out.append("This student has no clicks on course resources, so there is no practice history to show.")
    if info["withdrawn"]:
        out.append(f"This student withdrew on day {int(info['withdrawn_day'])}. Topics due after that day are not used, "
                   "and the charts stop there.")
    if decisions(info["student"]).empty:
        out.append("There is no decision point for this student: none occurs until an assessment has been submitted "
                   "while a later one is still open. No path is recommended.")
    return out


def decisions(student_id: str) -> pd.DataFrame:
    return _rows("decisions", student_id).sort_values(["day", "next_topic"]).reset_index(drop=True)


def decision_label(row) -> str:
    return f"Week {row['week']} (day {row['day']}), before topic {row['next_topic']}"


def knowledge_state(student_id: str, model: str, decision: int) -> pd.DataFrame:
    s = _rows("states", student_id)
    return s[(s["model"] == model) & (s["decision"] == decision)].sort_values("topic").reset_index(drop=True)


def path(student_id: str, model: str, decision: int) -> pd.DataFrame:
    p = _rows("paths", student_id)
    return p[(p["model"] == model) & (p["decision"] == decision)].sort_values("position").reset_index(drop=True)


def what_changed(student_id: str, decision: int) -> tuple[pd.DataFrame, str]:
    old, new = path(student_id, "v2", decision), path(student_id, "v3", decision)
    cols = ["position", "topic", "action"]
    m = old[cols].merge(new[cols], on="topic", how="outer", suffixes=("_v2", "_v3"))

    def change(r):
        if pd.isna(r["position_v2"]):
            return "Added in v3"
        if pd.isna(r["position_v3"]):
            return "Dropped in v3"
        d = int(r["position_v2"] - r["position_v3"])
        return "Same position" if d == 0 else f"{'Up' if d > 0 else 'Down'} {abs(d)}"

    m["change"] = [change(r) for _, r in m.iterrows()]
    m = m.sort_values(["position_v3", "position_v2"], na_position="last").reset_index(drop=True)
    n = int((m["change"] != "Same position").sum())
    text = "Both models recommend the same path." if n == 0 else f"{n} of {len(m)} topics differ between v2 and v3."
    return m, text


def curve(student_id: str, model: str) -> pd.DataFrame:
    c = table(f"curves_{student_id[:3]}")
    c = c[c["student"] == student_id]
    return c[["topic", "day", model]].rename(columns={model: "K"}).reset_index(drop=True)


def events(student_id: str, model: str) -> pd.DataFrame:
    e = table(f"events_{student_id[:3]}")
    return e[(e["student"] == student_id) & e["model"].isin([model, "both"])].reset_index(drop=True)


def plan(student_id: str) -> tuple[pd.DataFrame, str | None]:
    """(top-3 rows for each of the four weeks, or an explanation of why there is no plan)."""
    info = student(student_id)
    none = table("plans").iloc[0:0]
    if not info["module_has_exam"]:
        return none, (f"{info['module']} has no exam results in OULAD, so there is no pre-exam plan. "
                      "Choose CCC or DDD to see the planner.")
    if not info["sat_exam"]:
        why = "withdrew from the module" if info["withdrawn"] else "has no exam result"
        return none, f"This student {why}. Plans exist only for students who sat the exam."
    p = _rows("plans", student_id)
    p = p[p["flagged"]]
    if p.empty:
        return none, "No assessed topic is expected to lose knowledge before the exam, so nothing is flagged."
    return p.sort_values(["weeks_before", "rank"], ascending=[False, True]).reset_index(drop=True), None


def result_tables() -> list[dict]:
    cells = table("result_cells")
    out = []
    for r in table("result_index").sort_values("order").to_dict("records"):
        c = cells[cells["key"] == r["key"]]
        cols = list(c.sort_values("col_no")["col"].unique())
        out.append({**r, "table": c.pivot(index="row", columns="col", values="text")[cols].reset_index(drop=True)})
    return out
