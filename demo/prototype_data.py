"""Course data, replayed OULAD histories and uploaded files for the prototype. Nothing a student enters is saved."""
from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

import prototype_engine as E

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "prototype"
MODEL = HERE / "prototype" / "model.json"
PRESENTATION = "2014J"


class PrototypeDataMissing(RuntimeError):
    pass


@lru_cache(maxsize=None)
def table(name: str) -> pd.DataFrame:
    p = DATA / f"{name}.parquet"
    if not p.exists():
        raise PrototypeDataMissing("The prototype's data is not built yet. Run: python demo/build_prototype.py")
    return pd.read_parquet(p)


@lru_cache(maxsize=None)
def models() -> dict:
    if not MODEL.exists():
        raise PrototypeDataMissing("The prototype's model is missing. Run: python demo/build_prototype.py")
    return json.loads(MODEL.read_text(encoding="utf-8"))["models"]


def modules() -> list[str]:
    found = sorted(table("topics")["module"].unique())
    return sorted(found, key=lambda m: (m not in ("DDD", "FFF"), m))


@lru_cache(maxsize=None)
def course(module: str) -> E.Course:
    t, p = table("topics"), table("parts")
    t = t[t["module"] == module].sort_values("topic")
    return E.Course(module, PRESENTATION, int(t["length"].iloc[0]),
                    t[["topic", "t_open", "t_due", "kind"]].reset_index(drop=True),
                    p[p["module"] == module][["id_assessment", "topic", "weight"]].reset_index(drop=True))


@lru_cache(maxsize=256)
def resources(module: str, day: int) -> pd.DataFrame:
    return E.map_resources(course(module), table(f"sites_{module}"), table(f"cohort_{module}"), day)


def student_ids(module: str) -> list[int]:
    return sorted(table(f"students_{module}")["id_student"].astype(int))


@lru_cache(maxsize=None)
def example_student(module: str) -> int:
    """A starting point for the ID box: the lowest id still registered with a score for every assessment."""
    s, r = table(f"students_{module}"), table(f"records_{module}")
    stayed = s[s["date_unregistration"].isna()]["id_student"]
    n = r[r["id_student"].isin(stayed)].dropna(subset=["score"]).groupby("id_student")["id_assessment"].nunique()
    return int(n[n == n.max()].index.min())


def replay(module: str, student: int, day: int) -> dict:
    """What was known about an OULAD student at the start of `day`. Later scores, clicks and withdrawal are hidden."""
    s = table(f"students_{module}")
    s = s[s["id_student"] == student]
    if s.empty:
        raise E.InputError(f"There is no student {student} in {module} {PRESENTATION}.")
    rec, act = table(f"records_{module}"), table(f"clicks_{module}")
    rec = rec[(rec["id_student"] == student) & (rec["day"] < day)]
    act = act[(act["id_student"] == student) & (act["day"] < day)]
    left = s["date_unregistration"].iloc[0]
    return {"records": rec.reset_index(drop=True),
            "results": E.topic_results(rec, course(module), day),
            "activity": act[["day", "clicks", "id_site"]].reset_index(drop=True),
            "withdrew": int(left) if pd.notna(left) and left < day else None}


def _read_csv(content: bytes, what: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(content), na_values=["?"], encoding="utf-8-sig")
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise E.InputError(f"The {what} file could not be read as a CSV file.") from e
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise E.InputError(f"The {what} file has no rows.")
    return df


def _for_this_course(df: pd.DataFrame, module: str, what: str) -> pd.DataFrame:
    for col, want in (("code_module", module), ("code_presentation", PRESENTATION)):
        if col in df:
            df = df[df[col].astype(str) == want]
            if df.empty:
                raise E.InputError(f"The {what} file has no rows for {module} {PRESENTATION}. Check the selected module.")
    return df


def _numeric(df: pd.DataFrame, cols: list[str], what: str) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if (v.isna() & df[c].notna()).any():
            raise E.InputError(f"Column '{c}' of the {what} file must contain numbers.")
        df[c] = v
    return df


def read_vle_file(content: bytes, module: str) -> pd.DataFrame:
    """OULAD studentVle columns (id_site, date, sum_click), or the simple form (topic, day, clicks)."""
    what = "VLE"
    d = _read_csv(content, what).rename(columns={"date": "day", "sum_click": "clicks"})
    if not {"day", "clicks"} <= set(d) or not {"id_site", "topic"} & set(d):
        raise E.InputError("The VLE file needs the columns id_site, date, sum_click (or topic, day, clicks).")
    d = _for_this_course(d, module, what)
    cols = [c for c in ("id_student", "day", "clicks", "topic", "id_site") if c in d]
    return _numeric(d[cols], cols, what).reset_index(drop=True)


def read_score_file(content: bytes, module: str) -> pd.DataFrame:
    """OULAD studentAssessment columns (id_assessment, date_submitted, score), or the simple form (topic, day, score)."""
    what = "score"
    d = _read_csv(content, what).rename(columns={"date_submitted": "day"})
    if not {"day", "score"} <= set(d) or not {"id_assessment", "topic"} & set(d):
        raise E.InputError("The score file needs the columns id_assessment, date_submitted, score (or topic, day, score).")
    d = _for_this_course(d, module, what)
    cols = [c for c in ("id_student", "day", "score", "topic", "id_assessment") if c in d]
    d = _numeric(d[cols], cols, what)
    if "id_assessment" in d:
        d = d[d["id_assessment"].isin(course(module).parts["id_assessment"])]
        if d.empty:
            raise E.InputError(f"None of the assessments in the score file belongs to {module} {PRESENTATION}.")
    return d.reset_index(drop=True)


def students_in(*frames) -> list[int]:
    ids: set[int] = set()
    for f in frames:
        if f is not None and "id_student" in f:
            ids |= set(f["id_student"].dropna().astype(int))
    return sorted(ids)


def uploaded_history(vle, scores, module: str, student: int | None, day: int) -> dict:
    def mine(f):
        return f[f["id_student"] == student] if student is not None and "id_student" in f else f

    results = pd.DataFrame(columns=["topic", "day", "score"], dtype=float)
    activity = pd.DataFrame(columns=["day", "clicks", "topic", "id_site"], dtype=float)
    unknown = 0
    if scores is not None:
        s = mine(scores).dropna(subset=["score"])
        if "id_assessment" in s:
            if s["score"].lt(0).any() or s["score"].gt(100).any():
                raise E.InputError("Scores must be between 0 and 100.")
            if s["day"].isna().any():
                raise E.InputError("Every score in the file needs a submission day.")
            results = E.topic_results(s, course(module), day)
        else:
            results = s[["topic", "day", "score"]]
    if vle is not None:
        activity = mine(vle).drop(columns="id_student", errors="ignore")
        if "id_site" in activity:
            unknown = int((~activity["id_site"].isin(table(f"sites_{module}")["id_site"])).sum())
            if unknown and unknown == len(activity):
                raise E.InputError(f"None of the resources in the VLE file belongs to {module} {PRESENTATION}.")
    return {"results": results, "activity": activity, "unknown_sites": unknown}
