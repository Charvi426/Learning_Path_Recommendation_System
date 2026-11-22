"""Read the raw OULAD CSVs with explicit dtypes. studentVle is read in chunks."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

KEY = ["code_module", "code_presentation"]


def _read(raw: Path, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(raw / f"{name}.csv", na_values=["?", ""], keep_default_na=True, **kw)


def load_courses(raw: Path) -> pd.DataFrame:
    return _read(raw, "courses", dtype={"code_module": str, "code_presentation": str,
                                         "module_presentation_length": "int32"})


def load_assessments(raw: Path) -> pd.DataFrame:
    return _read(raw, "assessments", dtype={"code_module": str, "code_presentation": str,
                                             "id_assessment": "int64", "assessment_type": str,
                                             "date": "float64", "weight": "float64"})


def load_student_assessment(raw: Path) -> pd.DataFrame:
    return _read(raw, "studentAssessment", dtype={"id_assessment": "int64", "id_student": "int64",
                                                   "date_submitted": "int32", "is_banked": "int8",
                                                   "score": "float64"})


def load_vle(raw: Path) -> pd.DataFrame:
    return _read(raw, "vle", dtype={"id_site": "int64", "code_module": str, "code_presentation": str,
                                     "activity_type": str, "week_from": "float64", "week_to": "float64"})


def load_registration(raw: Path) -> pd.DataFrame:
    return _read(raw, "studentRegistration", dtype={"code_module": str, "code_presentation": str,
                                                     "id_student": "int64", "date_registration": "float64",
                                                     "date_unregistration": "float64"})


def load_info(raw: Path) -> pd.DataFrame:
    return _read(raw, "studentInfo", dtype={"code_module": str, "code_presentation": str,
                                             "id_student": "int64", "num_of_prev_attempts": "int16",
                                             "studied_credits": "int32"})


def load_student_vle(raw: Path, modules: list[str] | None = None, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Read studentVle in chunks, keep the selected modules and aggregate duplicate
    (student, site, day) rows by summing clicks."""
    dtype = {"code_module": str, "code_presentation": str, "id_student": "int32",
             "id_site": "int32", "date": "int16", "sum_click": "int32"}
    parts = []
    for ch in pd.read_csv(raw / "studentVle.csv", dtype=dtype, chunksize=chunksize):
        if modules is not None:
            ch = ch[ch["code_module"].isin(modules)]
        if len(ch):
            parts.append(ch.groupby(KEY + ["id_student", "id_site", "date"], as_index=False)["sum_click"].sum())
    df = pd.concat(parts, ignore_index=True)
    # chunks can split a (student, site, day) group; aggregate again
    df = df.groupby(KEY + ["id_student", "id_site", "date"], as_index=False)["sum_click"].sum()
    for c in KEY:
        df[c] = df[c].astype("category")
    return df
