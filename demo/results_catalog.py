"""The single definition of every result shown in the demo and in the report.

A value is never typed here: each one is a Cell (file, row selector, column) read from a frozen results file or from
the read-only defect audit (report/audit/). demo/build_demo_data.py, report/make_tables.py and
report/check_numbers.py all use this module, so the demo and the report cannot disagree.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

T, V2, V3, AUD = "results/tables", "results/v2", "results/v3", "report/audit"


@dataclass(frozen=True)
class Cell:
    file: str
    col: str
    where: tuple = ()

    def describe(self) -> str:
        sel = ",".join(f"{k}={v}" for k, v in self.where)
        return f"{self.file}[{sel}].{self.col}"


def cell(file: str, col: str, **where) -> Cell:
    return Cell(file, col, tuple(where.items()))


@lru_cache(maxsize=None)
def _load(file: str):
    p = ROOT / file
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return pd.read_csv(p, float_precision="round_trip")


def read(c: Cell):
    d = _load(c.file)
    if isinstance(d, (dict, list)):
        for part in c.col.split("/"):
            d = d[int(part)] if isinstance(d, list) else d[part]
        return d
    for k, v in c.where:
        d = d[d[k] == v]
    if len(d) != 1:
        raise LookupError(f"{c.describe()} selects {len(d)} rows (expected exactly 1)")
    return d[c.col].iloc[0]


def fmt(value, style: str) -> str:
    """Styles: '3' = 3 decimals, '+3' = signed, 'pct0' = share shown as a percentage, 'int', 'p' = p-value."""
    if isinstance(value, float) and math.isnan(value):
        return "--"
    if style == "s":
        return str(value)
    if style == "int":
        return f"{round(float(value)):,}"
    if style == "p":
        return "<0.0001" if float(value) < 1e-4 else f"{float(value):.3f}"
    if style.startswith("pct"):
        return f"{100 * float(value):.{int(style[3:])}f}%"
    if style.startswith("abs"):
        return f"{abs(float(value)):.{int(style[3:])}f}"
    sign = "+" if style.startswith("+") else ""
    s = f"{float(value):{sign}.{int(style.lstrip('+'))}f}"
    return s.lstrip("+-") if float(s) == 0 else s


@dataclass(frozen=True)
class Val:
    """An estimate, optionally with its 95% CI."""
    est: Cell
    style: str = "3"
    lo: Cell | None = None
    hi: Cell | None = None

    def text(self) -> str:
        s = fmt(read(self.est), self.style)
        if self.lo is None:
            return s
        lo, hi = read(self.lo), read(self.hi)
        if pd.isna(lo):
            return s
        return f"{s} [{fmt(lo, self.style)}, {fmt(hi, self.style)}]"


@dataclass
class Table:
    key: str
    title: str
    columns: list
    rows: list
    stage: str        # planned / post-hoc / preregistered / audit
    evidence: str     # primary / exploratory / validation / audit
    validity: str     # why the numbers can or cannot be relied on
    wide: bool = False
    notes: str = ""
    sources: list = field(default_factory=list)


LEAK = "Invalid as performance: the label leaks into the ability feature (D1)."
RANK = "Indirectly affected by the leak (D1) through fitted coefficients and tuned settings; A0 and A1 fit no model."
SCALE = "v3 rows were scored with raw K instead of the preregistered K = M*R (D3). " + RANK
EXAM = "Unreliable: 'exam day' is the last day of the presentation, after the real exam (D2); coefficients carry D1."

DEFECTS = [
    ("D1", "Label leakage",
     "The ability feature averages every result submitted before the prediction time, which is the due date. For a "
     "student who submitted early, that includes the result being predicted.",
     "All model metrics in all three stages, and everything tuned or selected on them."),
    ("D2", "Wrong exam day",
     "The exam was placed on the last day of the presentation. Students sat it one to several weeks earlier.",
     "All exam checks and the pre-exam planner: both count activity from after the exam."),
    ("D3", "v3 not scored as preregistered",
     "The preregistration fixes K = M*R for the recommender and planner. The scoring script used raw K outside "
     "validation.",
     "All v3 ranking numbers on the holdout and 2014J, criterion (c), and the planner evaluations outside validation."),
    ("D4", "Cohort features look ahead",
     "Resource-to-topic maps, navigation classes and attribution weights use the scored cohort's own clicks from the "
     "whole course, in part from the revision period.",
     "All stages (topic map); v3 most (attribution)."),
]


def _ci(file, col, style="3", lo=None, hi=None, **where) -> Val:
    return Val(cell(file, col, **where), style, cell(file, lo or f"{col}_lo", **where),
               cell(file, hi or f"{col}_hi", **where))


def _nd(file, config, rel, ci=True, **extra) -> Val:
    w = dict(config=config, relevance=rel, metric="NDCG@5", **extra)
    return _ci(file, "value", lo="ci_lo", hi="ci_hi", **w) if ci else Val(cell(file, "value", **w))


def criteria() -> list[dict]:
    m, e, r = (f"{V3}/holdout/{x}.csv" for x in ("model_metrics_holdout", "exam_check_CCC", "recommender_holdout"))
    c = f"{V3}/preregistered_criteria.csv"
    names = list(_load(c)["criterion"])
    return [
        {"id": "a", "name": "Holdout within-course AUC at most 0.002 below v2",
         "v3": _ci(m, "auc_v3"), "v2": _ci(m, "auc_v2"), "effect": _ci(m, "diff", "+3"),
         "met": cell(c, "met", criterion=names[0]), "evidence": "primary",
         "reliable": "No: both models contain the label leak (D1)."},
        {"id": "b", "name": "CCC exam gap above v2's, CI of the improvement above 0",
         "v3": Val(cell(e, "v3_delta_rho"), "+3"), "v2": Val(cell(e, "v2_delta_rho_recomputed"), "+3"),
         "effect": _ci(e, "improvement", "+3", lo="ci_lo", hi="ci_hi"),
         "met": cell(c, "met", criterion=names[1]), "evidence": "exploratory",
         "reliable": "No: wrong exam day (D2), and a model with no forgetting (gap = 0) would also pass."},
        {"id": "c", "name": "Holdout NDCG@5 of v3 A5 at least v2 A5",
         "v3": _nd(r, "v3:A5", "AB"), "v2": _nd(r, "v2:A5", "AB"),
         "effect": Val(cell(f"{V3}/holdout/recommender_wilcoxon_holdout.csv", "mean_diff",
                            comparison="v3 A5 vs v2 A5", metric="NDCG@5[AB]"), "+3"),
         "met": cell(c, "met", criterion=names[2]), "evidence": "primary",
         "reliable": "No: scored with raw K, not the preregistered configuration (D3)."},
    ]


def tables() -> dict[str, Table]:
    out = []

    crit = criteria()
    out.append(Table(
        "criteria", "Preregistered v3 criteria, as recorded",
        ["Criterion", "v3", "v2", "v3 minus v2", "Recorded", "Evidence", "Can it be relied on?"],
        [[f"({c['id']}) {c['name']}", c["v3"], c["v2"], c["effect"], "MET" if bool(read(c["met"])) else "NOT MET",
          c["evidence"], c["reliable"]] for c in crit],
        "post-hoc, preregistered", "primary (a, c); exploratory (b)",
        "The recorded statuses are unchanged; none is a reliable test.", wide=True))

    f = f"{T}/model_metrics_test.csv"
    lab = {"majority": "Majority", "topic_difficulty": "Topic difficulty", "pfa": "PFA", "none": "No forgetting",
           "fixed": "Fixed decay", "learned": "Learned half-life", "learnedI": "Half-life + interest",
           "das3h": "DAS3H windows"}
    out.append(Table(
        "pred_v1", "v1 knowledge models, DDD/FFF 2014J test", ["Model", "AUC [95% CI]", "Log-loss", "Brier"],
        [[lab[m], _ci(f, "AUC", model=m), Val(cell(f, "logloss", model=m)), Val(cell(f, "Brier", model=m))] for m in lab],
        "planned", "primary for v1", LEAK))

    f = f"{V2}/holdout/model_metrics_holdout.csv"
    out.append(Table(
        "pred_v2", "v2 knowledge models, five holdout modules (pooled)", ["Model", "AUC [95% CI]", "Log-loss"],
        [[lab[m], _ci(f, "AUC", model=m), Val(cell(f, "logloss", model=m))] for m in lab if m != "majority"],
        "post-hoc", "primary for v2", LEAK))

    h, j = f"{V3}/holdout/model_metrics_holdout.csv", f"{V3}/posthoc_2014J/model_metrics_2014J.csv"
    out.append(Table(
        "pred_v3", "v3 against v2, within-course AUC", ["Set", "v2", "v3", "Difference [95% CI]"],
        [["Holdout (primary)", Val(cell(h, "auc_v2")), Val(cell(h, "auc_v3")), _ci(h, "diff", "+3")],
         ["DDD/FFF 2014J (exploratory)", Val(cell(j, "auc_v2")), Val(cell(j, "auc_v3")), _ci(j, "diff", "+3")]],
        "post-hoc, preregistered", "primary and exploratory", LEAK))

    f = f"{T}/dkt_comparison.csv"
    out.append(Table(
        "dkt", "Optional DKT baseline, test rows with history", ["Model", "AUC [95% CI]"],
        [[n, _ci(f, "AUC", split="test", model=m)]
         for m, n in (("dkt", "DKT (no ability feature)"), ("pfa", "PFA"), ("learned", "Learned half-life"))],
        "planned, optional", "exploratory",
        "Not like for like: DKT lacks the leaking feature that the other models have (D1)."))

    f = f"{T}/ranking_metrics_test.csv"
    names = {"A0": "Random", "A1": "Popularity", "A2": "Mastery", "A3": "+ prerequisites", "A4": "+ fixed decay",
             "A5": "+ learned decay", "A6": "+ interest", "A7": "+ data edges"}
    out.append(Table(
        "rank_v1", "v1 ablation, DDD/FFF 2014J test, NDCG@5", ["Configuration", "Combined [95% CI]", "Rel-A", "Rel-B"],
        [[f"{a} {n}", _nd(f, a, "AB"), _nd(f, a, "A", False), _nd(f, a, "B", False)] for a, n in names.items()],
        "planned", "primary for v1", RANK))

    f2, f3 = f"{V2}/holdout/ranking_metrics_holdout.csv", f"{V3}/holdout/recommender_holdout.csv"
    rows = [[f"v1 {a}", _nd(f2, f"v1:{a}", "AB", module="pooled"), _nd(f2, f"v1:{a}", "A", False, module="pooled"),
             _nd(f2, f"v1:{a}", "B", False, module="pooled")] for a in ("A3", "A5", "A6")]
    rows += [[f"v2 {a}", _nd(f2, f"v2:{a}", "AB", module="pooled"), _nd(f2, f"v2:{a}", "A", False, module="pooled"),
              _nd(f2, f"v2:{a}", "B", False, module="pooled")] for a in ("A1", "A3", "A5", "A6", "A8", "A9")]
    rows += [[f"v3 {a}" + ("" if a == "A3" else " (raw K)"), _nd(f3, f"v3:{a}", "AB"), _nd(f3, f"v3:{a}", "A", False),
              _nd(f3, f"v3:{a}", "B", False)] for a in ("A3", "A5", "A6", "A8", "A9")]
    out.append(Table("rank_holdout", "Holdout NDCG@5 across stages",
                     ["Configuration", "Combined [95% CI]", "Rel-A", "Rel-B"], rows, "post-hoc", "primary", SCALE))

    f2, f3 = f"{V2}/posthoc_2014J/ranking_metrics.csv", f"{V3}/posthoc_2014J/recommender_2014J.csv"
    rows = [[f"v2 {a}", _nd(f2, f"v2:{a}", "AB"), _nd(f2, f"v2:{a}", "A", False), _nd(f2, f"v2:{a}", "B", False)]
            for a in ("A1", "A3", "A5", "A6", "A8", "A9")]
    rows += [[f"v3 {a} (raw K)", _nd(f3, f"v3:{a}", "AB"), _nd(f3, f"v3:{a}", "A", False), _nd(f3, f"v3:{a}", "B", False)]
             for a in ("A5", "A6", "A8", "A9")]
    out.append(Table("rank_2014j", "v2 and v3 on DDD/FFF 2014J, NDCG@5 (exploratory)",
                     ["Configuration", "Combined [95% CI]", "Rel-A", "Rel-B"], rows, "post-hoc", "exploratory", SCALE))

    w1, w2, w3 = (f"{T}/wilcoxon_holm.csv", f"{V2}/holdout/wilcoxon_holm_holdout.csv",
                  f"{V3}/holdout/recommender_wilcoxon_holdout.csv")
    tests = [("v1 test: A5 vs A3", w1, "forgetting: A5 vs A3"),
             ("v1 test: A5 vs A4", w1, "learned vs fixed decay: A5 vs A4"),
             ("v1 test: A6 vs A5", w1, "interest: A6 vs A5"),
             ("v2 holdout: A5 vs A3", w2, "forgetting: A5 vs A3 (v2)"),
             ("v2 holdout: A8 vs A1", w2, "popularity term: A8 vs A1 (v2)"),
             ("v2 holdout: A9 vs A8", w2, "interest: A9 vs A8 (v2)"),
             ("v3 holdout: A5 vs v2 A5", w3, "v3 A5 vs v2 A5"),
             ("v3 holdout: A5 vs A3", w3, "v3 A5 vs v3 A3"),
             ("v3 holdout: A9 vs A8", w3, "v3 A9 vs v3 A8")]
    out.append(Table(
        "tests", "Paired Wilcoxon tests on NDCG@5, Holm-corrected", ["Comparison", "Mean difference", "Holm p"],
        [[n, Val(cell(f, "mean_diff", comparison=c, metric="NDCG@5[AB]"), "+3"),
          Val(cell(f, "p_holm", comparison=c, metric="NDCG@5[AB]"), "p")] for n, f, c in tests],
        "planned (v1), post-hoc (v2, v3)", "v1 test; v2 and v3 holdout", SCALE))

    t, o = f"{T}/exam_retention_check.csv", f"{V2}/holdout/exam_retention_check_holdout.csv"
    d = "diff_spearman_K_minus_M"
    rows = [["v1 fixed, DDD 2014J (planned)", _ci(t, d, "+3", "diff_lo", "diff_hi", variant="fixed"), "--"],
            ["v1 learned, DDD 2014J (planned)", _ci(t, d, "+3", "diff_lo", "diff_hi", variant="learned"), "--"],
            ["v2 fixed, CCC (exploratory)", _ci(o, d, "+3", "diff_lo", "diff_hi", variant="fixed"), "--"],
            ["v2 learned, CCC (exploratory)", _ci(o, d, "+3", "diff_lo", "diff_hi", variant="learned"), "--"]]
    for n, f in (("v3, DDD 2014B (validation)", f"{V3}/planner/exam_check_val.csv"),
                 ("v3, CCC (exploratory)", f"{V3}/holdout/exam_check_CCC.csv"),
                 ("v3, DDD 2014J (exploratory)", f"{V3}/posthoc_2014J/exam_check_DDD_2014J.csv")):
        rows.append([n, Val(cell(f, "v3_delta_rho"), "+3"), _ci(f, "improvement", "+3", lo="ci_lo", hi="ci_hi")])
    out.append(Table("exam", "Exam check: rank correlation of K minus that of M",
                     ["Model and set", "Gap [95% CI]", "Improvement on v2"], rows,
                     "planned (v1), post-hoc (v2, v3)", "v1 planned; v2 and v3 exploratory", EXAM))

    f = f"{V3}/variant_comparison_val.csv"
    vn = {"V0": "V0 v2 half-life", "V1": "V1 half-life, attributed", "V2": "V2 ACT-R, fixed decay",
          "V3": "V3 ACT-R, spacing (selected)", "V4": "V4 DAS3H, attributed"}
    out.append(Table(
        "variants", "v3 memory variants on validation", ["Variant", "Log-loss", "AUC", "Exam gap"],
        [[n, Val(cell(f, "val_logloss", variant=v), "4"), Val(cell(f, "val_within_course_auc", variant=v)),
          Val(cell(f, "exam_delta_rho", variant=v), "+3")] for v, n in vn.items()],
        "post-hoc", "validation", "Selection used leaking losses (D1) and the wrong exam day (D2)."))

    rows = []
    for n, f in (("DDD 2014B (validation, M*R)", f"{V3}/planner/observational_check_val.csv"),
                 ("CCC (exploratory, raw K)", f"{V3}/holdout/observational_check_CCC.csv"),
                 ("DDD 2014J (exploratory, raw K)", f"{V3}/posthoc_2014J/observational_check_DDD_2014J.csv")):
        rows.append([n, Val(cell(f, "n_students", term="share_revision_on_flagged"), "int"),
                     _ci(f, "coef", "1", "ci_lo", "ci_hi", term="share_revision_on_flagged")])
    out.append(Table("planner", "Planner: observational association, not causal",
                     ["Set", "n", "Exam points per unit share [95% CI]"], rows, "post-hoc", "exploratory",
                     "Not causal. Much of the 'pre-exam' window is after the exam (D2); raw K outside validation (D3)."))

    f = f"{T}/coefficients.csv"
    out.append(Table(
        "interest", "v1 interest coefficients", ["Coefficient", "Estimate [95% CI]"],
        [[n, _ci(f, "estimate", "+3", "ci_lo", "ci_hi", model="learnedI", param=p)]
         for p, n in (("phi_I", "Interest on half-life"), ("theta_I", "Interest on passing"),
                      ("theta_Ic", "Interest x clicks"))],
        "planned", "training fit", "Fitted with the leak (D1); the interest feature also sees post-submission clicks."))

    le, fi = f"{AUD}/label_in_ability.csv", f"{AUD}/first_topic_ability_auc.csv"
    out.append(Table(
        "leak", "D1: label inside the ability feature, 2014J test rows",
        ["Module", "Rows affected", "Topic 1: affected", "Topic 1: AUC of the feature alone"],
        [[m, Val(cell(le, "share_label_in_feature", module=m, split="test"), "pct0"),
          Val(cell(fi, "share_label_in_feature", module=m, split="test"), "pct0"),
          Val(cell(fi, "auc_of_stored_ability_alone", module=m, split="test"))]
         for m in ("DDD", "FFF", "AAA", "BBB", "CCC", "EEE", "GGG")],
        "audit", "audit", "Read-only count on stored data. Topic 1 has no history, so a clean feature scores 0.5."))

    e = f"{AUD}/exam_day.csv"
    out.append(Table(
        "examday", "D2: exam day used against the day it was sat",
        ["Presentation", "Day used", "Median day sat", "Planner weeks after exam"],
        [[f"{m} {p}", Val(cell(e, "day_used", module=m, presentation=p), "int"),
          Val(cell(e, "median_day_sat", module=m, presentation=p), "int"),
          Val(cell(e, "planner_points_after_median_exam", module=m, presentation=p), "int")]
         for m, p in (("DDD", "2014B"), ("DDD", "2014J"), ("CCC", "2014B"), ("CCC", "2014J"))],
        "audit", "audit", "Read-only comparison with the recorded sitting day."))

    for t_ in out:
        t_.sources = sorted({c.file for r in t_.rows for v in r if isinstance(v, Val) for c in (v.est,)})
    return {t_.key: t_ for t_ in out}


def frame(table: Table) -> pd.DataFrame:
    return pd.DataFrame([[v.text() if isinstance(v, Val) else v for v in r] for r in table.rows], columns=table.columns)
