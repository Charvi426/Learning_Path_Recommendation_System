"""Template explanations (METHODOLOGY 6.5): the item's largest weighted term, plus a second reason if the
second-largest weighted term contributes more than 25% of the item's score."""
from __future__ import annotations

import numpy as np

from .scoring import TERMS


def _pct(x):
    return f"{100 * x:.0f}%"


def explain_item(name, term_vals, weights, info) -> str:
    """term_vals, weights: arrays of 5 (G, Q, F, D, I). info: dict with M, R, K, delta, p_hat, prereq_name,
    prereq_K, is_rev."""
    contrib = np.asarray(weights) * np.asarray(term_vals)
    S = contrib.sum()
    order = np.argsort(-contrib)
    reasons = [_template(TERMS[order[0]], name, info)]
    if S > 0 and contrib[order[1]] > 0.25 * S:
        reasons.append(_template(TERMS[order[1]], name, info))
    return " ".join(reasons)


def _template(term, name, info) -> str:
    if term == "F":
        return (f"Revise {name}: you scored {_pct(info['M'])}, but estimated retention has dropped to "
                f"{_pct(info['R'])} after {info['delta']:.0f} days without practice.")
    if term == "G":
        if info.get("is_rev"):
            return f"Revise {name}: your mastery is still low ({_pct(info['M'])} on its assessment)."
        return f"Learn {name}: your predicted mastery is still low ({_pct(info['M'])})."
    if term == "Q":
        if info.get("prereq_name") is None:
            return f"{name} is ready: it has no prerequisites you still need."
        return f"{name} is ready: its prerequisite {info['prereq_name']} is still {_pct(info['prereq_K'])} retained."
    if term == "D":
        return f"{name} is at the right level for you now (predicted success {_pct(info['p_hat'])})."
    if term == "I":
        return f"You've been exploring {name} beyond the required material."
    return ""
