"""Path construction (METHODOLOGY 6.4), vectorised over rows (decision point x weight vector).

1. rank candidates by score (ties -> lower topic index first);
2. take the top L with at most `max_rev` revision items;
3./4. order the selected items so that every new topic comes after its in-path prerequisites
   (revision prerequisites are inserted before it, an earlier new prerequisite is never placed after it);
   otherwise items keep score order (greedy topological sort by score).
5. labels: Revise / Practice (new, already clicked) / Learn (new, not opened).
"""
from __future__ import annotations

import numpy as np


def select_items(score, cand, is_rev, L=5, max_rev=3):
    """score/cand/is_rev (R, T) -> selected mask (R, T)."""
    R, T = score.shape
    s = np.where(cand, score, -np.inf)
    # stable descending order with ties by lower topic index
    order = np.argsort(-s, axis=1, kind="stable")
    c_sorted = np.take_along_axis(cand, order, 1)
    r_sorted = np.take_along_axis(is_rev, order, 1) & c_sorted
    rev_rank = np.cumsum(r_sorted, axis=1)
    keep = c_sorted & (~r_sorted | (rev_rank <= max_rev))
    keep &= np.cumsum(keep, axis=1) <= L
    sel = np.zeros_like(cand)
    np.put_along_axis(sel, order, keep, 1)
    return sel


def order_path(score, sel, is_rev, prereq, L=5):
    """Score order with prerequisite insertion. At each step take the highest-scored unplaced item; if it is a
    new topic with unplaced in-path prerequisites, descend to its earliest such prerequisite (recursively) and
    place that first. prereq (R, T, T) bool or None. Returns path (R, L) of topic idx (-1 = empty)."""
    R, T = score.shape
    rows = np.arange(R)
    placed = np.zeros((R, T), bool)
    path = np.full((R, L), -1, dtype=np.int64)
    s = np.where(sel, score, -np.inf)
    tie = -1e-9 * np.arange(T)[None, :]          # deterministic tie-break: lower topic index first
    for step in range(L):
        val = np.where(sel & ~placed, s + tie, -np.inf)
        x = val.argmax(1)
        ok = np.isfinite(val[rows, x])
        if prereq is not None:
            pending = sel & ~placed
            for _ in range(T):
                need = prereq[rows, x, :] & pending & ~is_rev[rows, x][:, None]
                has = need.any(1) & ok
                if not has.any():
                    break
                x = np.where(has, need.argmax(1), x)   # earliest unplaced prerequisite
        path[ok, step] = x[ok]
        placed[rows[ok], x[ok]] = True
    return path


def build_paths(score, cand, is_rev, prereq=None, L=5, max_rev=3):
    sel = select_items(score, cand, is_rev, L, max_rev)
    return order_path(score, sel, is_rev, prereq, L)


def labels(path_row, is_rev_row, opened_row):
    out = []
    for j in path_row:
        if j < 0:
            continue
        out.append("Revise" if is_rev_row[j] else ("Practice" if opened_row[j] else "Learn"))
    return out
