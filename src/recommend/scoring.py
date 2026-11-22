"""Score terms G, Q, F, D, I and the hybrid score (METHODOLOGY 6.2). All terms lie in [0, 1].

Arrays are (N decision points, T topics). `prereq` is (N, T, T) bool: prereq[n, k, j] = j is a prerequisite of k.
"""
from __future__ import annotations

import itertools

import numpy as np

TERMS = ["G", "Q", "F", "D", "I"]


def readiness(K: np.ndarray, prereq: np.ndarray) -> np.ndarray:
    """Q for a new topic k = min over its prerequisites of K_{s,j}(t); 1 if it has none."""
    big = np.where(prereq, K[:, None, :], np.inf)
    q = big.min(axis=2)
    return np.where(np.isfinite(q), q, 1.0)


def term_matrix(M, K, assessed, I, prereq, p_star) -> tuple[np.ndarray, np.ndarray]:
    """returns terms (N, T, 5) and Q_new (N, T)."""
    Qn = readiness(K, prereq)
    G = 1.0 - M
    Q = np.where(assessed, 1.0, Qn)
    F = np.where(assessed, np.clip(M - K, 0, 1), 0.0)
    D = 1.0 - np.abs(K - p_star) / max(p_star, 1 - p_star)
    terms = np.stack([G, Q, F, D, I], axis=-1)
    return np.clip(terms, 0.0, 1.0), Qn


def weight_grid(active: list[str], step: float = 0.1, eps_max: float = 0.2) -> np.ndarray:
    """All weight vectors over the active terms in steps of `step`, summing to 1, epsilon (I) <= eps_max.
    With all five terms active this gives the 671 combinations of Methodology 6.3."""
    n = int(round(1 / step))
    idx = [TERMS.index(a) for a in active]
    out = []
    for comb in itertools.product(range(n + 1), repeat=len(active)):
        if sum(comb) != n:
            continue
        w = np.zeros(5)
        w[idx] = np.array(comb) * step
        if w[4] > eps_max + 1e-9:
            continue
        out.append(w)
    return np.array(out)


def scores(terms: np.ndarray, W: np.ndarray) -> np.ndarray:
    """terms (N, T, 5), W (n_w, 5) -> (n_w, N, T)."""
    return np.einsum("ntc,wc->wnt", terms, W, optimize=True)
