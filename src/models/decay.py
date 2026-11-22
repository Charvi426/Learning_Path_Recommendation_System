"""Retention curves (METHODOLOGY 5.1, 5.2)."""
from __future__ import annotations

import numpy as np

LN2 = np.log(2.0)


def retention_none(delta):
    return np.ones_like(np.asarray(delta, float))


def retention_fixed(delta, S):
    """report eq. 3: exp(-delta / S)."""
    return np.exp(-np.asarray(delta, float) / S)


def log2_half_life(phi, Z, h_min=1.0, h_max=365.0):
    """log2 h = phi0 + Z @ phi[1:], clipped to [log2 h_min, log2 h_max]; returns (g, inside_mask)."""
    g = phi[0] + (Z @ phi[1:] if Z.shape[1] else 0.0)
    lo, hi = np.log2(h_min), np.log2(h_max)
    inside = (g > lo) & (g < hi)
    return np.clip(g, lo, hi), inside


def retention_hlr(delta, g):
    """2^(-delta / h) with h = 2^g."""
    return np.exp(-LN2 * np.asarray(delta, float) * np.exp2(-g))
