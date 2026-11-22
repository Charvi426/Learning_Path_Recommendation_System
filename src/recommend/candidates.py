"""Candidate generation (METHODOLOGY 6.1).

C_new = {k not yet assessed : M_k < tau_m, Q_k >= tau_q, k <= k_now + horizon}
C_rev = {j assessed : K_j(t) < tau_r}
"""
from __future__ import annotations

import numpy as np


def candidates(M, K, Qn, assessed, topic_no, k_now, tau_m, tau_q, tau_r, horizon=2, use_q=True):
    within = topic_no[None, :] <= (k_now[:, None] + horizon)
    new = (~assessed) & (M < tau_m) & within
    if use_q:
        new &= Qn >= tau_q
    rev = assessed & (K < tau_r)
    return new, rev
