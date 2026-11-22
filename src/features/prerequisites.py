"""Prerequisite graph (METHODOLOGY 4.5): chain k-1 -> k plus optional data-derived edges.

Topics are aligned across presentations of a module (DECISIONS.md M1-4), so edges are per module.
"""
from __future__ import annotations

import pandas as pd


def chain_edges(topics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, p), g in topics.groupby(["code_module", "code_presentation"]):
        for k in sorted(g["topic"])[1:]:
            rows.append({"code_module": m, "code_presentation": p, "from_topic": k - 1, "to_topic": k,
                         "source": "chain"})
    return pd.DataFrame(rows)


def data_edges(train_inter: pd.DataFrame, topics: pd.DataFrame, min_diff: float, min_group: int):
    """u -> v (u < v) when P(y_v=1 | y_u=1) - P(y_v=1 | y_u=0) > min_diff with >= min_group students per
    group, on the training split (submitted rows only). Returns (edges, full statistics table)."""
    d = train_inter[train_inter["is_submitted"] == 1]
    stats = []
    for m, g in d.groupby("code_module"):
        wide = g.pivot_table(index="sid", columns="topic", values="y")
        ks = sorted(wide.columns)
        for u in ks:
            for v in ks:
                if v <= u:
                    continue
                pair = wide[[u, v]].dropna()
                g1, g0 = pair[pair[u] == 1], pair[pair[u] == 0]
                p1 = g1[v].mean() if len(g1) else float("nan")
                p0 = g0[v].mean() if len(g0) else float("nan")
                stats.append({"code_module": m, "from_topic": u, "to_topic": v, "n_pass_u": len(g1),
                              "n_fail_u": len(g0), "p_v_given_pass_u": p1, "p_v_given_fail_u": p0,
                              "diff": p1 - p0})
    st = pd.DataFrame(stats)
    st["selected"] = (st["diff"] > min_diff) & (st["n_pass_u"] >= min_group) & (st["n_fail_u"] >= min_group)
    sel = st[st["selected"]]
    pres = topics[["code_module", "code_presentation"]].drop_duplicates()
    e = sel[["code_module", "from_topic", "to_topic"]].merge(pres, on="code_module")
    e["source"] = "data"
    return e, st


def combine(chain: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """chain + data edges; an edge already in the chain keeps source 'chain'."""
    all_e = pd.concat([chain, data], ignore_index=True)
    return all_e.drop_duplicates(["code_module", "code_presentation", "from_topic", "to_topic"], keep="first")


def prereq_dict(edges: pd.DataFrame) -> dict:
    """{module: {k: [prereqs]}} (topics are aligned across presentations)."""
    out: dict = {}
    for (m, v), g in edges.drop_duplicates(["code_module", "from_topic", "to_topic"]).groupby(["code_module", "to_topic"]):
        out.setdefault(m, {})[int(v)] = sorted(int(u) for u in g["from_topic"])
    return out
