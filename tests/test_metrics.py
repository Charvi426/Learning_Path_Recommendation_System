import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.model_metrics import all_metrics, calibration_table  # noqa: E402
from src.evaluation.ranking_metrics import ndcg_at_k, precision_at_k, recall_at_k, student_mean  # noqa: E402
from src.recommend.path import build_paths  # noqa: E402
from src.recommend.scoring import weight_grid  # noqa: E402


def test_ndcg_hand_computed_binary():
    rel = np.array([[1, 0, 1, 0, 0, 0]], float)
    path = np.array([[1, 2, 0, -1, -1]])
    # DCG@3 = 0/log2(2) + 1/log2(3) + 1/log2(4) = 0.63093 + 0.5 ; IDCG@3 = 1 + 0.63093
    expected = (1 / np.log2(3) + 0.5) / (1 + 1 / np.log2(3))
    assert ndcg_at_k(path, rel, 3)[0] == pytest.approx(expected)
    assert ndcg_at_k(path, rel, 3)[0] == pytest.approx(0.69343, abs=1e-5)
    assert ndcg_at_k(path, rel, 5)[0] == pytest.approx(expected)
    assert precision_at_k(path, rel, 3)[0] == pytest.approx(2 / 3)
    assert precision_at_k(path, rel, 5)[0] == pytest.approx(2 / 5)
    assert recall_at_k(path, rel, 3)[0] == pytest.approx(1.0)
    assert recall_at_k(path[:, :1], rel, 1)[0] == pytest.approx(0.0)


def test_ndcg_hand_computed_graded_and_empty():
    rel = np.array([[2, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], float)
    path = np.array([[1, 0, -1], [0, 1, 2]])
    # DCG = 1/1 + 2/log2(3); IDCG = 2/1 + 1/log2(3)
    assert ndcg_at_k(path, rel, 3)[0] == pytest.approx((1 + 2 / np.log2(3)) / (2 + 1 / np.log2(3)))
    assert np.isnan(ndcg_at_k(path, rel, 3)[1])        # no relevant topic -> undefined
    assert ndcg_at_k(np.array([[0, 1, -1]]), rel[:1], 3)[0] == pytest.approx(1.0)


def test_student_mean_averages_within_then_across():
    v = np.array([1.0, 0.0, 0.5, np.nan])
    s = np.array([0, 0, 1, 1])
    sm = student_mean(v, s)
    assert sm.tolist() == [0.5, 0.5]


def test_weight_grid_has_671_combinations():
    W = weight_grid(["G", "Q", "F", "D", "I"], 0.1, 0.2)
    assert len(W) == 671
    assert np.allclose(W.sum(1), 1) and (W[:, 4] <= 0.2 + 1e-9).all()
    assert len(weight_grid(["G", "Q", "D"], 0.1, 0.2)) == 66


def test_path_revision_cap_and_prerequisite_insertion():
    # topics 0..5; 0-3 revision candidates, 4-5 new; chain prerequisites k-1 -> k
    T = 6
    prereq = np.zeros((1, T, T), bool)
    for k in range(1, T):
        prereq[0, k, k - 1] = True
    is_rev = np.array([[1, 1, 1, 1, 0, 0]], bool)
    cand = np.ones((1, T), bool)
    score = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    p = build_paths(score, cand, is_rev, prereq, L=5, max_rev=3)
    assert p.tolist() == [[0, 1, 2, 4, 5]]                     # j3 dropped by the 3-revision cap
    # high-scoring new topic 4 whose prerequisite 3 is a selected revision item: 3 is inserted before 4
    score2 = np.array([[0.1, 0.8, 0.0, 0.6, 0.95, 0.4]])
    cand2 = np.array([[0, 1, 0, 1, 1, 1]], bool)
    p2 = build_paths(score2, cand2, is_rev, prereq, L=5, max_rev=3)
    assert p2.tolist() == [[3, 4, 1, 5, -1]]
    # a new topic never precedes an earlier new topic that is its prerequisite
    score3 = np.array([[0, 0, 0, 0, 0.3, 0.9]])
    cand3 = np.array([[0, 0, 0, 0, 1, 1]], bool)
    p3 = build_paths(score3, cand3, np.zeros((1, T), bool), prereq, L=5)
    assert p3[0, :2].tolist() == [4, 5]


def test_model_metrics_sanity():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = all_metrics(y, p)
    assert m["AUC"] == pytest.approx(0.75)
    assert m["Brier"] == pytest.approx(np.mean((p - y) ** 2))
    c = calibration_table(y, p, 10)
    assert c["n"].sum() == 4
