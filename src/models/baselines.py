"""Baselines (METHODOLOGY 5.4): majority and topic difficulty (IRT-like: beta_k + theta_a * ability)."""
from .das3h import Majority, SklearnLogit


def topic_difficulty_model(C: float = 1.0) -> SklearnLogit:
    return SklearnLogit("irt", C=C)


__all__ = ["Majority", "topic_difficulty_model"]
