"""PFA baseline (METHODOLOGY 5.4): topic intercept + ability + prerequisite success/failure counts + clicks,
no time information."""
from .das3h import SklearnLogit


def pfa_model(C: float = 1.0) -> SklearnLogit:
    return SklearnLogit("pfa", C=C)
