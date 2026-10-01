"""Effective Sample Size (ESS)."""
import numpy as np
from numpy.typing import NDArray


def effective_sample_size(weights: NDArray[np.float64]) -> float:
    """ESS = (sum w_i)^2 / sum(w_i^2); con pesos normalizados, 1 / sum(W_i^2).

    Igual a N con pesos uniformes, tiende a 1 si un peso domina.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) == 0:
        return 0.0
    total = np.sum(weights)
    if total == 0.0:
        return 0.0
    return total**2 / np.sum(weights**2)