"""Effective Sample Size computation."""
import numpy as np
from numpy.typing import NDArray


def effective_sample_size(weights: NDArray[np.float64]) -> float:
    """Compute the Effective Sample Size (ESS) from importance weights.

    ESS = (sum w_i)^2 / sum(w_i^2)

    For normalized weights W_i: ESS = 1 / sum(W_i^2)

    Args:
        weights: Unnormalized importance weights (N,).

    Returns:
        ESS value. Equal to N when all weights are equal,
        approaches 1 when a single weight dominates.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) == 0:
        return 0.0
    total = np.sum(weights)
    if total == 0.0:
        return 0.0
    return total**2 / np.sum(weights**2)