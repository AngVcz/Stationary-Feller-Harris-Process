"""Importance sampling estimator and weight computation."""
import numpy as np
from numpy.typing import NDArray


def importance_weights(
    target_log: NDArray[np.float64],
    proposal_log: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute unnormalized importance weights from log-densities.

    w_i = exp(log_target(x_i) - log_proposal(x_i))

    Args:
        target_log: Log-density of the target distribution at each sample (N,).
        proposal_log: Log-density of the proposal distribution at each sample (N,).

    Returns:
        Unnormalized importance weights (N,).
    """
    target_log = np.asarray(target_log, dtype=np.float64)
    proposal_log = np.asarray(proposal_log, dtype=np.float64)
    log_weights = target_log - proposal_log
    # Subtract max for numerical stability before exponentiating
    log_weights -= np.max(log_weights)
    return np.exp(log_weights)


def normalize_weights(
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize importance weights to sum to 1.

    Args:
        weights: Unnormalized importance weights (N,).

    Returns:
        Normalized weights summing to 1 (N,).
    """
    weights = np.asarray(weights, dtype=np.float64)
    total = np.sum(weights)
    if total == 0.0:
        n = len(weights)
        return np.ones(n) / n
    return weights / total


def is_estimate(
    values: NDArray[np.float64],
    normalized_weights: NDArray[np.float64],
) -> float:
    """Compute importance sampling estimate of E_target[f(X)].

    E_target[f(X)] ≈ sum_i W_i * f(x_i)

    Args:
        values: Function evaluations f(x_i) at each sample (N,).
        normalized_weights: Normalized importance weights W_i (N,).

    Returns:
        Importance sampling estimate.
    """
    values = np.asarray(values, dtype=np.float64)
    normalized_weights = np.asarray(normalized_weights, dtype=np.float64)
    return float(np.sum(normalized_weights * values))