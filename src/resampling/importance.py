"""Importance sampling: pesos y estimador."""
import numpy as np
from numpy.typing import NDArray


def importance_weights(
    target_log: NDArray[np.float64],
    proposal_log: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Pesos no normalizados: w_i = exp(log p_t(x_i) - log q(x_i))."""
    target_log = np.asarray(target_log, dtype=np.float64)
    proposal_log = np.asarray(proposal_log, dtype=np.float64)
    log_weights = target_log - proposal_log
    # restar el max evita overflow en exp
    log_weights -= np.max(log_weights)
    return np.exp(log_weights)


def normalize_weights(
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normaliza pesos a que sumen 1 (uniformes si la suma es 0)."""
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
    """Estimador IS: E_target[f(X)] ≈ sum_i W_i f(x_i)."""
    values = np.asarray(values, dtype=np.float64)
    normalized_weights = np.asarray(normalized_weights, dtype=np.float64)
    return float(np.sum(normalized_weights * values))