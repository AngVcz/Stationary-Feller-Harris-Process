"""Probability kernel computations for state-space models."""
import numpy as np
from numpy.typing import NDArray


def gaussian_kernel(
    x_prev: float,
    sigma: float,
    x_curr: float,
    phi: float = 0.0,
) -> float:
    """Evaluate Gaussian transition kernel K(x_prev, x_curr).

    For AR(1): K(x_prev, x_curr) = N(x_curr; phi * x_prev, sigma^2).
    For standard Gaussian: set phi=0.

    Args:
        x_prev: Previous state.
        sigma: Standard deviation of the transition noise.
        x_curr: Current state.
        phi: AR(1) coefficient. Set to 0 for shift-invariant kernel.

    Returns:
        Kernel value K(x_prev, x_curr).
    """
    mean = phi * x_prev
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_curr - mean) / sigma) ** 2
    )


def normalize_kernel(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a set of importance weights to sum to 1.

    Args:
        weights: Unnormalized importance weights.

    Returns:
        Normalized weights summing to 1.
    """
    return weights / np.sum(weights)