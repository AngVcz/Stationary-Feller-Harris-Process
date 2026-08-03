"""Systematic resampling for particle filtering."""
import numpy as np
from numpy.typing import NDArray


def systematic_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Systematic resampling: single random draw, evenly spaced points.

    Generate one uniform u ~ U(0, 1/N), then place points at
    u, u+1/N, u+2/N, ..., u+(N-1)/N. Assign each point to the
    corresponding particle based on cumulative weights.

    Lowest variance among standard resampling schemes. Most commonly
    used in practice.

    Args:
        weights: Normalized importance weights (N,). Must sum to 1.
        rng: Random number generator for reproducibility.

    Returns:
        Array of N resampled indices.
    """
    if rng is None:
        rng = np.random.default_rng()
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)

    # Single random draw, then evenly spaced points
    u = (rng.uniform(size=1)[0] + np.arange(n)) / n

    # Cumulative sum of weights
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # ensure it sums to exactly 1

    # Assign indices
    indices = np.searchsorted(cumsum, u).astype(np.int64)
    indices = np.clip(indices, 0, n - 1)
    return indices