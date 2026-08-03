"""Multinomial resampling for particle filtering."""
import numpy as np
from numpy.typing import NDArray


def multinomial_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Multinomial resampling: draw N indices with probabilities proportional to weights.

    This is the simplest resampling scheme but has the highest variance.
    Each index is drawn independently from the categorical distribution.

    Args:
        weights: Normalized importance weights (N,). Must sum to 1.
        rng: Random number generator for reproducibility.

    Returns:
        Array of N resampled indices (each in 0..N-1).
    """
    if rng is None:
        rng = np.random.default_rng()
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)
    return rng.choice(n, size=n, p=weights).astype(np.int64)