"""Stratified resampling for particle filtering."""
import numpy as np
from numpy.typing import NDArray


def stratified_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Stratified resampling: one random point per stratum of [0,1].

    Divide [0,1] into N strata of size 1/N. In each stratum, draw
    one uniform point and assign the corresponding particle.

    Lower variance than multinomial resampling.

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

    # Stratified points: one uniform per stratum
    u = (np.arange(n) + rng.uniform(size=n)) / n

    # Cumulative sum of weights
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # ensure it sums to exactly 1

    # Assign indices
    indices = np.searchsorted(cumsum, u).astype(np.int64)
    # Clip to valid range (shouldn't be needed but numerical safety)
    indices = np.clip(indices, 0, n - 1)
    return indices