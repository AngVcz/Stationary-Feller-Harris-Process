"""Residual resampling for particle filtering."""
import numpy as np
from numpy.typing import NDArray


def residual_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Residual resampling: deterministic copies + multinomial for remainders.

    1. Take floor(N * W_i) copies of each particle (deterministic part)
    2. Draw remaining copies multinomially from the residuals

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

    # Deterministic part: floor(N * W_i) copies
    copies = np.floor(n * weights).astype(int)
    residual = n * weights - copies

    # How many remain?
    remaining = n - np.sum(copies)

    # Multinomial draw from the residuals
    if remaining > 0:
        residual_normalized = residual / np.sum(residual)
        extra_indices = rng.choice(n, size=remaining, p=residual_normalized)
    else:
        extra_indices = np.array([], dtype=np.int64)

    # Build index array from deterministic copies + extra draws
    indices = np.concatenate([
        np.repeat(np.arange(n), copies),
        extra_indices,
    ])

    # Shuffle to avoid ordering bias
    rng.shuffle(indices)
    return indices.astype(np.int64)