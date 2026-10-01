"""Resampleo multinomial para filtrado de partículas."""
import numpy as np
from numpy.typing import NDArray


def multinomial_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Multinomial: N índices iid ~ Categorical(W). La de mayor varianza."""
    if rng is None:
        rng = np.random.default_rng()
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)
    return rng.choice(n, size=n, p=weights).astype(np.int64)