"""Resampleo residual para filtrado de partículas."""
import numpy as np
from numpy.typing import NDArray


def residual_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Residual: floor(N*W_i) copias deterministas + multinomial sobre residuos.

    Menor varianza que el multinomial.
    """
    if rng is None:
        rng = np.random.default_rng()
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)

    # parte determinista: floor(N * W_i) copias
    copies = np.floor(n * weights).astype(int)
    residual = n * weights - copies

    # cuántas faltan
    remaining = n - np.sum(copies)

    # parte aleatoria: multinomial sobre los residuos
    if remaining > 0:
        residual_normalized = residual / np.sum(residual)
        extra_indices = rng.choice(n, size=remaining, p=residual_normalized)
    else:
        extra_indices = np.array([], dtype=np.int64)

    # juntar copias deterministas + extras
    indices = np.concatenate([
        np.repeat(np.arange(n), copies),
        extra_indices,
    ])

    # mezclar para no introducir sesgo de orden
    rng.shuffle(indices)
    return indices.astype(np.int64)