"""Resampleo sistemático para filtrado de partículas."""
import numpy as np
from numpy.typing import NDArray


def systematic_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Sistemático: un solo u ~ U(0, 1/N), puntos en u, u+1/N, ..., u+(N-1)/N.

    La de menor varianza entre los esquemas estándar; la más usada en práctica.
    """
    if rng is None:
        rng = np.random.default_rng()
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)

    # un solo draw aleatorio, luego puntos equiespaciados
    u = (rng.uniform(size=1)[0] + np.arange(n)) / n

    # suma acumulada de pesos
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # asegurar que suma exactamente 1

    # asignar índices
    indices = np.searchsorted(cumsum, u).astype(np.int64)
    indices = np.clip(indices, 0, n - 1)
    return indices