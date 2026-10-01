"""Resampleo estratificado para filtrado de partículas."""
import numpy as np
from numpy.typing import NDArray


def stratified_resample(
    weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Estratificado: [0,1] en N estratos de tamaño 1/N, un uniforme por estrato.

    Menor varianza que el multinomial.
    """
    if rng is None:
        rng = np.random.default_rng()
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)

    # puntos estratificados: un uniforme por estrato
    u = (np.arange(n) + rng.uniform(size=n)) / n

    # suma acumulada de pesos
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # asegurar que suma exactamente 1

    # asignar índices
    indices = np.searchsorted(cumsum, u).astype(np.int64)
    indices = np.clip(indices, 0, n - 1)  # seguridad numérica
    return indices