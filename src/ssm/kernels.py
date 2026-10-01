"""Núcleos de probabilidad para modelos estado-espacio."""
import numpy as np
from numpy.typing import NDArray


def gaussian_kernel(
    x_prev: float,
    sigma: float,
    x_curr: float,
    phi: float = 0.0,
) -> float:
    """Núcleo gaussiano K(x_prev, x_curr) = N(x_curr; phi*x_prev, sigma^2).

    Con phi=0 es el núcleo gaussiano estándar.
    """
    mean = phi * x_prev
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_curr - mean) / sigma) ** 2
    )


def normalize_kernel(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normaliza pesos de importancia a que sumen 1."""
    return weights / np.sum(weights)