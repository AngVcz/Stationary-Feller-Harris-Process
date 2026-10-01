"""Filtro de Kalman para modelos estado-espacio lineales gaussianos."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class FilterResult:
    """Salida del filtro de Kalman sobre una secuencia de observaciones."""

    filtered_means: NDArray[np.float64]
    filtered_covs: NDArray[np.float64]
    predicted_means: NDArray[np.float64]
    predicted_covs: NDArray[np.float64]
    log_likelihoods: NDArray[np.float64]
    innovations: NDArray[np.float64]
    innovation_covs: NDArray[np.float64]


class KalmanFilter:
    """Filtro de Kalman lineal gaussiano (pasos predict/update).

    Modelo:
        X_t = A @ X_{t-1} + v_t,   v_t ~ N(0, Q)
        Y_t = H @ X_t + w_t,       w_t ~ N(0, R)
    """

    def __init__(
        self,
        A: NDArray[np.float64],
        H: NDArray[np.float64],
        Q: NDArray[np.float64],
        R: NDArray[np.float64],
        x0_mean: NDArray[np.float64],
        x0_cov: NDArray[np.float64],
    ):
        self.A = np.atleast_2d(np.asarray(A, dtype=np.float64))
        self.H = np.atleast_2d(np.asarray(H, dtype=np.float64))
        self.Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
        self.R = np.atleast_2d(np.asarray(R, dtype=np.float64))
        self.x0_mean = np.atleast_1d(np.asarray(x0_mean, dtype=np.float64))
        self.x0_cov = np.atleast_2d(np.asarray(x0_cov, dtype=np.float64))

        # estado interno: arranca del prior
        self._x_filt = self.x0_mean.copy()
        self._P_filt = self.x0_cov.copy()
        self._x_pred = self.x0_mean.copy()
        self._P_pred = self.x0_cov.copy()

    def predict(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict: propaga el estado y su covarianza.

        Returns:
            (x_pred, P_pred) — parámetros de N(x_pred, P_pred).
        """
        x_pred = self.A @ self._x_filt
        P_pred = self.A @ self._P_filt @ self.A.T + self.Q
        self._x_pred = x_pred
        self._P_pred = P_pred
        return x_pred, P_pred

    def update(
        self,
        y: NDArray[np.float64],
        P_pred: NDArray[np.float64],
        x_pred: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update: incorpora la observación y.

        Returns:
            (x_filt, P_filt) — parámetros de N(x_filt, P_filt).
        """
        if x_pred is None:
            x_pred = self._x_pred

        y = np.atleast_1d(np.asarray(y, dtype=np.float64))

        # innovación: v = y - H x_pred
        v = y - self.H @ x_pred
        # covarianza de la innovación
        S = self.H @ P_pred @ self.H.T + self.R
        # ganancia de Kalman
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # estado/covarianza filtrados
        x_filt = x_pred + K @ v
        I = np.eye(self.A.shape[0])
        P_filt = (I - K @ self.H) @ P_pred

        # guardar para el siguiente predict
        self._x_filt = x_filt
        self._P_filt = P_filt

        return x_filt, P_filt

    def filter(
        self, observations: NDArray[np.float64]
    ) -> FilterResult:
        """Corre el filtro sobre una secuencia de observaciones.

        Returns:
            FilterResult con medias/covarianzas filtradas y predichas,
            log-verosimilitudes, innovaciones y sus covarianzas.
        """
        observations = np.atleast_2d(observations)
        if observations.shape[0] == 1 and observations.shape[1] > 1:
            # observaciones escalares: (1, T) -> (T, 1)
            observations = observations.T
        T = observations.shape[0]

        n = self.A.shape[0]
        m = self.H.shape[0]

        # reset al prior
        self._x_filt = self.x0_mean.copy()
        self._P_filt = self.x0_cov.copy()

        filtered_means = np.zeros((T, n))
        filtered_covs = np.zeros((T, n))
        predicted_means = np.zeros((T, n))
        predicted_covs = np.zeros((T, n))
        log_likelihoods = np.zeros(T)
        innovations = np.zeros((T, m))
        innovation_covs = np.zeros((T, m))

        for t in range(T):
            # paso 1: predict
            x_pred, P_pred = self.predict()

            # paso 2: update
            y_t = observations[t]
            x_filt, P_filt = self.update(y_t, P_pred, x_pred)

            # paso 3: log-verosimilitud vía innovación
            # log p(y_t) = -0.5 (m log 2pi + log|S| + v' S^{-1} v)
            v = y_t - self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + self.R
            log_lik = -0.5 * (
                m * np.log(2 * np.pi)
                + np.log(np.linalg.det(S))
                + float(v.T @ np.linalg.inv(S) @ v)
            )

            # guardar (solo diagonales, son marginales)
            filtered_means[t] = x_filt.flatten()
            filtered_covs[t] = np.diag(P_filt).flatten()
            predicted_means[t] = x_pred.flatten()
            predicted_covs[t] = np.diag(P_pred).flatten()
            log_likelihoods[t] = log_lik
            innovations[t] = v.flatten()
            innovation_covs[t] = np.diag(S).flatten()

        return FilterResult(
            filtered_means=filtered_means,
            filtered_covs=filtered_covs,
            predicted_means=predicted_means,
            predicted_covs=predicted_covs,
            log_likelihoods=log_likelihoods,
            innovations=innovations,
            innovation_covs=innovation_covs,
        )