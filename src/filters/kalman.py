"""Kalman filter for linear Gaussian state-space models."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class FilterResult:
    """Output of a Kalman filter run over an observation sequence."""

    filtered_means: NDArray[np.float64]
    filtered_covs: NDArray[np.float64]
    predicted_means: NDArray[np.float64]
    predicted_covs: NDArray[np.float64]
    log_likelihoods: NDArray[np.float64]
    innovations: NDArray[np.float64]
    innovation_covs: NDArray[np.float64]


class KalmanFilter:
    """Linear Gaussian Kalman filter with predict/update steps.

    State-space model:
        X_t = A @ X_{t-1} + v_t,   v_t ~ N(0, Q)
        Y_t = H @ X_t + w_t,       w_t ~ N(0, R)

    Args:
        A: State transition matrix (n x n).
        H: Observation matrix (m x n).
        Q: Process noise covariance (n x n).
        R: Observation noise covariance (m x m).
        x0_mean: Initial state mean (n,).
        x0_cov: Initial state covariance (n x n).
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

        # Internal filter state: start from prior
        self._x_filt = self.x0_mean.copy()
        self._P_filt = self.x0_cov.copy()
        self._x_pred = self.x0_mean.copy()
        self._P_pred = self.x0_cov.copy()

    def predict(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict step: propagate state and covariance forward.

        Returns:
            Tuple of (x_pred, P_pred) — the predictive distribution
            parameters N(x_pred, P_pred).
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
        """Update step: incorporate observation y.

        Args:
            y: Observation vector (m,).
            P_pred: Predicted covariance from predict step (n x n).
            x_pred: Predicted mean from predict step. If None, uses
                the last predict step result stored internally.

        Returns:
            Tuple of (x_filt, P_filt) — the filtered distribution
            parameters N(x_filt, P_filt).
        """
        if x_pred is None:
            x_pred = self._x_pred

        y = np.atleast_1d(np.asarray(y, dtype=np.float64))

        # Innovation
        v = y - self.H @ x_pred
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Filtered state and covariance
        x_filt = x_pred + K @ v
        I = np.eye(self.A.shape[0])
        P_filt = (I - K @ self.H) @ P_pred

        # Store internal state for next predict
        self._x_filt = x_filt
        self._P_filt = P_filt

        return x_filt, P_filt

    def filter(
        self, observations: NDArray[np.float64]
    ) -> FilterResult:
        """Run Kalman filter over an observation sequence.

        Args:
            observations: Array of observations (T,) for scalar or
                (T, m) for multidimensional.

        Returns:
            FilterResult with filtered/predicted means and covariances,
            log-likelihoods, innovations, and innovation covariances.
        """
        observations = np.atleast_2d(observations)
        if observations.shape[0] == 1 and observations.shape[1] > 1:
            # Scalar observations: shape (1, T) -> (T, 1)
            observations = observations.T
        T = observations.shape[0]

        n = self.A.shape[0]
        m = self.H.shape[0]

        # Reset to prior
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
            # Predict
            x_pred, P_pred = self.predict()

            # Update
            y_t = observations[t]
            x_filt, P_filt = self.update(y_t, P_pred, x_pred)

            # Compute log-likelihood from innovation
            v = y_t - self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + self.R
            log_lik = -0.5 * (
                m * np.log(2 * np.pi)
                + np.log(np.linalg.det(S))
                + float(v.T @ np.linalg.inv(S) @ v)
            )

            # Store results
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