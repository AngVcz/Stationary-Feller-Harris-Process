"""Tests del filtro de Kalman."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.filters.kalman import FilterResult, KalmanFilter


class TestFilterResult:
    def test_filter_result_creation(self):
        """FilterResult guarda los arreglos de salida."""
        result = FilterResult(
            filtered_means=np.array([0.0, 1.0]),
            filtered_covs=np.array([1.0, 0.5]),
            predicted_means=np.array([0.0, 0.9]),
            predicted_covs=np.array([2.0, 1.0]),
            log_likelihoods=np.array([-1.5, -2.0]),
            innovations=np.array([0.5, -0.3]),
            innovation_covs=np.array([2.0, 1.5]),
        )
        assert result.filtered_means.shape == (2,)
        assert result.log_likelihoods.shape == (2,)


class TestKalmanPredict:
    def test_predict_scalar_state(self):
        """Predicción propaga media y covarianza (SSM escalar)."""
        kf = KalmanFilter(
            A=np.array([[0.9]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        x_pred, P_pred = kf.predict()
        # (a) x_pred = A @ x0 = 0.9*0 = 0.0
        assert_allclose(x_pred, np.array([0.0]), atol=1e-10)
        # (b) P_pred = A P0 A' + Q = 0.9^2 + 0.1 = 0.91 # teórica: 0.91
        assert_allclose(P_pred, np.array([[0.91]]), atol=1e-10)

    def test_predict_covariance_grows_with_unstable_system(self):
        """Sistema inestable (|A|>1): la varianza crece en predict."""
        kf = KalmanFilter(
            A=np.array([[1.1]]),  # inestable
            H=np.array([[1.0]]),
            Q=np.array([[0.5]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        P0 = kf.x0_cov.copy()
        _, P_pred = kf.predict()
        # P_pred = 1.1^2 * 1 + 0.5 = 1.71 > 1.0 # teórica: 1.71
        assert P_pred[0, 0] > P0[0, 0]

    def test_predict_stable_system_reduces_variance(self):
        """Sistema estable (|A|<1): predict puede reducir la varianza."""
        kf = KalmanFilter(
            A=np.array([[0.5]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        _, P_pred = kf.predict()
        # P_pred = 0.5^2 * 1 + 0.1 = 0.35 < 1.0 # teórica: 0.35
        assert P_pred[0, 0] < 1.0

    def test_predict_multidimensional(self):
        """Predict con estado 2D."""
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.1
        R = np.array([[1.0]])
        x0 = np.array([1.0, 0.5])
        P0 = np.eye(2)

        kf = KalmanFilter(A=A, H=H, Q=Q, R=R, x0_mean=x0, x0_cov=P0)
        x_pred, P_pred = kf.predict()

        expected_x = A @ x0
        expected_P = A @ P0 @ A.T + Q
        assert_allclose(x_pred, expected_x, atol=1e-10)
        assert_allclose(P_pred, expected_P, atol=1e-10)


class TestKalmanUpdate:
    def test_update_reduces_covariance(self):
        """Update reduce la covarianza tras observar."""
        kf = KalmanFilter(
            A=np.array([[0.9]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        _, P_pred = kf.predict()
        _, P_filt = kf.update(np.array([1.0]), P_pred)
        # P_filt < P_pred (ganancia de información)
        assert P_filt[0, 0] < P_pred[0, 0]

    def test_update_pulls_toward_observation(self):
        """La media filtrada queda entre predicción y observación."""
        kf = KalmanFilter(
            A=np.array([[0.9]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        x_pred, P_pred = kf.predict()
        x_filt, _ = kf.update(np.array([2.0]), P_pred)
        # entre x_pred=0.0 y obs=2.0
        assert x_pred[0] < x_filt[0] <= 2.0

    def test_update_no_observation_noise_equals_observation(self):
        """R=0: el filtro copia la observación."""
        kf = KalmanFilter(
            A=np.array([[1.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[0.0]]),  # obs perfectas
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        x_pred, P_pred = kf.predict()
        x_filt, P_filt = kf.update(np.array([5.0]), P_pred)
        # R=0 => x_filt = obs, P_filt = 0 # teórica: 5.0, 0.0
        assert_allclose(x_filt[0], 5.0, atol=1e-10)
        assert_allclose(P_filt[0, 0], 0.0, atol=1e-10)

    def test_update_innovation_and_gain(self):
        """Innovación y ganancia de Kalman (caso escalar)."""
        kf = KalmanFilter(
            A=np.array([[0.9]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        x_pred, P_pred = kf.predict()
        y = np.array([1.5])
        # v = y - H x_pred, S = H P_pred H' + R, K = P_pred H' S^{-1}
        expected_innovation = y - kf.H @ x_pred
        expected_S = kf.H @ P_pred @ kf.H.T + kf.R
        expected_K = P_pred @ kf.H.T @ np.linalg.inv(expected_S)

        x_filt, _ = kf.update(y, P_pred)
        expected_x = x_pred + expected_K @ expected_innovation
        assert_allclose(x_filt, expected_x, atol=1e-10)


class TestKalmanFilter:
    def test_filter_scalar_matches_ar1_ssm(self):
        """Kalman recupera el estado oculto de un AR(1) simulado."""
        from src.ssm.models import LinearGaussianSSM

        ssm = LinearGaussianSSM(
            transition_phi=0.95,
            transition_sigma=0.3,
            observation_sigma=1.0,
        )
        rng = np.random.default_rng(42)  # semilla 42
        states, observations = ssm.simulate(n_steps=200, x0=0.0, rng=rng)

        kf = KalmanFilter(
            A=np.array([[0.95]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.09]]),  # sigma_v^2 = 0.3^2
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        result = kf.filter(observations)

        # RMSE filtrado < RMSE de obs crudas < sigma_w = 1.0
        rmse_filtered = np.sqrt(np.mean((result.filtered_means.flatten() - states) ** 2))
        rmse_obs_only = np.sqrt(np.mean((observations - states) ** 2))
        assert rmse_filtered < rmse_obs_only, (
            f"Filtered RMSE ({rmse_filtered:.4f}) >= obs-only RMSE ({rmse_obs_only:.4f})"
        )
        assert rmse_filtered < 1.0, f"Filtered RMSE ({rmse_filtered:.4f}) >= obs noise std"

    def test_filter_log_likelihood_negative(self):
        """Log-verosimilitud negativa para cualquier secuencia."""
        kf = KalmanFilter(
            A=np.array([[0.9]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        observations = np.array([1.0, -0.5, 0.3, 1.2, -0.8])
        result = kf.filter(observations)
        assert np.all(result.log_likelihoods < 0)

    def test_filter_log_likelihood_matches_manual(self):
        """Log-verosimilitud total vs cálculo manual (caso conocido)."""
        kf = KalmanFilter(
            A=np.array([[1.0]]),  # random walk
            H=np.array([[1.0]]),
            Q=np.array([[1.0]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        observations = np.array([1.0])
        result = kf.filter(observations)
        # x_pred=0, P_pred=2, v=1, S=3 -> log p(y_1) ~ N(0, 3) # teórica: norm.logpdf(1; 0, sqrt(3))
        from scipy.stats import norm
        expected_ll = norm.logpdf(1.0, loc=0.0, scale=np.sqrt(3.0))
        assert_allclose(result.log_likelihoods[0], expected_ll, atol=1e-6)

    def test_filter_result_shapes(self):
        """Formas de FilterResult consistentes con T observaciones."""
        kf = KalmanFilter(
            A=np.array([[0.9]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        T = 50
        observations = np.random.randn(T)
        result = kf.filter(observations)
        assert result.filtered_means.shape == (T, 1)
        assert result.filtered_covs.shape == (T, 1)
        assert result.predicted_means.shape == (T, 1)
        assert result.predicted_covs.shape == (T, 1)
        assert result.log_likelihoods.shape == (T,)
        assert result.innovations.shape == (T, 1)
        assert result.innovation_covs.shape == (T, 1)


class TestKalmanMultidimensional:
    def test_multidimensional_state(self):
        """Filtro Kalman con estado 2D (posición + velocidad)."""
        # modelo velocidad constante, solo observamos posición
        dt = 1.0
        A = np.array([[1.0, dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.1
        R = np.array([[1.0]])
        x0 = np.array([0.0, 1.0])  # pos=0, vel=1
        P0 = np.eye(2)

        kf = KalmanFilter(A=A, H=H, Q=Q, R=R, x0_mean=x0, x0_cov=P0)

        # posición crece ~1 por paso (vel=1), semilla 42
        rng = np.random.default_rng(42)
        T = 30
        observations = np.zeros((T, 1))
        true_pos = np.zeros(T)
        pos, vel = 0.0, 1.0
        for t in range(T):
            pos = pos + vel * dt + rng.normal(0, np.sqrt(0.1))
            vel = vel + rng.normal(0, np.sqrt(0.1))
            true_pos[t] = pos
            observations[t, 0] = pos + rng.normal(0, 1.0)

        result = kf.filter(observations)

        # el filtro debe seguir la posición razonablemente
        rmse = np.sqrt(np.mean((result.filtered_means[:, 0] - true_pos) ** 2))
        assert rmse < 2.0, f"RMSE too high: {rmse:.4f}"

    def test_scalar_shapes_consistent(self):
        """Formas consistentes en el caso escalar."""
        kf = KalmanFilter(
            A=np.array([[0.9]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        observations = np.array([1.0, -0.5, 0.3, 1.2, -0.8])
        result = kf.filter(observations)
        assert result.filtered_means.shape == (5, 1)
        assert result.log_likelihoods.shape == (5,)


class TestKalmanIntegration:
    def test_end_to_end_with_ssm(self):
        """End-to-end: simular SSM, filtrar y verificar RMSE y log-verosimilitud."""
        from src.ssm.models import LinearGaussianSSM

        phi = 0.95
        sigma_v = 0.3
        sigma_w = 1.0

        ssm = LinearGaussianSSM(
            transition_phi=phi,
            transition_sigma=sigma_v,
            observation_sigma=sigma_w,
        )
        rng = np.random.default_rng(123)  # semilla 123
        states, observations = ssm.simulate(n_steps=500, x0=0.0, rng=rng)

        kf = KalmanFilter(
            A=np.array([[phi]]),
            H=np.array([[1.0]]),
            Q=np.array([[sigma_v**2]]),
            R=np.array([[sigma_w**2]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        result = kf.filter(observations)

        # 1) RMSE filtrado < sigma_w (ruido de obs)
        rmse_filt = np.sqrt(np.mean((result.filtered_means.flatten() - states) ** 2))
        assert rmse_filt < sigma_w, f"Filtered RMSE {rmse_filt:.4f} >= σ_w {sigma_w}"

        # 2) RMSE filtrado < RMSE de obs crudas
        rmse_obs = np.sqrt(np.mean((observations - states) ** 2))
        assert rmse_filt < rmse_obs, f"Filtered RMSE {rmse_filt:.4f} >= obs RMSE {rmse_obs:.4f}"

        # 3) cov filtrada < cov predicha (ganancia de información)
        avg_pred_cov = np.mean(result.predicted_covs)
        avg_filt_cov = np.mean(result.filtered_covs)
        assert avg_filt_cov < avg_pred_cov, (
            f"Filtered cov {avg_filt_cov:.4f} >= predicted cov {avg_pred_cov:.4f}"
        )

        # 4) log-verosimilitud total finita y negativa
        total_ll = np.sum(result.log_likelihoods)
        assert total_ll < 0, f"Total log-likelihood should be negative, got {total_ll:.4f}"
        assert np.isfinite(total_ll), "Total log-likelihood should be finite"