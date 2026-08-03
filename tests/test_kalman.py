"""Tests for the Kalman filter."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.filters.kalman import FilterResult, KalmanFilter


class TestFilterResult:
    def test_filter_result_creation(self):
        """FilterResult should store filter output arrays."""
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
        """Predict step propagates mean and covariance for scalar SSM."""
        kf = KalmanFilter(
            A=np.array([[0.9]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        x_pred, P_pred = kf.predict()
        # x_pred = A @ x0_mean = 0.9 * 0.0 = 0.0
        assert_allclose(x_pred, np.array([0.0]), atol=1e-10)
        # P_pred = A @ P0 @ A.T + Q = 0.9 * 1.0 * 0.9 + 0.1 = 0.91
        assert_allclose(P_pred, np.array([[0.91]]), atol=1e-10)

    def test_predict_covariance_grows_with_unstable_system(self):
        """Predict step increases covariance for unstable system (|A|>1)."""
        kf = KalmanFilter(
            A=np.array([[1.1]]),  # unstable
            H=np.array([[1.0]]),
            Q=np.array([[0.5]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        P0 = kf.x0_cov.copy()
        _, P_pred = kf.predict()
        # P_pred = 1.1^2 * 1.0 + 0.5 = 1.71 > 1.0
        assert P_pred[0, 0] > P0[0, 0]

    def test_predict_stable_system_reduces_variance(self):
        """For stable system (|A|<1), predict can reduce prior uncertainty."""
        kf = KalmanFilter(
            A=np.array([[0.5]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        _, P_pred = kf.predict()
        # P_pred = 0.5^2 * 1.0 + 0.1 = 0.35 < 1.0 (stable system shrinks)
        assert P_pred[0, 0] < 1.0

    def test_predict_multidimensional(self):
        """Predict step works for 2D state."""
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
        """Update step should reduce uncertainty after observation."""
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
        # After update, filtered covariance < predicted covariance
        assert P_filt[0, 0] < P_pred[0, 0]

    def test_update_pulls_toward_observation(self):
        """After update, filtered mean should be between prediction and observation."""
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
        # Filtered mean should be between prediction (0.0) and observation (2.0)
        assert x_pred[0] < x_filt[0] <= 2.0

    def test_update_no_observation_noise_equals_observation(self):
        """With R=0, filtered mean should equal the observation."""
        kf = KalmanFilter(
            A=np.array([[1.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[0.0]]),  # perfect observations
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        x_pred, P_pred = kf.predict()
        x_filt, P_filt = kf.update(np.array([5.0]), P_pred)
        # With R=0, the observation is perfect — filtered state = observation
        assert_allclose(x_filt[0], 5.0, atol=1e-10)
        assert_allclose(P_filt[0, 0], 0.0, atol=1e-10)

    def test_update_innovation_and_gain(self):
        """Check innovation and Kalman gain for scalar case."""
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
        # Innovation: v = y - H @ x_pred
        expected_innovation = y - kf.H @ x_pred
        # Innovation covariance: S = H @ P_pred @ H.T + R
        expected_S = kf.H @ P_pred @ kf.H.T + kf.R
        # Kalman gain: K = P_pred @ H.T @ inv(S)
        expected_K = P_pred @ kf.H.T @ np.linalg.inv(expected_S)

        x_filt, _ = kf.update(y, P_pred)
        expected_x = x_pred + expected_K @ expected_innovation
        assert_allclose(x_filt, expected_x, atol=1e-10)


class TestKalmanFilter:
    def test_filter_scalar_matches_ar1_ssm(self):
        """Kalman filter recovers hidden state from LinearGaussianSSM."""
        from src.ssm.models import LinearGaussianSSM

        ssm = LinearGaussianSSM(
            transition_phi=0.95,
            transition_sigma=0.3,
            observation_sigma=1.0,
        )
        rng = np.random.default_rng(42)
        states, observations = ssm.simulate(n_steps=200, x0=0.0, rng=rng)

        kf = KalmanFilter(
            A=np.array([[0.95]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.09]]),  # 0.3^2
            R=np.array([[1.0]]),
            x0_mean=np.array([0.0]),
            x0_cov=np.array([[1.0]]),
        )
        result = kf.filter(observations)

        rmse_filtered = np.sqrt(np.mean((result.filtered_means.flatten() - states) ** 2))
        rmse_obs_only = np.sqrt(np.mean((observations - states) ** 2))
        assert rmse_filtered < rmse_obs_only, (
            f"Filtered RMSE ({rmse_filtered:.4f}) >= obs-only RMSE ({rmse_obs_only:.4f})"
        )
        assert rmse_filtered < 1.0, f"Filtered RMSE ({rmse_filtered:.4f}) >= obs noise std"

    def test_filter_log_likelihood_negative(self):
        """Log-likelihood should be negative for any observation sequence."""
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
        """Total log-likelihood should match manual computation for known case."""
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
        # Prediction: x_pred=0, P_pred=1+1=2
        # Innovation: v=1.0, S=2+1=3
        # log p(y_1) = -0.5 * (log(2*pi*3) + (1-0)^2/3)
        from scipy.stats import norm
        expected_ll = norm.logpdf(1.0, loc=0.0, scale=np.sqrt(3.0))
        assert_allclose(result.log_likelihoods[0], expected_ll, atol=1e-6)

    def test_filter_result_shapes(self):
        """FilterResult arrays should match observation length."""
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
        """Kalman filter works with n=2 state dimensions."""
        # 2D state: position + velocity (constant velocity model)
        dt = 1.0
        A = np.array([[1.0, dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])  # observe position only
        Q = np.eye(2) * 0.1
        R = np.array([[1.0]])
        x0 = np.array([0.0, 1.0])  # position=0, velocity=1
        P0 = np.eye(2)

        kf = KalmanFilter(A=A, H=H, Q=Q, R=R, x0_mean=x0, x0_cov=P0)

        # Simulate: position increases by ~1 per step (velocity=1)
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

        # Filter should track position reasonably
        rmse = np.sqrt(np.mean((result.filtered_means[:, 0] - true_pos) ** 2))
        assert rmse < 2.0, f"RMSE too high: {rmse:.4f}"

    def test_scalar_shapes_consistent(self):
        """For scalar case, filter result arrays have consistent shapes."""
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
        """End-to-end: simulate SSM, run Kalman filter, verify RMSE and likelihood."""
        from src.ssm.models import LinearGaussianSSM

        phi = 0.95
        sigma_v = 0.3
        sigma_w = 1.0

        ssm = LinearGaussianSSM(
            transition_phi=phi,
            transition_sigma=sigma_v,
            observation_sigma=sigma_w,
        )
        rng = np.random.default_rng(123)
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

        # 1. Filtered RMSE < observation noise std
        rmse_filt = np.sqrt(np.mean((result.filtered_means.flatten() - states) ** 2))
        assert rmse_filt < sigma_w, f"Filtered RMSE {rmse_filt:.4f} >= σ_w {sigma_w}"

        # 2. Filtered RMSE < raw observation RMSE
        rmse_obs = np.sqrt(np.mean((observations - states) ** 2))
        assert rmse_filt < rmse_obs, f"Filtered RMSE {rmse_filt:.4f} >= obs RMSE {rmse_obs:.4f}"

        # 3. Covariance decreases after update (information gain)
        avg_pred_cov = np.mean(result.predicted_covs)
        avg_filt_cov = np.mean(result.filtered_covs)
        assert avg_filt_cov < avg_pred_cov, (
            f"Filtered cov {avg_filt_cov:.4f} >= predicted cov {avg_pred_cov:.4f}"
        )

        # 4. Total log-likelihood is finite and negative
        total_ll = np.sum(result.log_likelihoods)
        assert total_ll < 0, f"Total log-likelihood should be negative, got {total_ll:.4f}"
        assert np.isfinite(total_ll), "Total log-likelihood should be finite"