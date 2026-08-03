"""Tests for state-space model simulation."""
import numpy as np
import pytest
from src.ssm.models import LinearGaussianSSM, AR1Process
from src.ssm.kernels import gaussian_kernel, normalize_kernel


class TestAR1Process:
    def test_ar1_stationary_mean(self):
        """AR(1) with phi=0.9, sigma=1 should have stationary mean ~0."""
        ar1 = AR1Process(phi=0.9, sigma=1.0)
        states = ar1.simulate(n_steps=10000, x0=0.0)
        assert abs(np.mean(states)) < 0.1

    def test_ar1_stationary_variance(self):
        """AR(1) stationary variance should be sigma^2 / (1 - phi^2)."""
        phi, sigma = 0.8, 1.0
        ar1 = AR1Process(phi=phi, sigma=sigma)
        states = ar1.simulate(n_steps=50000, x0=0.0)
        expected_var = sigma**2 / (1 - phi**2)
        assert abs(np.var(states) - expected_var) < 0.2

    def test_ar1_explosive_rejects(self):
        """AR(1) with |phi| >= 1 should raise ValueError for stationary check."""
        with pytest.raises(ValueError):
            AR1Process(phi=1.1, sigma=1.0, stationary=True)


class TestLinearGaussianSSM:
    def test_ssm_observation_dimension(self):
        """SSM observations should have same length as states."""
        ssm = LinearGaussianSSM(
            transition_phi=0.9,
            transition_sigma=0.5,
            observation_sigma=1.0,
        )
        states, observations = ssm.simulate(n_steps=100, x0=0.0)
        assert len(states) == len(observations) == 100

    def test_ssm_observation_equals_state_plus_noise(self):
        """In LG-SSM, Y_t = X_t + eta_t."""
        ssm = LinearGaussianSSM(
            transition_phi=0.9,
            transition_sigma=0.1,
            observation_sigma=0.1,
        )
        rng = np.random.default_rng(42)
        states, observations = ssm.simulate(n_steps=1000, x0=0.0, rng=rng)
        residuals = observations - states
        # Residuals should have mean ~0 and std ~observation_sigma
        assert abs(np.mean(residuals)) < 0.1
        assert abs(np.std(residuals) - 0.1) < 0.05

    def test_ssm_reproducibility_with_seed(self):
        """Same seed should produce same trajectory."""
        ssm = LinearGaussianSSM(transition_phi=0.9, transition_sigma=0.5, observation_sigma=1.0)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        _, obs1 = ssm.simulate(n_steps=50, x0=0.0, rng=rng1)
        _, obs2 = ssm.simulate(n_steps=50, x0=0.0, rng=rng2)
        np.testing.assert_array_equal(obs1, obs2)


class TestGaussianKernel:
    def test_kernel_integrates_to_one(self):
        """Gaussian kernel should integrate to 1 over all space."""
        import scipy.integrate as integrate

        def kernel_at_x(x):
            return gaussian_kernel(x_prev=0.0, sigma=1.0, x_curr=x)

        result, _ = integrate.quad(kernel_at_x, -np.inf, np.inf)
        assert abs(result - 1.0) < 1e-6

    def test_kernel_symmetry_around_mean(self):
        """Gaussian kernel should be symmetric around its mean: K(x, mean+d) = K(x, mean-d)."""
        phi, sigma = 0.9, 1.0
        x_prev = 2.0
        mean = phi * x_prev
        k_above = gaussian_kernel(x_prev=x_prev, sigma=sigma, x_curr=mean + 0.5, phi=phi)
        k_below = gaussian_kernel(x_prev=x_prev, sigma=sigma, x_curr=mean - 0.5, phi=phi)
        assert abs(k_above - k_below) < 1e-10

    def test_kernel_peak_at_mean(self):
        """Kernel should be maximized at x_curr = phi * x_prev."""
        phi, sigma = 0.9, 1.0
        x_prev = 2.0
        peak_x = phi * x_prev
        k_at_peak = gaussian_kernel(x_prev=x_prev, sigma=sigma, x_curr=peak_x, phi=phi)
        k_off_peak = gaussian_kernel(x_prev=x_prev, sigma=sigma, x_curr=peak_x + 1.0, phi=phi)
        assert k_at_peak > k_off_peak


class TestNormalizeKernel:
    def test_normalized_weights_sum_to_one(self):
        """Normalized importance weights should sum to 1."""
        weights = np.array([2.0, 3.0, 5.0])
        normalized = normalize_kernel(weights)
        assert abs(np.sum(normalized) - 1.0) < 1e-10
        np.testing.assert_allclose(normalized, np.array([0.2, 0.3, 0.5]))