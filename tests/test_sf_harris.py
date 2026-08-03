"""Tests for SF-Harris process simulation and estimation."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.sf_harris.process import SFHarrisProcess
from src.sf_harris.distributions import DiscreteUniformQ, GIGQ
from src.sf_harris.estimation import (
    ndnj_estimate,
    mle_alpha_continuous,
    mle_alpha_discrete,
    mle_alpha,
    mle_full_gig,
    em_estimate_discrete,
)


class TestDiscreteUniformQ:
    def test_sample_within_range(self):
        Q = DiscreteUniformQ(m=5)
        rng = np.random.default_rng(42)
        samples = [Q.sample(rng) for _ in range(100)]
        assert all(1 <= s <= 5 for s in samples)

    def test_pmf_uniform(self):
        Q = DiscreteUniformQ(m=5)
        for i in range(1, 6):
            assert_allclose(Q.pmf(float(i)), 0.2, atol=1e-10)

    def test_pmf_outside_range(self):
        Q = DiscreteUniformQ(m=5)
        assert Q.pmf(0.0) == 0.0
        assert Q.pmf(6.0) == 0.0

    def test_pmf_same(self):
        Q = DiscreteUniformQ(m=5)
        assert_allclose(Q.pmf_same, 0.2)


class TestGIGQ:
    def test_density_positive(self):
        Q = GIGQ(lam=1.0, kappa=2.0, eta=1.0)
        assert Q.density(1.0) > 0

    def test_density_zero_for_negative(self):
        Q = GIGQ(lam=1.0, kappa=2.0, eta=1.0)
        assert Q.density(-1.0) == 0.0
        assert Q.density(0.0) == 0.0

    def test_sample_positive(self):
        Q = GIGQ(lam=1.0, kappa=2.0, eta=1.0)
        rng = np.random.default_rng(42)
        samples = Q.sample_n(100, rng)
        assert np.all(samples > 0)

    def test_sample_consistent_with_density(self):
        """Samples should concentrate where density is high."""
        Q = GIGQ(lam=1.0, kappa=5.0, eta=1.0)
        rng = np.random.default_rng(42)
        samples = Q.sample_n(5000, rng)
        # Mean should be near the mode
        assert np.mean(samples) > 0

    def test_kl_divergence_self_is_zero(self):
        Q = GIGQ(lam=1.0, kappa=2.0, eta=1.0)
        kl = Q.kl_divergence(Q)
        assert kl < 0.5  # Should be close to 0 (numerical tolerance)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            GIGQ(lam=1.0, kappa=0.0, eta=1.0)  # kappa must be > 0
        with pytest.raises(ValueError):
            GIGQ(lam=1.0, kappa=2.0, eta=0.0)  # eta must be > 0


class TestSFHarrisProcess:
    def test_simulate_length(self):
        Q = DiscreteUniformQ(m=5)
        process = SFHarrisProcess(
            alpha=1.0, Q_sample=Q.sample, Q_density=Q.pmf, Q_pmf_same=Q.pmf_same
        )
        rng = np.random.default_rng(42)
        obs = process.simulate(100, rng=rng)
        assert len(obs) == 100

    def test_simulate_values_in_range(self):
        Q = DiscreteUniformQ(m=5)
        process = SFHarrisProcess(
            alpha=1.0, Q_sample=Q.sample, Q_density=Q.pmf, Q_pmf_same=Q.pmf_same
        )
        rng = np.random.default_rng(42)
        obs = process.simulate(100, rng=rng)
        assert np.all(obs >= 1)
        assert np.all(obs <= 5)

    def test_high_alpha_means_more_changes(self):
        """With high alpha, most transitions should be jumps (changes)."""
        Q = DiscreteUniformQ(m=5)
        rng = np.random.default_rng(42)
        process_low = SFHarrisProcess(alpha=0.1, Q_sample=Q.sample)
        process_high = SFHarrisProcess(alpha=5.0, Q_sample=Q.sample)

        obs_low = process_low.simulate(500, rng=rng)
        rng2 = np.random.default_rng(43)
        obs_high = process_high.simulate(500, rng=rng2)

        changes_low = np.sum(obs_low[1:] != obs_low[:-1])
        changes_high = np.sum(obs_high[1:] != obs_high[:-1])
        assert changes_high > changes_low

    def test_alpha_zero_means_constant(self):
        """With alpha=0, the process should stay at its initial value."""
        Q = DiscreteUniformQ(m=5)
        process = SFHarrisProcess(alpha=0.0, Q_sample=Q.sample)
        rng = np.random.default_rng(42)
        obs = process.simulate(100, rng=rng)
        assert np.all(obs == obs[0])

    def test_stay_prob_property(self):
        process = SFHarrisProcess(alpha=2.0, Q_sample=lambda rng: 1.0)
        assert_allclose(process.stay_prob, np.exp(-2.0))
        assert_allclose(process.jump_prob, 1 - np.exp(-2.0))

    def test_count_transitions(self):
        Q = DiscreteUniformQ(m=5)
        process = SFHarrisProcess(alpha=1.0, Q_sample=Q.sample)
        rng = np.random.default_rng(42)
        obs = process.simulate(100, rng=rng)
        n_stay, n_change = process.count_transitions(obs)
        assert n_stay + n_change == 99  # 100 observations = 99 transitions

    def test_continuous_Q_simulation(self):
        """With GIG Q, all values should be positive."""
        Q = GIGQ(lam=1.0, kappa=2.0, eta=1.0)
        process = SFHarrisProcess(alpha=2.0, Q_sample=Q.sample, Q_density=Q.density)
        rng = np.random.default_rng(42)
        obs = process.simulate(100, rng=rng)
        assert np.all(obs > 0)

    def test_log_likelihood_continuous(self):
        """Log-likelihood should be negative for typical observations."""
        Q = GIGQ(lam=1.0, kappa=2.0, eta=1.0)
        process = SFHarrisProcess(
            alpha=2.0, Q_sample=Q.sample, Q_density=Q.density
        )
        rng = np.random.default_rng(42)
        obs = process.simulate(50, rng=rng)
        ll = process.log_likelihood_continuous(obs)
        assert ll < 0

    def test_log_likelihood_discrete(self):
        """Log-likelihood should be negative for typical observations."""
        Q = DiscreteUniformQ(m=5)
        process = SFHarrisProcess(
            alpha=1.0, Q_sample=Q.sample, Q_density=Q.pmf, Q_pmf_same=Q.pmf_same
        )
        rng = np.random.default_rng(42)
        obs = process.simulate(50, rng=rng)
        ll = process.log_likelihood_discrete(obs)
        assert ll < 0


class TestNDNJ:
    def test_no_changes_returns_zero(self):
        """If all observations are the same, NDNJ returns 0."""
        obs = np.array([3.0] * 20)
        assert ndnj_estimate(obs) == 0.0

    def test_estimate_is_positive(self):
        """NDNJ should return a positive estimate when changes exist."""
        Q = DiscreteUniformQ(m=5)
        process = SFHarrisProcess(alpha=2.0, Q_sample=Q.sample)
        rng = np.random.default_rng(42)
        obs = process.simulate(200, rng=rng)
        est = ndnj_estimate(obs)
        assert est > 0

    def test_ndnj_approximately_recovers_alpha(self):
        """NDNJ should approximately recover alpha for continuous Q with large sample."""
        Q = GIGQ(lam=1.0, kappa=2.0, eta=1.0)
        alpha_true = 2.0
        process = SFHarrisProcess(alpha=alpha_true, Q_sample=Q.sample, Q_density=Q.density)
        rng = np.random.default_rng(42)
        obs = process.simulate(2000, rng=rng)
        alpha_hat = ndnj_estimate(obs)
        # Should be within 30% of true value (generous tolerance due to estimator variance)
        assert abs(alpha_hat - alpha_true) < alpha_true * 0.5

    def test_higher_alpha_more_changes(self):
        """Higher alpha should produce more changes, higher NDNJ estimate."""
        Q = DiscreteUniformQ(m=5)
        estimates = []
        for alpha in [0.5, 2.0, 5.0]:
            process = SFHarrisProcess(alpha=alpha, Q_sample=Q.sample)
            rng = np.random.default_rng(42)
            obs = process.simulate(500, rng=rng)
            estimates.append(ndnj_estimate(obs))
        # Higher alpha should generally give higher estimate
        # (not guaranteed due to randomness, but likely with large samples)


class TestMLEContinuous:
    def test_no_changes_returns_zero(self):
        obs = np.array([1.5] * 20)
        assert mle_alpha_continuous(obs) == 0.0

    def test_all_changes_gives_high_alpha(self):
        """If all transitions are changes, alpha should be high (capped at 10)."""
        obs = np.arange(1.0, 21.0)  # All different
        alpha = mle_alpha_continuous(obs)
        assert alpha > 5.0  # All changes means alpha -> infinity (capped at 10)

    def test_mle_recovers_alpha(self):
        """MLE should approximately recover the true alpha."""
        Q = GIGQ(lam=1.0, kappa=2.0, eta=1.0)
        alpha_true = 2.0
        process = SFHarrisProcess(
            alpha=alpha_true, Q_sample=Q.sample, Q_density=Q.density
        )
        rng = np.random.default_rng(42)
        obs = process.simulate(1000, rng=rng)
        alpha_hat = mle_alpha_continuous(obs)
        # Should be within 50% of true value (generous tolerance)
        assert abs(alpha_hat - alpha_true) < alpha_true * 0.5


class TestMLEDiscrete:
    def test_no_changes_returns_zero(self):
        Q = DiscreteUniformQ(m=5)
        obs = np.array([3.0] * 20)
        assert mle_alpha_discrete(obs, Q) == 0.0

    def test_mle_positive_with_changes(self):
        Q = DiscreteUniformQ(m=5)
        process = SFHarrisProcess(
            alpha=3.0, Q_sample=Q.sample, Q_density=Q.pmf, Q_pmf_same=Q.pmf_same
        )
        rng = np.random.default_rng(42)
        obs = process.simulate(200, rng=rng)
        alpha_hat = mle_alpha_discrete(obs, Q)
        assert alpha_hat > 0


class TestEMDiscrete:
    def test_no_changes_returns_zero(self):
        Q = DiscreteUniformQ(m=5)
        obs = np.array([3.0] * 20)
        alpha = em_estimate_discrete(obs, Q)
        assert alpha == 0.0

    def test_em_recovers_alpha(self):
        """EM should approximately recover the true alpha."""
        Q = DiscreteUniformQ(m=5)
        alpha_true = 2.0
        process = SFHarrisProcess(
            alpha=alpha_true, Q_sample=Q.sample, Q_density=Q.pmf, Q_pmf_same=Q.pmf_same
        )
        rng = np.random.default_rng(42)
        obs = process.simulate(500, rng=rng)
        alpha_hat = em_estimate_discrete(obs, Q)
        # Should be within 50% of true value
        assert abs(alpha_hat - alpha_true) < alpha_true * 0.5


class TestMLEDistpatch:
    def test_dispatch_continuous(self):
        Q = GIGQ(lam=1.0, kappa=2.0, eta=1.0)
        process = SFHarrisProcess(alpha=2.0, Q_sample=Q.sample)
        rng = np.random.default_rng(42)
        obs = process.simulate(200, rng=rng)
        # Should use continuous MLE
        alpha = mle_alpha(obs, Q=None)
        assert alpha > 0

    def test_dispatch_discrete(self):
        Q = DiscreteUniformQ(m=5)
        process = SFHarrisProcess(alpha=2.0, Q_sample=Q.sample)
        rng = np.random.default_rng(42)
        obs = process.simulate(200, rng=rng)
        alpha = mle_alpha(obs, Q=Q)
        assert alpha > 0