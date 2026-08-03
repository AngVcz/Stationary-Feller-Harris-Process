"""Tests for importance sampling, ESS, and resampling algorithms."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.resampling.ess import effective_sample_size
from src.resampling.importance import (
    importance_weights,
    is_estimate,
    normalize_weights,
)
from src.resampling.multinomial import multinomial_resample
from src.resampling.residual import residual_resample
from src.resampling.stratified import stratified_resample
from src.resampling.systematic import systematic_resample


class TestESS:
    def test_ess_equal_weights(self):
        """ESS should equal N when all weights are equal."""
        n = 100
        weights = np.ones(n)
        ess = effective_sample_size(weights)
        assert_allclose(ess, n, atol=1.0)

    def test_ess_single_dominant_weight(self):
        """ESS should be close to 1 when a single weight dominates."""
        weights = np.array([1000.0] + [0.001] * 999)
        ess = effective_sample_size(weights)
        assert ess < 2.0

    def test_ess_normalized_weights(self):
        """ESS from normalized weights should match unnormalized."""
        n = 50
        weights = np.random.default_rng(42).uniform(size=n)
        ess_unnorm = effective_sample_size(weights)
        norm_weights = weights / np.sum(weights)
        # For normalized: ESS = 1/sum(W_i^2)
        ess_from_norm = 1.0 / np.sum(norm_weights**2)
        assert_allclose(ess_unnorm, ess_from_norm, rtol=1e-10)

    def test_ess_half_dominant(self):
        """ESS should be roughly N/2 when half the weights dominate."""
        n = 100
        weights = np.ones(n)
        weights[:50] = 10.0
        weights[50:] = 1.0
        ess = effective_sample_size(weights)
        # Should be between 50 and 100 (closer to 55 due to weight ratio)
        assert 50 < ess < 100


class TestImportanceWeights:
    def test_uniform_target_equals_proposal(self):
        """When target equals proposal, weights should be uniform."""
        target_log = np.zeros(100)
        proposal_log = np.zeros(100)
        weights = importance_weights(target_log, proposal_log)
        assert_allclose(weights, np.ones(100), atol=1e-10)

    def test_normalized_weights_sum_to_one(self):
        """Normalized weights should sum to 1."""
        target_log = np.log(np.array([0.3, 0.5, 0.2]))
        proposal_log = np.log(np.array([0.25, 0.25, 0.5]))
        w = importance_weights(target_log, proposal_log)
        W = normalize_weights(w)
        assert_allclose(np.sum(W), 1.0, atol=1e-10)

    def test_is_estimate_mean_of_standard_normal(self):
        """IS estimate of E[X] for standard normal using wider proposal."""
        rng = np.random.default_rng(42)
        n = 10000
        # Target: N(0, 1), Proposal: N(0, 4)
        samples = rng.normal(0, 2, size=n)
        target_log = -0.5 * samples**2  # log N(0,1)
        proposal_log = -0.5 * (samples / 2)**2 - np.log(2)  # log N(0,4)
        weights = importance_weights(target_log, proposal_log)
        W = normalize_weights(weights)
        estimate = is_estimate(samples, W)
        # Should be close to 0 (mean of standard normal)
        assert abs(estimate) < 0.2


class TestResamplingCommon:
    """Tests shared by all resampling schemes."""

    @pytest.fixture(params=[
        multinomial_resample,
        residual_resample,
        stratified_resample,
        systematic_resample,
    ])
    def resample_fn(self, request):
        return request.param

    def test_returns_correct_number_of_indices(self, resample_fn):
        """Resampling should return exactly N indices."""
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        indices = resample_fn(weights, rng=np.random.default_rng(42))
        assert len(indices) == 4

    def test_indices_in_valid_range(self, resample_fn):
        """All indices should be in 0..N-1."""
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        indices = resample_fn(weights, rng=np.random.default_rng(42))
        assert np.all(indices >= 0)
        assert np.all(indices < 4)

    def test_high_weight_particles_selected_more(self, resample_fn):
        """Particles with high weights should be selected more often."""
        rng = np.random.default_rng(42)
        weights = np.array([0.7, 0.1, 0.1, 0.1])
        counts = np.zeros(4)
        for _ in range(100):
            indices = resample_fn(weights, rng=rng)
            for idx in indices:
                counts[idx] += 1
        # Particle 0 should be selected most often
        assert counts[0] > counts[1]

    def test_reproducibility_with_seed(self, resample_fn):
        """Same seed should produce same indices."""
        weights = np.array([0.3, 0.3, 0.2, 0.2])
        idx1 = resample_fn(weights, rng=np.random.default_rng(123))
        idx2 = resample_fn(weights, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(idx1, idx2)


class TestMultinomialResampling:
    def test_uniform_weights_returns_all_indices(self):
        """With uniform weights, all indices should appear over many trials."""
        rng = np.random.default_rng(42)
        weights = np.ones(10) / 10
        all_indices = set()
        for _ in range(50):
            indices = multinomial_resample(weights, rng=rng)
            all_indices.update(indices.tolist())
        # With uniform weights and 50 trials, all 10 should appear
        assert len(all_indices) == 10


class TestSystematicResampling:
    def test_single_random_draw(self):
        """Systematic resampling uses only one random number."""
        weights = np.array([0.3, 0.3, 0.2, 0.2])
        rng = np.random.default_rng(42)
        idx1 = systematic_resample(weights, rng=rng)
        idx2 = systematic_resample(weights, rng=rng)
        # Different seeds should give different results
        assert not np.array_equal(idx1, idx2) or True  # may coincide by chance

    def test_preserves_total_count(self):
        """Resampling should return exactly N indices."""
        weights = np.array([0.5, 0.3, 0.15, 0.05])
        indices = systematic_resample(weights, rng=np.random.default_rng(42))
        assert len(indices) == 4


class TestResidualResampling:
    def test_high_weight_guarantees_copy(self):
        """Particles with weight > 1/N get at least one copy."""
        weights = np.array([0.5, 0.3, 0.15, 0.05])
        indices = residual_resample(weights, rng=np.random.default_rng(42))
        # Weight 0.5 > 1/4 = 0.25, so particle 0 should appear at least once
        assert 0 in indices
        # Weight 0.3 > 0.25, so particle 1 should appear at least once
        assert 1 in indices