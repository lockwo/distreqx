"""Tests for `independent.py`."""

from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from distreqx.distributions import Bernoulli, Independent, MultivariateNormalTri, Normal


class IndependentTest(TestCase):
    """Class to test miscellaneous methods of the `Independent` distribution."""

    def setUp(self):
        np.random.seed(42)
        self.loc = jnp.array(np.random.randn(2, 3, 4))
        self.scale = jnp.array(np.abs(np.random.randn(2, 3, 4)))
        self.base = Normal(loc=self.loc, scale=self.scale)
        self.dist = Independent(self.base)
        self.key = jax.random.key(0)

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def _create_vmapped_mvn(self, M=20, N=10, D=3):
        """Helper to generate a 2D vmapped MultivariateNormalTri."""
        locs = jnp.zeros((M, N, D))
        scales_tri = jnp.stack([jnp.tri(D)] * M * N, axis=0).reshape(M, N, D, D)
        batch_mvn = eqx.filter_vmap(eqx.filter_vmap(MultivariateNormalTri))(
            locs, scales_tri
        )
        return batch_mvn, locs, scales_tri

    def test_constructor_is_jittable_given_ndims(self):
        constructor = lambda d: Independent(d)
        model = eqx.filter_jit(constructor)(self.base)
        self.assertIsInstance(model, Independent)

    def test_mapped_event_shape_inference(self):
        """
        Tests the eqx.filter_eval_shape logic
        for dynamically vmapped distributions.
        """
        M, N, D = 20, 10, 3

        # Create a distribution vmapped exactly ONCE
        locs_1d = jnp.zeros((M, D))
        scales_tri_1d = jnp.stack([jnp.tri(D)] * M, axis=0).reshape(M, D, D)
        batch_mvn_1d = eqx.filter_vmap(MultivariateNormalTri)(locs_1d, scales_tri_1d)

        indep_1d = Independent(batch_mvn_1d, reinterpreted_batch_ndims=1)
        self.assertEqual(
            indep_1d.event_shape, (M, D)
        )  # (20, 3) because base is (3,) and 1st vmap is over M

        # Create a distribution vmapped exactly TWICE using our helper
        batch_mvn_2d, _, _ = self._create_vmapped_mvn(M, N, D)

        indep_2d = Independent(batch_mvn_2d, reinterpreted_batch_ndims=2)
        self.assertEqual(indep_2d.event_shape, (M, N, D))  # (20, 10, 3)

    def test_multiple_reinterpret_log_prob(self):
        """Tests that stacked mapped log_probs evaluate correctly."""
        M, N, D = 20, 10, 3
        batch_mvn, locs, scales_tri = self._create_vmapped_mvn(M, N, D)
        xs = jnp.ones((M, N, D))

        mvn = Independent(batch_mvn, reinterpreted_batch_ndims=2)

        # Get the underlying log prob and multiply by the number
        # of independent dimensions (Since all locs and scales are
        # identical in this test setup, summing is equal to multiplying)
        log_prob_underlying = (
            MultivariateNormalTri(locs[0][0], scales_tri[0][0]).log_prob(xs[0][0])
            * M
            * N
        )
        log_prob_independent = mvn.log_prob(xs)

        self.assertion_fn(1e-5)(log_prob_underlying, log_prob_independent)

    def test_sample_and_log_prob_equivalence(self):
        """
        Tests PRNG key splitting logic and equivalence
        to separate sample + log_prob calls.
        """
        M, N, D = 5, 4, 3
        batch_mvn, _, _ = self._create_vmapped_mvn(M, N, D)
        mvn = Independent(batch_mvn, reinterpreted_batch_ndims=2)

        # Draw from sample_and_log_prob
        samples, log_probs = mvn.sample_and_log_prob(self.key)

        # Verify sample shape
        self.assertEqual(samples.shape, mvn.event_shape)

        # Verify log_prob from sample_and_log_prob matches a manual call to log_prob
        expected_log_probs = mvn.log_prob(samples)
        self.assertion_fn()(log_probs, expected_log_probs)

    def test_moments_and_entropy(self):
        """Tests that moments map correctly and entropy sums over the right axes."""
        M, N, D = 5, 4, 3
        batch_mvn, locs, _ = self._create_vmapped_mvn(M, N, D)
        mvn = Independent(batch_mvn, reinterpreted_batch_ndims=2)

        # Mean should be the shape of the full reinterpreted event
        mean = mvn.mean()
        self.assertEqual(mean.shape, (M, N, D))
        self.assertion_fn()(mean, locs)  # For a MVN with zero locs, mean is zeros

        # Variance should also map correctly without crashing
        variance = mvn.variance()
        self.assertEqual(variance.shape, (M, N, D))

        # Entropy should be summed over the 2 reinterpreted dimensions
        entropy = mvn.entropy()
        self.assertEqual(entropy.shape, ())  # Reduced to scalar

    def test_kl_divergence(self):
        """
        Tests that KL divergence maps and sums correctly
        between identical/different distributions.
        """
        M, N, D = 5, 4, 3
        batch_mvn_p, _, scales_tri_p = self._create_vmapped_mvn(M, N, D)

        # Create a second distribution with slightly different scales
        locs_q = jnp.zeros((M, N, D))
        scales_tri_q = scales_tri_p * 2.0
        batch_mvn_q = eqx.filter_vmap(eqx.filter_vmap(MultivariateNormalTri))(
            locs_q, scales_tri_q
        )

        p = Independent(batch_mvn_p, reinterpreted_batch_ndims=2)
        q = Independent(batch_mvn_q, reinterpreted_batch_ndims=2)

        # KL(p || p) should be 0
        kl_self = p.kl_divergence(p)
        self.assertion_fn()(kl_self, 0.0)

        # KL(p || q) should evaluate properly and reduce to a scalar
        kl_pq = p.kl_divergence(q)
        self.assertEqual(kl_pq.shape, ())
        self.assertTrue(kl_pq > 0.0)  # KL should be strictly positive

    def test_reinterpretation_across_distribution_types(self):
        """Checks reinterpretation doesn't misbehave for natively-broadcasting
        distributions other than Normal (e.g. discrete distributions like
        Bernoulli), not just the vmap-constructed MultivariateNormalTri case."""
        M, N = 5, 4

        bernoulli = Bernoulli(probs=jnp.full((M, N), 0.3))
        indep_bernoulli = Independent(bernoulli, reinterpreted_batch_ndims=2)
        self.assertEqual(indep_bernoulli.event_shape, (M, N))

        xs = jnp.ones((M, N))
        expected_log_prob = jnp.sum(bernoulli.log_prob(xs))
        self.assertion_fn()(indep_bernoulli.log_prob(xs), expected_log_prob)

        sample = indep_bernoulli.sample(self.key)
        self.assertEqual(sample.shape, (M, N))

        mean = indep_bernoulli.mean()
        self.assertEqual(mean.shape, (M, N))

        normal = Normal(loc=jnp.zeros((M, N)), scale=jnp.ones((M, N)))
        indep_normal = Independent(normal, reinterpreted_batch_ndims=1)
        self.assertEqual(indep_normal.event_shape, (M, N))
        self.assertion_fn()(
            indep_normal.log_prob(xs), jnp.sum(normal.log_prob(xs), axis=0)
        )

    def test_kl_divergence_requires_matching_reinterpreted_batch_ndims(self):
        # Same underlying (natively-broadcasting) distribution and event_shape,
        # but reinterpreted over a different number of axes.
        normal = Normal(loc=jnp.zeros((5, 4)), scale=jnp.ones((5, 4)))
        p = Independent(normal, reinterpreted_batch_ndims=2)
        q = Independent(normal, reinterpreted_batch_ndims=1)
        self.assertEqual(p.event_shape, q.event_shape)

        with self.assertRaisesRegex(ValueError, "reinterpreted_batch_ndims"):
            p.kl_divergence(q)

    def test_methods_are_jittable(self):
        """Ensures that wrapped mapping logic survives JAX compilation."""
        M, N, D = 5, 4, 3
        batch_mvn, _, _ = self._create_vmapped_mvn(M, N, D)
        mvn = Independent(batch_mvn, reinterpreted_batch_ndims=2)
        xs = jnp.ones((M, N, D))

        # JIT compile the log_prob and sample methods
        jitted_log_prob = eqx.filter_jit(lambda d, x: d.log_prob(x))
        jitted_sample = eqx.filter_jit(lambda d, k: d.sample(k))

        # If abstract tracing or shape inference fails, these will crash
        lp = jitted_log_prob(mvn, xs)
        samples = jitted_sample(mvn, self.key)

        self.assertEqual(lp.shape, ())
        self.assertEqual(samples.shape, (M, N, D))
