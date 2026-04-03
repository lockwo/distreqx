from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import Empirical


class EmpiricalTest(TestCase):
    def setUp(self):
        self.key = jax.random.key(0)

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_invalid_parameters(self):
        # Should raise error if samples is not 1D
        with self.assertRaises(ValueError):
            Empirical(samples=jnp.array([[1.0, 2.0], [3.0, 4.0]]))
            
        # Should raise error if samples is empty
        with self.assertRaises(ValueError):
            Empirical(samples=jnp.array([]))

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_sample_and_stats(self, name, dtype):
        # Dataset: three 1.0s, one 5.0
        samples = jnp.array([1.0, 1.0, 1.0, 5.0], dtype=dtype)
        dist = Empirical(samples=samples)

        # Test Sample validity
        sample = dist.sample(self.key)
        self.assertTrue(sample in samples)
        self.assertEqual(sample.dtype, dtype)

        # Test Stats
        self.assertion_fn()(dist.mean(), 2.0)  # (1+1+1+5)/4
        self.assertion_fn()(dist.mode(), 1.0)
        self.assertion_fn()(dist.median(), 1.0)
        
        # Variance of [1, 1, 1, 5] is 3.0
        self.assertion_fn()(dist.variance(), 3.0)

        # P(1.0) = 0.75, P(5.0) = 0.25
        # Entropy = -(0.75 * log(0.75) + 0.25 * log(0.25))
        expected_entropy = -(0.75 * jnp.log(0.75) + 0.25 * jnp.log(0.25))
        self.assertion_fn()(dist.entropy(), expected_entropy)

    def test_prob_with_slack(self):
        # Set an absolute tolerance of 0.1
        samples = jnp.array([1.0, 2.0, 3.0])
        dist = Empirical(samples=samples, atol=0.1)

        # Exact match (1 out of 3 samples)
        self.assertion_fn()(dist.prob(jnp.array(2.0)), 1.0 / 3.0)
        
        # Within slack
        self.assertion_fn()(dist.prob(jnp.array(2.05)), 1.0 / 3.0)
        self.assertion_fn()(dist.prob(jnp.array(1.95)), 1.0 / 3.0)
        
        # Outside slack
        self.assertEqual(dist.prob(jnp.array(2.2)), 0.0)
        
        # Log Prob bounds check
        self.assertion_fn()(dist.log_prob(jnp.array(2.0)), jnp.log(1.0 / 3.0))
        self.assertEqual(dist.log_prob(jnp.array(5.0)), -jnp.inf) # log(0)

    def test_cdf_and_icdf(self):
        samples = jnp.array([1.0, 2.0, 3.0, 4.0])
        dist = Empirical(samples=samples)

        self.assertEqual(dist.cdf(jnp.array(0.0)), 0.0)
        self.assertEqual(dist.cdf(jnp.array(2.5)), 0.5)  # 1.0 and 2.0 are <= 2.5
        self.assertEqual(dist.cdf(jnp.array(4.0)), 1.0)
        self.assertEqual(dist.cdf(jnp.array(5.0)), 1.0)

        # ICDF (Quantile)
        self.assertEqual(dist.icdf(jnp.array(0.0)), 1.0)
        self.assertEqual(dist.icdf(jnp.array(0.5)), 2.5) # Median of [1, 2, 3, 4]
        self.assertEqual(dist.icdf(jnp.array(1.0)), 4.0)

    def test_jittable(self):
        @eqx.filter_jit
        def f(dist):
            # Test that dynamic shape workarounds for mode/entropy don't break JIT
            return dist.entropy(), dist.mode()

        dist = Empirical(samples=jnp.array([1.0, 2.0, 2.0]))
        entropy, mode = f(dist)
        
        self.assertIsInstance(entropy, jax.Array)
        self.assertIsInstance(mode, jax.Array)