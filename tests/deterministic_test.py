from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import Deterministic


class DeterministicTest(TestCase):
    def setUp(self):
        self.key = jax.random.key(0)

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_invalid_parameters(self):
        # Should raise error if loc is not a scalar
        with self.assertRaises(ValueError):
            Deterministic(loc=jnp.array([1.0, 2.0]))

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_sample_and_stats(self, name, dtype):
        loc_val = jnp.array(5.5, dtype=dtype)
        dist = Deterministic(loc=loc_val)

        # Test Sample
        sample = dist.sample(self.key)
        self.assertion_fn()(sample, loc_val)
        self.assertEqual(sample.dtype, dtype)

        # Test Stats
        self.assertion_fn()(dist.mean(), loc_val)
        self.assertion_fn()(dist.mode(), loc_val)
        self.assertion_fn()(dist.median(), loc_val)
        self.assertion_fn()(dist.variance(), jnp.zeros_like(loc_val))
        self.assertion_fn()(dist.entropy(), jnp.zeros_like(loc_val))

    def test_prob_with_slack(self):
        # Set an absolute tolerance of 0.1
        dist = Deterministic(loc=5.0, atol=0.1)

        # Exact match
        self.assertEqual(dist.prob(jnp.array(5.0)), 1.0)

        # Within slack
        self.assertEqual(dist.prob(jnp.array(5.05)), 1.0)
        self.assertEqual(dist.prob(jnp.array(4.95)), 1.0)

        # Outside slack
        self.assertEqual(dist.prob(jnp.array(5.2)), 0.0)

        # Log Prob bounds check
        self.assertEqual(dist.log_prob(jnp.array(5.0)), 0.0)  # log(1)
        self.assertEqual(dist.log_prob(jnp.array(6.0)), -jnp.inf)  # log(0)

    def test_cdf_and_icdf(self):
        dist = Deterministic(loc=2.0)

        # CDF should be a step function at loc
        self.assertEqual(dist.cdf(jnp.array(1.0)), 0.0)
        self.assertEqual(dist.cdf(jnp.array(2.0)), 1.0)
        self.assertEqual(dist.cdf(jnp.array(3.0)), 1.0)

        # ICDF should return loc for any valid prob
        self.assertEqual(dist.icdf(jnp.array(0.5)), 2.0)
        self.assertTrue(jnp.isnan(dist.icdf(jnp.array(-0.1))))

    def test_kl_divergence(self):
        dist1 = Deterministic(loc=1.0)
        dist2 = Deterministic(loc=1.0)
        dist3 = Deterministic(loc=2.0)

        # KL(dist1 || dist2) should be 0 since they are identical
        self.assertEqual(dist1.kl_divergence(dist2), 0.0)

        # KL(dist1 || dist3) should be infinite since supports don't overlap
        self.assertEqual(dist1.kl_divergence(dist3), jnp.inf)

    def test_jittable(self):
        @eqx.filter_jit
        def f(dist):
            return dist.sample_and_log_prob(key=self.key)

        dist = Deterministic(loc=3.14)
        sample, log_prob = f(dist)

        self.assertIsInstance(sample, jax.Array)
        self.assertIsInstance(log_prob, jax.Array)
