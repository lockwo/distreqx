from unittest import TestCase

import jax

# Must be set before any JAX arrays are initialized
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import Empirical, WeightedEmpirical


class EmpiricalTest(TestCase):
    def setUp(self):
        self.key = jax.random.key(0)

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            Empirical(samples=jnp.array([]))

        with self.assertRaisesRegex(ValueError, "at least one dimension"):
            Empirical(samples=jnp.array(5.0))

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_sample_and_stats(self, name, dtype):
        samples = jnp.array([1.0, 1.0, 1.0, 5.0], dtype=dtype)
        dist = Empirical(samples=samples)

        sample = dist.sample(self.key)
        self.assertTrue(sample in samples)
        self.assertEqual(sample.dtype, dtype)

        self.assertion_fn()(dist.mean(), 2.0)
        self.assertion_fn()(dist.mode(), 1.0)
        self.assertion_fn()(dist.variance(), 3.0)

        expected_entropy = -(0.75 * jnp.log(0.75) + 0.25 * jnp.log(0.25))
        self.assertion_fn()(dist.entropy(), expected_entropy)

    def test_multivariate_stats(self):
        # Dataset of 4 2D vectors. Shape (4, 2)
        samples = jnp.array([[0.0, 1.0], [0.0, 1.0], [2.0, 3.0], [2.0, -1.0]])
        dist = Empirical(samples=samples)

        self.assertEqual(dist.event_shape, (2,))

        # Mean component-wise: [(0+0+2+2)/4, (1+1+3-1)/4] = [1.0, 1.0]
        self.assertion_fn()(dist.mean(), jnp.array([1.0, 1.0]))

        # Prob requires exact multi-dimensional match
        self.assertion_fn()(dist.prob(jnp.array([0.0, 1.0])), 0.5)
        self.assertEqual(dist.prob(jnp.array([0.0, 0.0])), 0.0)

        # Joint CDF: % of vectors where x <= 2.0 AND y <= 1.0
        # Matches: [0.0, 1.0], [0.0, 1.0], [2.0, -1.0] -> 3/4
        self.assertion_fn()(dist.cdf(jnp.array([2.0, 1.0])), 0.75)

        # ICDF and Median should be rejected for multivariate
        with self.assertRaisesRegex(NotImplementedError, "intractable"):
            dist.median()
        with self.assertRaisesRegex(NotImplementedError, "intractable"):
            dist.icdf(jnp.array(0.5))

    def test_cdf_and_icdf(self):
        samples = jnp.array([1.0, 2.0, 3.0, 4.0])
        dist = Empirical(samples=samples)

        self.assertEqual(dist.cdf(jnp.array(0.0)), 0.0)
        self.assertEqual(dist.cdf(jnp.array(2.5)), 0.5)

        self.assertEqual(dist.icdf(jnp.array(0.5)), 2.0)

    def test_jittable(self):
        @eqx.filter_jit
        def f(dist):
            return dist.entropy(), dist.mode()

        dist = Empirical(samples=jnp.array([[1.0, 2.0], [1.0, 2.0]]))
        entropy, mode = f(dist)
        self.assertIsInstance(entropy, jax.Array)
        self.assertIsInstance(mode, jax.Array)


class WeightedEmpiricalTest(TestCase):
    def setUp(self):
        self.key = jax.random.key(42)

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "1D array"):
            WeightedEmpirical(samples=jnp.array([1.0, 2.0]), weights=jnp.array([[1.0]]))

        with self.assertRaisesRegex(ValueError, "must match number of samples"):
            WeightedEmpirical(samples=jnp.array([1.0, 2.0]), weights=jnp.array([1.0]))

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_sample_and_stats(self, name, dtype):
        samples = jnp.array([1.0, 3.0], dtype=dtype)
        weights = jnp.array([1.0, 3.0], dtype=dtype)
        dist = WeightedEmpirical(samples=samples, weights=weights)

        self.assertion_fn()(dist.mean(), 2.5)
        self.assertion_fn()(dist.variance(), 0.75)
        self.assertion_fn()(dist.mode(), 3.0)

        self.assertion_fn()(dist.prob(jnp.array(1.0, dtype=dtype)), 0.25)
        self.assertion_fn()(dist.prob(jnp.array(3.0, dtype=dtype)), 0.75)

    def test_multivariate_stats(self):
        samples = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        weights = jnp.array([9.0, 1.0])  # 90% chance of [1.0, 0.0]
        dist = WeightedEmpirical(samples=samples, weights=weights)

        self.assertion_fn()(dist.mean(), jnp.array([0.9, 0.1]))
        self.assertion_fn()(dist.prob(jnp.array([1.0, 0.0])), 0.9)
