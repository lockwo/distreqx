"""Tests for `uniform.py`."""

from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import Uniform


class UniformTest(TestCase):
    def setUp(self):
        self.key = jax.random.key(0)

    @parameterized.expand(
        [
            ("1d", (0.0, 1.0)),
            ("2d", (np.zeros(2), np.ones(2))),
            ("rank 2", (np.zeros((3, 2)), np.ones((3, 2)))),
            ("broadcasted low", (0.0, np.ones(3))),
            ("broadcasted high", (np.ones(3), 2.0)),
        ]
    )
    def test_event_shape(self, name, distr_params):
        low, high = distr_params
        low = jnp.asarray(low, dtype=jnp.float32)
        high = jnp.asarray(high, dtype=jnp.float32)
        dist = Uniform(low, high)
        expected_shape = low.shape
        self.assertEqual(dist.event_shape, expected_shape)

    @parameterized.expand(
        [
            ("1d, no shape", (0.0, 1.0)),
            ("2d, no shape", (np.zeros(2), np.ones(2))),
            ("rank 2", (np.zeros((3, 2)), np.ones((3, 2)))),
            ("broadcasted low", (0.0, np.ones(3))),
            ("broadcasted high", (np.ones(3), 2.0)),
        ]
    )
    def test_sample_shape(self, name, distr_params):
        low, high = distr_params
        low = jnp.asarray(low, dtype=jnp.float32)
        high = jnp.asarray(high, dtype=jnp.float32)
        dist = Uniform(low, high)
        sample = dist.sample(self.key)
        expected_shape = jnp.broadcast_shapes(low.shape, high.shape)
        self.assertEqual(sample.shape, expected_shape)

    @parameterized.expand(
        [
            ("1d", (0.0, 1.0)),
            ("2d", (np.zeros(2), np.ones(2))),
            ("rank 2", (np.zeros((3, 2)), np.ones((3, 2)))),
            ("broadcasted low", (0.0, np.ones(3))),
            ("broadcasted high", (np.ones(3), 2.0)),
        ]
    )
    @jax.numpy_rank_promotion("raise")
    def test_sample_and_log_prob(self, name, distr_params):
        low, high = distr_params
        low = jnp.asarray(low, dtype=jnp.float32)
        high = jnp.asarray(high, dtype=jnp.float32)
        dist = Uniform(low, high)
        sample, log_prob = dist.sample_and_log_prob(self.key)
        expected_shape = jnp.broadcast_shapes(low.shape, high.shape)
        self.assertEqual(sample.shape, expected_shape)
        self.assertEqual(log_prob.shape, expected_shape)
        # Check that samples are within bounds
        self.assertTrue(jnp.all(sample >= low))
        self.assertTrue(jnp.all(sample <= high))

    @parameterized.expand(
        [
            ("log_prob",),
            ("prob",),
            ("cdf",),
            ("log_cdf",),
            ("survival_function",),
            ("log_survival_function",),
        ]
    )
    def test_method_with_inputs(self, method):
        low, high = -1.0, 1.0
        dist = Uniform(jnp.array(low), jnp.array(high))
        inputs = 10.0 * np.random.normal(size=(100,))
        inputs = jnp.asarray(inputs, dtype=jnp.float32)
        result = getattr(dist, method)(inputs)
        self.assertEqual(result.shape, inputs.shape)

    @parameterized.expand(
        [
            ("entropy", (0.0, 1.0)),
            ("mean", (0, 1)),
            ("variance", (0, 1)),
            ("variance from 1d params", ([0, 0], [1, 2])),
            ("stddev", (0, 1)),
            ("stddev from rank 2 params", (np.zeros((2, 3)), np.ones((2, 3)))),
        ]
    )
    def test_method(self, name, distr_params):
        low, high = distr_params
        low = jnp.asarray(low, dtype=jnp.float32)
        high = jnp.asarray(high, dtype=jnp.float32)
        dist = Uniform(low, high)

        if name.startswith("entropy"):
            result = dist.entropy()
        elif name.startswith("mean"):
            result = dist.mean()
        elif name.startswith("variance"):
            result = dist.variance()
        elif name.startswith("stddev"):
            result = dist.stddev()
        else:
            raise ValueError(f"Unknown method: {name}")

        expected_shape = jnp.broadcast_shapes(low.shape, high.shape)
        self.assertEqual(result.shape, expected_shape)

    def test_median(self):
        dist = Uniform(jnp.array(-1.0), jnp.array(1.0))
        np.testing.assert_allclose(dist.median(), 0.0)
        np.testing.assert_allclose(dist.median(), dist.mean())

    @parameterized.expand(
        [
            ("kl_divergence",),
            ("cross_entropy",),
        ]
    )
    def test_with_two_distributions(self, function_string):
        dist1_kwargs = {
            "low": jnp.array(-0.5 + np.random.rand(3, 2)),
            "high": jnp.array(0.5 + np.random.rand(3, 2)),
        }
        dist2_kwargs = {
            "low": jnp.array(-1.0 + np.random.rand(3, 2)),
            "high": jnp.array(1.5 + np.random.rand(3, 2)),
        }
        dist1 = Uniform(**dist1_kwargs)
        dist2 = Uniform(**dist2_kwargs)

        result = getattr(dist1, function_string)(dist2)
        expected_shape = jnp.broadcast_shapes(
            jnp.broadcast_shapes(dist1_kwargs["low"].shape, dist1_kwargs["high"].shape),
            jnp.broadcast_shapes(dist2_kwargs["low"].shape, dist2_kwargs["high"].shape)
        )
        self.assertEqual(result.shape, expected_shape)

    def test_jittable(self):
        @jax.jit
        def create_and_sample(key):
            dist = Uniform(jnp.array(0.0), jnp.array(1.0))
            return dist.sample(key)

        sample = create_and_sample(self.key)
        self.assertEqual(sample.shape, ())
        self.assertTrue(0.0 <= sample <= 1.0)

    def test_sample_values(self):
        """Test that sample statistics match theoretical values."""
        low = jnp.array([-1.0, 0.0, 2.0])
        high = jnp.array([1.0, 2.0, 5.0])
        dist = Uniform(low, high)

        n_samples = 100000
        keys = jax.random.split(self.key, n_samples)
        sample_fn = jax.jit(jax.vmap(lambda k: dist.sample(k)))
        samples = sample_fn(keys)

        self.assertEqual(samples.shape, (n_samples,) + low.shape)

        # Check mean and variance
        np.testing.assert_allclose(
            np.mean(np.asarray(samples), axis=0),
            dist.mean(),
            rtol=1e-2,
            atol=1e-2
        )
        np.testing.assert_allclose(
            np.var(np.asarray(samples), axis=0),
            dist.variance(),
            rtol=1e-2,
            atol=1e-2
        )

        # Check that all samples are within bounds
        self.assertTrue(np.all(samples >= low))
        self.assertTrue(np.all(samples <= high))

    def test_prob_and_log_prob(self):
        """Test probability density function."""
        low = jnp.array(0.0)
        high = jnp.array(2.0)
        dist = Uniform(low, high)

        # Inside the support
        x_in = jnp.array([0.5, 1.0, 1.5])
        prob_in = dist.prob(x_in)
        expected_prob = 1.0 / (high - low)
        np.testing.assert_allclose(prob_in, expected_prob)

        log_prob_in = dist.log_prob(x_in)
        np.testing.assert_allclose(log_prob_in, jnp.log(expected_prob))

        # Outside the support
        x_out = jnp.array([-0.5, 2.5])
        prob_out = dist.prob(x_out)
        np.testing.assert_allclose(prob_out, 0.0)

        log_prob_out = dist.log_prob(x_out)
        np.testing.assert_allclose(log_prob_out, -jnp.inf)

    def test_cdf_and_survival(self):
        """Test cumulative distribution function and survival function."""
        low = jnp.array(0.0)
        high = jnp.array(1.0)
        dist = Uniform(low, high)

        # Test CDF
        x = jnp.array([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
        cdf = dist.cdf(x)
        expected_cdf = jnp.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
        np.testing.assert_allclose(cdf, expected_cdf, rtol=1e-6)

        # Test log_cdf
        log_cdf = dist.log_cdf(x)
        with np.errstate(divide='ignore'):
            expected_log_cdf = jnp.log(expected_cdf)
        np.testing.assert_allclose(log_cdf, expected_log_cdf, rtol=1e-6)

        # Test survival function
        survival = dist.survival_function(x)
        np.testing.assert_allclose(survival, 1.0 - cdf, rtol=1e-6)

    def test_entropy_values(self):
        """Test entropy calculation."""
        low = jnp.array([0.0, -1.0, 2.0])
        high = jnp.array([1.0, 1.0, 5.0])
        dist = Uniform(low, high)

        expected_entropy = jnp.log(high - low)
        np.testing.assert_allclose(dist.entropy(), expected_entropy, rtol=1e-6)

    def test_kl_divergence_values(self):
        # Test when support of dist1 is contained in dist2
        dist1 = Uniform(jnp.array(0.2), jnp.array(0.8))
        dist2 = Uniform(jnp.array(0.0), jnp.array(1.0))
        kl = dist1.kl_divergence(dist2)
        expected_kl = jnp.log(1.0 - 0.0) - jnp.log(0.8 - 0.2)
        np.testing.assert_allclose(kl, expected_kl, rtol=1e-6)

        # Test when support of dist1 is not contained in dist2
        dist1 = Uniform(jnp.array(-0.5), jnp.array(0.8))
        dist2 = Uniform(jnp.array(0.0), jnp.array(1.0))
        kl = dist1.kl_divergence(dist2)
        np.testing.assert_allclose(kl, jnp.inf)

        # Test KL divergence with itself should be 0
        kl_self = dist1.kl_divergence(dist1)
        np.testing.assert_allclose(kl_self, 0.0, rtol=1e-6)

    def test_vmap(self):
        def log_prob_sum(dist, x):
            return dist.log_prob(x).sum()

        low = jnp.arange(3 * 4 * 5, dtype=jnp.float32).reshape((3, 4, 5))
        high = low + jnp.ones((3, 4, 5))
        dist = Uniform(low, high)
        x = low + 0.5

        with self.subTest("no vmap"):
            actual = log_prob_sum(dist, x)
            expected = dist.log_prob(x).sum()
            np.testing.assert_allclose(actual, expected)

        with self.subTest("axis=0"):
            actual = jax.vmap(log_prob_sum, in_axes=0)(dist, x)
            expected = dist.log_prob(x).sum(axis=(1, 2))
            np.testing.assert_allclose(actual, expected)

        with self.subTest("axis=1"):
            actual = jax.vmap(log_prob_sum, in_axes=1)(dist, x)
            expected = dist.log_prob(x).sum(axis=(0, 2))
            np.testing.assert_allclose(actual, expected)