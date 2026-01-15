from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import _gamma as gamma_module


class GammaTest(TestCase):
    def setUp(self):
        self.dist_cls = gamma_module.Gamma
        self.concentration = np.asarray([0.5, 1.0, 2.5, 5.0])
        self.rate = np.asarray([0.5, 2.0, 1.5, 1.0])
        self.key = jax.random.key(0)

    @parameterized.expand(
        [
            (
                "0d params",
                (),
            ),
            ("1d params", (4,)),
            ("2d params", (3, 4)),
        ]
    )
    def test_properties(self, name, shape):
        rng = np.random.default_rng(0)
        concentration = rng.uniform(0.1, 5.0, size=shape)
        rate = rng.uniform(0.1, 5.0, size=shape)
        dist = self.dist_cls(
            concentration=jnp.array(concentration), rate=jnp.array(rate)
        )
        np.testing.assert_allclose(dist.concentration, concentration, rtol=1e-6)
        np.testing.assert_allclose(dist.rate, rate, rtol=1e-6)

    @parameterized.expand(
        [
            ("scalar", 2.0, 1.0),
            ("vector", [0.5, 1.0, 2.0], [1.5, 2.0, 3.0]),
            (
                "matrix",
                [[0.5, 1.0, 2.0], [1.5, 2.0, 3.0]],
                [[1.5, 2.5, 3.5], [0.5, 1.0, 1.5]],
            ),
        ]
    )
    def test_mean_variance(self, name, concentration, rate):
        dist = self.dist_cls(
            concentration=jnp.asarray(concentration), rate=jnp.asarray(rate)
        )
        concentration_arr = np.asarray(concentration)
        rate_arr = np.asarray(rate)
        expected_mean = concentration_arr / rate_arr
        expected_var = concentration_arr / np.square(rate_arr)
        np.testing.assert_allclose(dist.mean(), expected_mean, rtol=1e-6)
        np.testing.assert_allclose(dist.variance(), expected_var, rtol=1e-6)

    @parameterized.expand(
        [
            ("scalar", 2.0, 1.0),
            ("vector", [0.5, 1.0, 2.0], [1.5, 2.0, 3.0]),
        ]
    )
    def test_stddev(self, name, concentration, rate):
        dist = self.dist_cls(
            concentration=jnp.asarray(concentration), rate=jnp.asarray(rate)
        )
        concentration_arr = np.asarray(concentration)
        rate_arr = np.asarray(rate)
        expected_stddev = np.sqrt(concentration_arr) / rate_arr
        np.testing.assert_allclose(dist.stddev(), expected_stddev, rtol=1e-6)

    @parameterized.expand(
        [
            (
                "1d shape",
                [0.5, 1.2, 2.0],
                [1.5, 1.0, 0.5],
                (3,),
            ),
            (
                "2d shape",
                [[0.5, 1.0], [2.0, 3.0]],
                [[1.0, 2.0], [0.5, 1.5]],
                (2, 2),
            ),
        ]
    )
    def test_sample_shape(self, name, concentration, rate, sample_shape):
        dist = self.dist_cls(
            concentration=jnp.asarray(concentration), rate=jnp.asarray(rate)
        )
        samples = dist.sample(self.key)
        self.assertEqual(samples.shape, sample_shape)

    def test_sample_values(self):
        concentration = np.array([1.0, 2.0, 5.0])
        rate = np.array([0.5, 2.0, 1.0])
        dist = self.dist_cls(
            concentration=jnp.array(concentration), rate=jnp.array(rate)
        )
        n_samples = 100000
        keys = jax.random.split(self.key, n_samples)
        sample_fn = jax.jit(jax.vmap(lambda k: dist.sample(k)))
        samples = sample_fn(keys)
        self.assertEqual(samples.shape, (n_samples,) + concentration.shape)
        np.testing.assert_allclose(
            np.mean(np.asarray(samples), axis=0), dist.mean(), rtol=2e-2, atol=1e-2
        )
        np.testing.assert_allclose(
            np.std(np.asarray(samples), axis=0),
            np.sqrt(dist.variance()),
            rtol=5e-2,
            atol=1e-2,
        )

    @parameterized.expand(
        [
            ("float16", jnp.float16),
            ("float32", jnp.float32),
        ]
    )
    def test_sample_dtype(self, name, dtype):
        dist = self.dist_cls(
            concentration=jnp.array(self.concentration, dtype=dtype),
            rate=jnp.array(self.rate, dtype=dtype),
        )
        samples = dist.sample(self.key)
        self.assertEqual(samples.dtype, dtype)

    def test_median_not_implemented(self):
        dist = self.dist_cls(concentration=jnp.array(2.0), rate=jnp.array(1.0))
        with self.assertRaises(NotImplementedError):
            dist.median()

    @parameterized.expand(
        [
            (
                "vector params",
                [1.0, 2.0, 3.0],
                [2.0, 1.0, 1.5],
                [0.5, 1.0, 2.0],
            ),
        ]
    )
    def test_cdf_logcdf_shapes(self, name, concentration, rate, values):
        dist = self.dist_cls(
            concentration=jnp.asarray(concentration), rate=jnp.asarray(rate)
        )
        x = jnp.asarray(values)
        self.assertEqual(dist.cdf(x).shape, dist.concentration.shape)
        self.assertEqual(dist.log_cdf(x).shape, dist.concentration.shape)

    def test_mode_edge_cases(self):
        # Mode is (concentration - 1) / rate when concentration >= 1
        dist1 = self.dist_cls(concentration=2.0, rate=1.0)
        np.testing.assert_allclose(dist1.mode(), 1.0, rtol=1e-6)
        dist2 = self.dist_cls(concentration=3.0, rate=2.0)
        np.testing.assert_allclose(dist2.mode(), 1.0, rtol=1e-6)
        # Mode is NaN when concentration < 1
        dist3 = self.dist_cls(concentration=0.5, rate=1.0)
        self.assertTrue(jnp.isnan(dist3.mode()))

    def test_kl_divergence(self):
        dist1 = self.dist_cls(
            concentration=jnp.array([1.0, 2.0]), rate=jnp.array([0.5, 1.0])
        )
        dist2 = self.dist_cls(
            concentration=jnp.array([2.0, 1.0]), rate=jnp.array([1.0, 2.0])
        )
        kl = dist1.kl_divergence(dist2)
        self.assertEqual(kl.shape, dist1.concentration.shape)
        self.assertTrue(np.all(np.asarray(kl) >= 0))

    def test_entropy(self):
        concentration = np.array([1.0, 2.0, 5.0])
        rate = np.array([0.5, 1.0, 2.0])
        dist = self.dist_cls(
            concentration=jnp.array(concentration), rate=jnp.array(rate)
        )
        entropy = dist.entropy()
        self.assertEqual(entropy.shape, concentration.shape)
        # Entropy should be finite for valid parameters
        self.assertTrue(np.all(np.isfinite(np.asarray(entropy))))

    def test_log_prob(self):
        concentration = np.array([1.0, 2.0, 3.0])
        rate = np.array([1.0, 1.0, 1.0])
        dist = self.dist_cls(
            concentration=jnp.array(concentration), rate=jnp.array(rate)
        )
        x = jnp.array([0.5, 1.0, 2.0])
        log_prob = dist.log_prob(x)
        self.assertEqual(log_prob.shape, concentration.shape)
        # Log prob should be finite for valid inputs
        self.assertTrue(np.all(np.isfinite(np.asarray(log_prob))))
