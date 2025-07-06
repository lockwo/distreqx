from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import beta as beta_module


class BetaTest(TestCase):
    def setUp(self):
        self.dist_cls = beta_module.Beta
        self.alpha = np.asarray([0.5, 1.0, 2.5, 5.0])
        self.beta = np.asarray([0.5, 2.0, 1.5, 1.0])
        self.key = jax.random.key(0)

    @parameterized.expand(
        [
            ("0d params", (),),
            ("1d params", (4,)),
            ("2d params", (3, 4)),
        ]
    )
    def test_properties(self, name, shape):
        rng = np.random.default_rng(0)
        alpha = rng.uniform(0.1, 5.0, size=shape)
        beta = rng.uniform(0.1, 5.0, size=shape)
        dist = self.dist_cls(alpha=jnp.array(alpha), beta=jnp.array(beta))
        np.testing.assert_allclose(dist.alpha, alpha, rtol=1e-6)
        np.testing.assert_allclose(dist.beta, beta, rtol=1e-6)

    @parameterized.expand(
        [
            ("scalar", 0.7, 1.3),
            ("vector", [0.5, 1.0, 2.0], [1.5, 2.0, 3.0]),
            (
                "matrix",
                [[0.5, 1.0, 2.0], [1.5, 2.0, 3.0]],
                [[1.5, 2.5, 3.5], [0.5, 1.0, 1.5]],
            ),
        ]
    )
    def test_mean_variance(self, name, alpha, beta):
        dist = self.dist_cls(alpha=jnp.asarray(alpha), beta=jnp.asarray(beta))
        alpha_arr = np.asarray(alpha)
        beta_arr = np.asarray(beta)
        expected_mean = alpha_arr / (alpha_arr + beta_arr)
        expected_var = alpha_arr * beta_arr / (
            np.square(alpha_arr + beta_arr) * (alpha_arr + beta_arr + 1.0)
        )
        np.testing.assert_allclose(dist.mean(), expected_mean, rtol=1e-6)
        np.testing.assert_allclose(dist.variance(), expected_var, rtol=1e-6)

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
    def test_sample_shape(self, name, alpha, beta, sample_shape):
        dist = self.dist_cls(alpha=jnp.asarray(alpha), beta=jnp.asarray(beta))
        samples = dist.sample(self.key)
        self.assertEqual(samples.shape, sample_shape)

    def test_sample_values(self):
        alpha = np.array([0.5, 2.0, 5.0])
        beta = np.array([0.5, 2.0, 1.0])
        dist = self.dist_cls(alpha=jnp.array(alpha), beta=jnp.array(beta))
        n_samples = 100000
        keys = jax.random.split(self.key, n_samples)
        sample_fn = jax.jit(jax.vmap(lambda k: dist.sample(k)))
        samples = sample_fn(keys)
        self.assertEqual(samples.shape, (n_samples,) + alpha.shape)
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
            alpha=jnp.array(self.alpha, dtype=dtype),
            beta=jnp.array(self.beta, dtype=dtype),
        )
        samples = dist.sample(self.key)
        self.assertEqual(samples.dtype, dtype)

    def test_median_not_implemented(self):
        dist = self.dist_cls(alpha=jnp.array(0.5), beta=jnp.array(0.5))
        with self.assertRaises(NotImplementedError):
            dist.median()

    @parameterized.expand(
        [
            (
                "vector params",
                [1.0, 2.0, 3.0],
                [2.0, 1.0, 1.5],
                [0.2, 0.5, 0.8],
            ),
        ]
    )
    def test_cdf_logcdf_shapes(self, name, alpha, beta, values):
        dist = self.dist_cls(alpha=jnp.asarray(alpha), beta=jnp.asarray(beta))
        x = jnp.asarray(values)
        self.assertEqual(dist.cdf(x).shape, dist.alpha.shape)
        self.assertEqual(dist.log_cdf(x).shape, dist.alpha.shape)

    def test_mode_edge_cases(self):
        dist1 = self.dist_cls(alpha=0.5, beta=2.0)
        self.assertEqual(dist1.mode(), 0.0)
        dist2 = self.dist_cls(alpha=2.0, beta=0.5)
        self.assertEqual(dist2.mode(), 1.0)
        dist3 = self.dist_cls(alpha=0.5, beta=0.5)
        self.assertTrue(jnp.isnan(dist3.mode()))

    def test_kl_divergence(self):
        dist1 = self.dist_cls(alpha=jnp.array([0.5, 2.0]), beta=jnp.array([0.5, 1.0]))
        dist2 = self.dist_cls(alpha=jnp.array([1.0, 1.0]), beta=jnp.array([1.0, 2.0]))
        kl = dist1.kl_divergence(dist2)
        self.assertEqual(kl.shape, dist1.alpha.shape)
        self.assertTrue(np.all(np.asarray(kl) >= 0))