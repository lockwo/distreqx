from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import MultivariateNormalFullCovariance


class MultivariateNormalFullCovarianceTest(TestCase):
    def setUp(self):
        self.key = jax.random.key(0)

    def _test_raises_error(self, dist_kwargs):
        with self.assertRaises(ValueError):
            MultivariateNormalFullCovariance(**dist_kwargs)

    def test_invalid_parameters(self):
        # Neither specified
        self._test_raises_error(dist_kwargs={"loc": None, "covariance_matrix": None})

        # 0D loc (distreqx expects explicitly 1D for events)
        self._test_raises_error(
            dist_kwargs={"loc": jnp.array(1.0), "covariance_matrix": None}
        )

        # 1D covariance matrix
        self._test_raises_error(
            dist_kwargs={"loc": None, "covariance_matrix": jnp.array([1.0, 1.0])}
        )

        # 3D batched covariance matrix (distreqx users should use vmap instead)
        self._test_raises_error(
            dist_kwargs={
                "loc": None,
                "covariance_matrix": jnp.array([[[1.0, 0.0], [0.0, 1.0]]]),
            }
        )

        # Non-square covariance matrix
        self._test_raises_error(
            dist_kwargs={
                "loc": None,
                "covariance_matrix": jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            }
        )

        # Mismatched event shapes
        self._test_raises_error(
            dist_kwargs={"loc": jnp.zeros((5,)), "covariance_matrix": jnp.eye(4)}
        )

    def test_raw_python_float_raises_value_error_not_attribute_error(self):
        """A raw (uncast) `loc`/`covariance_matrix` should fail shape validation
        cleanly, instead of crashing on a missing `.shape`/`.ndim` attribute."""
        with self.assertRaises(ValueError):
            MultivariateNormalFullCovariance(loc=1.0, covariance_matrix=None)
        with self.assertRaises(ValueError):
            MultivariateNormalFullCovariance(loc=None, covariance_matrix=1.0)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_sample_dtype(self, name, dtype):
        dist_params = {
            "loc": jnp.array([0.0, 0.0], dtype),
            "covariance_matrix": jnp.array([[1.0, 0.5], [0.5, 1.0]], dtype),
        }
        dist = MultivariateNormalFullCovariance(**dist_params)
        samples = dist.sample(key=self.key)
        self.assertEqual(samples.dtype, dist.dtype)
        self.assertEqual(samples.dtype, dtype)

    def test_covariance_and_variance(self):
        cov = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        dist = MultivariateNormalFullCovariance(loc=jnp.zeros(2), covariance_matrix=cov)

        np.testing.assert_allclose(dist.covariance(), cov)
        np.testing.assert_allclose(dist.variance(), jnp.array([2.0, 3.0]))

    def test_jittable(self):
        @eqx.filter_jit
        def f(dist):
            return dist.sample(key=jax.random.key(0))

        dist_params = {
            "loc": jnp.zeros(2),
            "covariance_matrix": jnp.array([[2.0, 0.1], [0.1, 2.0]]),
        }
        dist = MultivariateNormalFullCovariance(**dist_params)

        y = f(dist)
        self.assertIsInstance(y, jax.Array)
        self.assertEqual(y.shape, (2,))
