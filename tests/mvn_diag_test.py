from unittest import TestCase

import jax


jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import MultivariateNormalDiag


class MultivariateNormalDiagTest(TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(0)

    def _test_raises_error(self, dist_kwargs):
        with self.assertRaises(ValueError):
            dist = MultivariateNormalDiag(**dist_kwargs)
            dist.sample(key=self.key)

    def test_invalid_parameters(self):
        self._test_raises_error(dist_kwargs={"loc": None, "scale_diag": None})
        self._test_raises_error(dist_kwargs={"loc": None, "scale_diag": jnp.array(1.0)})
        self._test_raises_error(dist_kwargs={"loc": jnp.array(1.0), "scale_diag": None})
        self._test_raises_error(
            dist_kwargs={"loc": jnp.zeros((3, 5)), "scale_diag": jnp.ones((3, 4))}
        )

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_sample_dtype(self, name, dtype):
        dist_params = {
            "loc": jnp.array([0.0, 0.0], dtype),
            "scale_diag": jnp.array([1.0, 1.0], dtype),
        }
        dist = MultivariateNormalDiag(**dist_params)
        samples = dist.sample(key=self.key)
        self.assertEqual(samples.dtype, dist.dtype)
        self.assertEqual(samples.dtype, dtype)

    def test_median(self):
        dist_params = {
            "loc": jnp.array([0.3, -0.1, 0.0]),
            "scale_diag": jnp.array([0.1, 1.4, 0.5]),
        }
        dist = MultivariateNormalDiag(**dist_params)
        np.testing.assert_allclose(dist.median(), dist.mean(), rtol=1e-3)

    def test_jittable(self):
        @eqx.filter_jit
        def f(dist):
            return dist.sample(key=jax.random.PRNGKey(0))

        dist_params = {"loc": jnp.zeros(2), "scale_diag": jnp.ones(2)}
        dist = MultivariateNormalDiag(**dist_params)
        y = f(dist)
        self.assertIsInstance(y, jax.Array)

    def assertion_fn(self, rtol):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)
