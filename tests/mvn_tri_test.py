from unittest import TestCase

import jax


jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import MultivariateNormalTri


class MultivariateNormalTriTest(TestCase):
    def setUp(self):
        self.key = jax.random.key(0)

    def _test_raises_error(self, dist_kwargs):
        with self.assertRaises(ValueError):
            dist = MultivariateNormalTri(**dist_kwargs)
            dist.sample(key=self.key)

    def test_invalid_parameters(self):
        self._test_raises_error(dist_kwargs={"loc": None, "scale_tri": None})
        self._test_raises_error(dist_kwargs={"loc": jnp.array(1.0), "scale_tri": None})
        self._test_raises_error(dist_kwargs={"loc": None, "scale_tri": jnp.array(1.0)})
        self._test_raises_error(
            dist_kwargs={"loc": None, "scale_tri": jnp.array([1.0])}
        )
        self._test_raises_error(
            dist_kwargs={
                "loc": None,
                "scale_tri": jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            }
        )
        self._test_raises_error(
            dist_kwargs={"loc": jnp.zeros((5,)), "scale_tri": jnp.ones((4, 4))}
        )

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_sample_dtype(self, name, dtype):
        dist_params = {
            "loc": jnp.array([0.0, 0.0], dtype),
            "scale_tri": jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype),
            "is_lower": True,
        }
        dist = MultivariateNormalTri(**dist_params)
        samples = dist.sample(key=self.key)
        self.assertEqual(samples.dtype, dist.dtype)
        self.assertEqual(samples.dtype, dtype)

    def test_jittable(self):
        @eqx.filter_jit
        def f(dist):
            return dist.sample(key=jax.random.key(0))

        dist_params = {"loc": jnp.zeros(2), "scale_tri": jnp.eye(2), "is_lower": True}
        dist = MultivariateNormalTri(**dist_params)
        y = f(dist)
        self.assertIsInstance(y, jax.Array)

    def assertion_fn(self, rtol):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)
