from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Exp, Identity


class ExpTest(TestCase):
    def setUp(self):
        self.bij = Exp()

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_forward_and_log_det(self, name, dtype):
        x = jnp.array([-2.5, 0.0, 1.0, 3.5], dtype=dtype)
        y, log_det = self.bij.forward_and_log_det(x)

        self.assertion_fn()(y, jnp.exp(x))
        self.assertion_fn()(log_det, x)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(log_det.dtype, dtype)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_inverse_and_log_det(self, name, dtype):
        # We must use strictly positive numbers for the domain of log(y)
        y = jnp.array([0.1, 1.0, jnp.e, 10.0], dtype=dtype)
        x, log_det = self.bij.inverse_and_log_det(y)

        self.assertion_fn()(x, jnp.log(y))
        self.assertion_fn()(log_det, -jnp.log(y))
        self.assertEqual(x.dtype, dtype)
        self.assertEqual(log_det.dtype, dtype)

    def test_jittable(self):
        @eqx.filter_jit
        def f(bij, x):
            return bij.forward_and_log_det(x)

        x = jnp.array([1.0, 2.0])
        y, log_det = f(self.bij, x)
        self.assertIsInstance(y, jax.Array)
        self.assertIsInstance(log_det, jax.Array)

    def test_same_as(self):
        same_bij = Exp()
        diff_bij = Identity()

        self.assertTrue(self.bij.same_as(same_bij))
        self.assertFalse(self.bij.same_as(diff_bij))
