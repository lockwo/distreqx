from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Identity, Inverse


class InverseTest(TestCase):
    def setUp(self):
        self.base_bij = Identity()
        self.inv_bij = Inverse(self.base_bij)

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_forward_and_inverse(self, name, dtype):
        x = jnp.array([1.5, -0.5, 3.0], dtype=dtype)

        # Because the base bijector is Identity, inverse acts as an Identity bijector
        y, fwd_log_det = self.inv_bij.forward_and_log_det(x)
        x_rec, inv_log_det = self.inv_bij.inverse_and_log_det(y)

        self.assertion_fn()(y, x)
        self.assertion_fn()(x_rec, x)
        self.assertion_fn()(fwd_log_det, jnp.zeros_like(x))
        self.assertion_fn()(inv_log_det, jnp.zeros_like(x))
        self.assertEqual(y.dtype, dtype)

    def test_jittable(self):
        @eqx.filter_jit
        def f(bij, x):
            return bij.forward_and_log_det(x)

        x = jnp.array([1.0, 2.0])
        y, log_det = f(self.inv_bij, x)
        self.assertIsInstance(y, jax.Array)
        self.assertIsInstance(log_det, jax.Array)

    def test_same_as(self):
        same_bij = Inverse(Identity())
        self.assertTrue(self.inv_bij.same_as(same_bij))
        self.assertFalse(self.inv_bij.same_as(Identity()))

    def test_flags_passed_correctly(self):
        self.assertTrue(self.inv_bij.is_constant_jacobian)
        self.assertTrue(self.inv_bij.is_constant_log_det)