from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Reshape


class ReshapeTest(TestCase):
    def setUp(self):
        self.bij = Reshape(in_shape=(2, 3), out_shape=(6,))

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_invalid_shapes(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            Reshape(in_shape=(2, 3), out_shape=(5,))

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_forward_and_log_det(self, name, dtype):
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        y, log_det = self.bij.forward_and_log_det(x)

        self.assertion_fn()(y, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype))
        self.assertEqual(y.shape, (6,))

        # log_det must be an unbatched scalar 0.0 of the matching dtype
        self.assertEqual(log_det.shape, ())
        self.assertEqual(log_det, 0.0)

        self.assertEqual(y.dtype, dtype)
        self.assertEqual(log_det.dtype, dtype)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_inverse_and_log_det(self, name, dtype):
        y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype)
        x, log_det = self.bij.inverse_and_log_det(y)

        self.assertion_fn()(
            x, jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        )
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(log_det, 0.0)
        self.assertEqual(x.dtype, dtype)
        self.assertEqual(log_det.dtype, dtype)

    def test_jittable(self):
        @eqx.filter_jit
        def f(bij, x):
            return bij.forward_and_log_det(x)

        x = jnp.ones((2, 3))
        y, log_det = f(self.bij, x)
        self.assertIsInstance(y, jax.Array)
        self.assertIsInstance(log_det, jax.Array)

    def test_same_as(self):
        same_bij = Reshape(in_shape=(2, 3), out_shape=(6,))
        diff_bij = Reshape(in_shape=(6,), out_shape=(2, 3))

        self.assertTrue(self.bij.same_as(same_bij))
        self.assertFalse(self.bij.same_as(diff_bij))
