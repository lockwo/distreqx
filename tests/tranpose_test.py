from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Transpose


class TransposeTest(TestCase):
    def setUp(self):
        # A simple 2D transpose (swapping axes 0 and 1)
        self.bij = Transpose(permutation=(1, 0))

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_invalid_permutation(self):
        with self.assertRaisesRegex(ValueError, "valid combination"):
            # Invalid because it skips axis 1
            Transpose(permutation=(0, 2))

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_forward_and_log_det(self, name, dtype):
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        y, log_det = self.bij.forward_and_log_det(x)

        expected_y = jnp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=dtype)

        self.assertion_fn()(y, expected_y)
        self.assertEqual(y.shape, (3, 2))

        self.assertEqual(log_det.shape, ())
        self.assertEqual(log_det, 0.0)

        self.assertEqual(y.dtype, dtype)
        self.assertEqual(log_det.dtype, dtype)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_inverse_and_log_det(self, name, dtype):
        y = jnp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=dtype)
        x, log_det = self.bij.inverse_and_log_det(y)

        expected_x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)

        self.assertion_fn()(x, expected_x)
        self.assertEqual(x.shape, (2, 3))

        self.assertEqual(log_det.shape, ())
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
        same_bij = Transpose(permutation=(1, 0))
        diff_bij = Transpose(permutation=(0, 1))

        self.assertTrue(self.bij.same_as(same_bij))
        self.assertFalse(self.bij.same_as(diff_bij))
