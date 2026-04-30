from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Split


class SplitTest(TestCase):
    def setUp(self):
        # Bijector 1: Split into 3 equal sections
        self.bij_equal = Split(indices_or_sections=3, axis=-1)

        # Bijector 2: Split at specific indices
        # (elements up to index 2, from 2 to 5, and from 5 onward)
        self.bij_indices = Split(indices_or_sections=(2, 5), axis=-1)

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_forward_and_log_det_equal_sections(self, name, dtype):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype)
        y, log_det = self.bij_equal.forward_and_log_det(x)

        self.assertIsInstance(y, tuple)
        self.assertEqual(len(y), 3)
        self.assertion_fn()(y[0], jnp.array([1.0, 2.0], dtype=dtype))
        self.assertion_fn()(y[1], jnp.array([3.0, 4.0], dtype=dtype))
        self.assertion_fn()(y[2], jnp.array([5.0, 6.0], dtype=dtype))

        self.assertEqual(log_det.shape, ())
        self.assertEqual(log_det, 0.0)
        self.assertEqual(log_det.dtype, dtype)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_forward_and_log_det_indices(self, name, dtype):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=dtype)
        y, log_det = self.bij_indices.forward_and_log_det(x)

        self.assertIsInstance(y, tuple)
        self.assertEqual(len(y), 3)
        self.assertion_fn()(y[0], jnp.array([1.0, 2.0], dtype=dtype))  # x[:2]
        self.assertion_fn()(y[1], jnp.array([3.0, 4.0, 5.0], dtype=dtype))  # x[2:5]
        self.assertion_fn()(y[2], jnp.array([6.0, 7.0], dtype=dtype))  # x[5:]

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_inverse_and_log_det(self, name, dtype):
        y = (
            jnp.array([1.0, 2.0], dtype=dtype),
            jnp.array([3.0, 4.0], dtype=dtype),
            jnp.array([5.0, 6.0], dtype=dtype),
        )
        x, log_det = self.bij_equal.inverse_and_log_det(y)

        self.assertion_fn()(x, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype))

        self.assertEqual(log_det.shape, ())
        self.assertEqual(log_det, 0.0)
        self.assertEqual(x.dtype, dtype)
        self.assertEqual(log_det.dtype, dtype)

    def test_jittable(self):
        @eqx.filter_jit
        def f_forward(bij, x):
            return bij.forward_and_log_det(x)

        @eqx.filter_jit
        def f_inverse(bij, y):
            return bij.inverse_and_log_det(y)

        x = jnp.ones((6,))
        y, log_det_fwd = f_forward(self.bij_equal, x)

        self.assertIsInstance(y, tuple)
        self.assertIsInstance(log_det_fwd, jax.Array)

        x_reconstructed, log_det_inv = f_inverse(self.bij_equal, y)
        self.assertIsInstance(x_reconstructed, jax.Array)
        self.assertIsInstance(log_det_inv, jax.Array)

    def test_same_as(self):
        same_bij = Split(indices_or_sections=3, axis=-1)
        diff_bij_1 = Split(indices_or_sections=4, axis=-1)
        diff_bij_2 = Split(indices_or_sections=3, axis=0)

        self.assertTrue(self.bij_equal.same_as(same_bij))
        self.assertFalse(self.bij_equal.same_as(diff_bij_1))
        self.assertFalse(self.bij_equal.same_as(diff_bij_2))

    def test_list_to_tuple_conversion(self):
        # A list should be converted to a tuple upon initialization for hashability
        bij_list = Split(indices_or_sections=[2, 5])
        self.assertIsInstance(bij_list.indices_or_sections, tuple)
        self.assertTrue(self.bij_indices.same_as(bij_list))
