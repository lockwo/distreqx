from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Restructure


class RestructureTest(TestCase):
    def setUp(self):
        # We will test mapping a 3-element list to a dictionary
        self.in_structure = [0, 1, 2]
        self.out_structure = {"a": 0, "b": 2, "c": 1}
        self.bij = Restructure(
            in_structure=self.in_structure, out_structure=self.out_structure
        )

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_invalid_structures(self):
        # Duplicate tokens in the input structure
        with self.assertRaisesRegex(ValueError, "duplicate tokens"):
            Restructure([0, 0, 1], {"a": 0, "b": 1, "c": 2})

        # Duplicate tokens in the output structure
        with self.assertRaisesRegex(ValueError, "duplicate tokens"):
            Restructure([0, 1, 2], {"a": 0, "b": 0, "c": 1})

        # Mismatched token sets
        with self.assertRaisesRegex(ValueError, "incompatible"):
            Restructure([0, 1, 2], {"a": 0, "b": 1, "c": 3})

    def test_invalid_input_trees(self):
        # Passing the wrong structure to forward
        # (e.g., passing the dict instead of the list)
        bad_x = {"a": jnp.ones(2), "b": jnp.ones(2), "c": jnp.ones(2)}
        with self.assertRaisesRegex(ValueError, "does not match the expected"):
            self.bij.forward_and_log_det(bad_x)

        # Passing the wrong structure to inverse
        bad_y = [jnp.ones(2), jnp.ones(2), jnp.ones(2)]
        with self.assertRaisesRegex(ValueError, "does not match the expected"):
            self.bij.inverse_and_log_det(bad_y)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_forward_and_log_det(self, name, dtype):
        x = [
            jnp.array([1.0, 2.0], dtype=dtype),
            jnp.array([[3.0]], dtype=dtype),
            jnp.array(4.0, dtype=dtype),
        ]
        y, log_det = self.bij.forward_and_log_det(x)

        # Verify the structure and values map correctly according to the tokens
        self.assertIsInstance(y, dict)
        self.assertEqual(set(y.keys()), {"a", "b", "c"})
        self.assertion_fn()(y["a"], x[0])  # Token 0
        self.assertion_fn()(y["c"], x[1])  # Token 1
        self.assertion_fn()(y["b"], x[2])  # Token 2

        # log_det must be an unbatched scalar 0.0 of the matching dtype
        self.assertEqual(log_det.shape, ())
        self.assertEqual(log_det, 0.0)

        # Check dtype preservation
        self.assertEqual(y["a"].dtype, dtype)
        self.assertEqual(log_det.dtype, dtype)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_inverse_and_log_det(self, name, dtype):
        y = {
            "a": jnp.array([1.0, 2.0], dtype=dtype),
            "b": jnp.array(4.0, dtype=dtype),
            "c": jnp.array([[3.0]], dtype=dtype),
        }
        x, log_det = self.bij.inverse_and_log_det(y)

        # Verify the structure and values map correctly back to the list
        self.assertIsInstance(x, list)
        self.assertEqual(len(x), 3)
        self.assertion_fn()(x[0], y["a"])  # Token 0
        self.assertion_fn()(x[1], y["c"])  # Token 1
        self.assertion_fn()(x[2], y["b"])  # Token 2

        self.assertEqual(log_det.shape, ())
        self.assertEqual(log_det, 0.0)
        self.assertEqual(x[0].dtype, dtype)
        self.assertEqual(log_det.dtype, dtype)

    def test_jittable(self):
        @eqx.filter_jit
        def f_forward(bij, x):
            return bij.forward_and_log_det(x)

        @eqx.filter_jit
        def f_inverse(bij, y):
            return bij.inverse_and_log_det(y)

        x = [jnp.ones(1), jnp.ones(2), jnp.ones(3)]
        y, log_det_fwd = f_forward(self.bij, x)

        self.assertIsInstance(y, dict)
        self.assertIsInstance(log_det_fwd, jax.Array)

        x_reconstructed, log_det_inv = f_inverse(self.bij, y)
        self.assertIsInstance(x_reconstructed, list)
        self.assertIsInstance(log_det_inv, jax.Array)

    def test_same_as(self):
        same_bij = Restructure(
            in_structure=[0, 1, 2], out_structure={"a": 0, "b": 2, "c": 1}
        )
        diff_bij = Restructure(
            in_structure=[0, 1, 2], out_structure={"a": 1, "b": 2, "c": 0}
        )

        self.assertTrue(self.bij.same_as(same_bij))
        self.assertFalse(self.bij.same_as(diff_bij))
