from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Exp, Permute


class PermuteTest(TestCase):
    def setUp(self):
        self.bij = Permute(permutation=[2, 0, 1])

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_invalid_permutation(self):
        with self.assertRaisesRegex(ValueError, "1D array"):
            Permute([[0, 1], [2, 3]])
            
        with self.assertRaisesRegex(ValueError, "exactly once"):
            Permute([0, 1, 1])  # Duplicate 1, missing 2
            
        with self.assertRaisesRegex(ValueError, "exactly once"):
            Permute([0, 2, 3])  # Missing 1

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_forward_and_inverse(self, name, dtype):
        # Original: [A, B, C] -> Permuted: [C, A, B]
        x = jnp.array([10.0, 20.0, 30.0], dtype=dtype)
        
        y, log_det_fwd = self.bij.forward_and_log_det(x)
        self.assertion_fn()(y, jnp.array([30.0, 10.0, 20.0], dtype=dtype))
        self.assertEqual(log_det_fwd, 0.0)

        x_rec, log_det_inv = self.bij.inverse_and_log_det(y)
        self.assertion_fn()(x_rec, x)
        self.assertEqual(log_det_inv, 0.0)

    def test_jittable(self):
        @eqx.filter_jit
        def f(bij, x):
            return bij.forward_and_log_det(x)

        x = jnp.array([1.0, 2.0, 3.0])
        y, logdet = f(self.bij, x)
        self.assertIsInstance(y, jax.Array)
        self.assertIsInstance(logdet, jax.Array)
