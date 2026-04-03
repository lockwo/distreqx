from unittest import TestCase

import jax

# Must be set before any JAX arrays are initialized
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import R2ToComplex


class R2ToComplexTest(TestCase):
    def setUp(self):
        self.bij = R2ToComplex()

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_invalid_shapes_and_types(self):
        with self.assertRaisesRegex(ValueError, "last dimension"):
            self.bij.forward_and_log_det(jnp.array([1.0, 2.0, 3.0]))
            
        with self.assertRaisesRegex(ValueError, "complex array"):
            self.bij.inverse_and_log_det(jnp.array([1.0, 2.0]))

    @parameterized.expand([
        ("float32", jnp.float32, jnp.complex64), 
        ("float64", jnp.float64, jnp.complex128)
    ])
    def test_forward_and_log_det(self, name, real_dtype, comp_dtype):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=real_dtype)
        y, log_det = self.bij.forward_and_log_det(x)

        expected_y = jnp.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=comp_dtype)
        self.assertion_fn()(y, expected_y)
        self.assertEqual(y.shape, (2,))
        
        self.assertEqual(log_det, 0.0)
        self.assertEqual(log_det.dtype, real_dtype)

    @parameterized.expand([
        ("float32", jnp.float32, jnp.complex64), 
        ("float64", jnp.float64, jnp.complex128)
    ])
    def test_inverse_and_log_det(self, name, real_dtype, comp_dtype):
        y = jnp.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=comp_dtype)
        x, log_det = self.bij.inverse_and_log_det(y)

        expected_x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=real_dtype)
        self.assertion_fn()(x, expected_x)
        self.assertEqual(x.shape, (2, 2))
        
        self.assertEqual(log_det, 0.0)
        self.assertEqual(log_det.dtype, real_dtype)

    def test_jittable(self):
        @eqx.filter_jit
        def f(bij, x):
            return bij.forward_and_log_det(x)

        x = jnp.array([1.0, 2.0])
        y, log_det = f(self.bij, x)
        self.assertIsInstance(y, jax.Array)
        self.assertIsInstance(log_det, jax.Array)

    def test_same_as(self):
        same_bij = R2ToComplex()
        self.assertTrue(self.bij.same_as(same_bij))