from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Chain, R2ToComplex, ScalarAffine


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

    @parameterized.expand(
        [
            ("float32", jnp.float32, jnp.complex64),
            ("float64", jnp.float64, jnp.complex128),
        ]
    )
    def test_forward_and_log_det(self, name, real_dtype, comp_dtype):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=real_dtype)
        y, log_det = self.bij.forward_and_log_det(x)

        expected_y = jnp.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=comp_dtype)
        self.assertion_fn()(y, expected_y)
        self.assertEqual(y.shape, (2,))

        self.assertEqual(log_det, 0.0)
        self.assertEqual(log_det.dtype, real_dtype)

    @parameterized.expand(
        [
            ("float32", jnp.float32, jnp.complex64),
            ("float64", jnp.float64, jnp.complex128),
        ]
    )
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

    @parameterized.expand(
        [
            ("float32", jnp.float32, jnp.complex64),
            ("float64", jnp.float64, jnp.complex128),
        ]
    )
    def test_forward_inverse_round_trip(self, name, real_dtype, comp_dtype):
        # forward then inverse should recover the original real input exactly,
        # confirming the mapping is a genuine bijection (not just type punning).
        x = jnp.array([[1.0, -2.5], [0.0, 3.0], [-4.0, -4.0]], dtype=real_dtype)
        y, _ = self.bij.forward_and_log_det(x)
        x_rec, _ = self.bij.inverse_and_log_det(y)
        self.assertion_fn()(x_rec, x)
        self.assertEqual(x_rec.dtype, real_dtype)

        # inverse then forward should recover the original complex input.
        z = jnp.array([1.0 - 2.5j, 3.0 + 0.5j], dtype=comp_dtype)
        x2, _ = self.bij.inverse_and_log_det(z)
        z_rec, _ = self.bij.forward_and_log_det(x2)
        self.assertion_fn()(z_rec, z)
        self.assertEqual(z_rec.dtype, comp_dtype)

    def test_chaining(self):
        # Chain a real-valued affine transform ahead of R2ToComplex, to check
        # R2ToComplex composes correctly as part of a larger bijector pipeline.
        affine = ScalarAffine(shift=jnp.array(1.0), scale=jnp.array(2.0))
        chain = Chain([R2ToComplex(), affine])
        x = jnp.array([[1.0, 2.0], [-3.0, 4.5]])

        y, log_det = chain.forward_and_log_det(x)

        affine_out = affine.scale * x + affine.shift
        expected_y = affine_out[..., 0] + 1j * affine_out[..., 1]
        self.assertion_fn()(y, expected_y)
        self.assertion_fn()(log_det, affine.forward_log_det_jacobian(x))

        x_rec, inv_log_det = chain.inverse_and_log_det(y)
        self.assertion_fn()(x_rec, x)
        self.assertion_fn()(inv_log_det, -affine.forward_log_det_jacobian(x))
