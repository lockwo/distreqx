from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Lambda


class LambdaTest(TestCase):
    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_raises_error_on_missing_functions(self):
        # Should raise an error if neither forward nor inverse is given
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            Lambda()

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_forward_and_inverse_full_spec(self, name, dtype):
        # Test case where all functions are explicitly provided (no derivation needed)
        bij = Lambda(
            forward=lambda x: 3.0 * x,
            inverse=lambda y: y / 3.0,
            forward_log_det_jacobian=lambda x: jnp.full_like(x, jnp.log(3.0)),
            inverse_log_det_jacobian=lambda y: jnp.full_like(y, jnp.log(1.0 / 3.0)),
            is_constant_jacobian=True
        )
        
        x = jnp.array([1.0, -2.0, 3.0], dtype=dtype)
        y, fwd_log_det = bij.forward_and_log_det(x)
        
        self.assertion_fn()(y, x * 3.0)
        self.assertion_fn()(fwd_log_det, jnp.full_like(x, jnp.log(3.0)))
        self.assertEqual(y.dtype, dtype)

        x_rec, inv_log_det = bij.inverse_and_log_det(y)
        self.assertion_fn()(x_rec, x)
        self.assertion_fn()(inv_log_det, jnp.full_like(x, jnp.log(1.0 / 3.0)))

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_automatic_derivation_from_forward(self, name, dtype):
        # Providing ONLY forward. The rest should be derived by `transformations`
        # Note: This test will fail if `distreqx._utils.transformations` is not yet implemented.
        bij = Lambda(forward=jnp.exp)
        
        x = jnp.array([1.0, 0.0, -1.0], dtype=dtype)
        
        # Test forward pass
        y, fwd_log_det = bij.forward_and_log_det(x)
        self.assertion_fn()(y, jnp.exp(x))
        # log|det J(exp(x))| = x
        self.assertion_fn()(fwd_log_det, x)

        # Test automatically derived inverse
        x_rec, inv_log_det = bij.inverse_and_log_det(y)
        self.assertion_fn()(x_rec, x)
        # log|det J(log(y))| = log(1/y) = -log(y) = -x
        self.assertion_fn()(inv_log_det, -x)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_automatic_derivation_from_inverse(self, name, dtype):
        # Providing ONLY inverse. 
        bij = Lambda(inverse=jnp.log)
        
        y = jnp.array([1.0, jnp.e, jnp.exp(2.0)], dtype=dtype)
        
        # Test inverse pass
        x_rec, inv_log_det = bij.inverse_and_log_det(y)
        self.assertion_fn()(x_rec, jnp.log(y))
        # log|det J(log(y))| = -log(y)
        self.assertion_fn()(inv_log_det, -jnp.log(y))

        # Test automatically derived forward
        y_rec, fwd_log_det = bij.forward_and_log_det(x_rec)
        self.assertion_fn()(y_rec, y)
        self.assertion_fn()(fwd_log_det, x_rec)

    def test_jittable(self):
        bij = Lambda(
            forward=lambda x: 2.0 * x,
            inverse=lambda y: y / 2.0,
            forward_log_det_jacobian=lambda x: jnp.full_like(x, jnp.log(2.0)),
            inverse_log_det_jacobian=lambda y: jnp.full_like(y, jnp.log(0.5)),
        )

        @eqx.filter_jit
        def f(b, x):
            return b.forward_and_log_det(x)

        x = jnp.array([1.0, 2.0])
        y, log_det = f(bij, x)
        self.assertIsInstance(y, jax.Array)
        self.assertIsInstance(log_det, jax.Array)