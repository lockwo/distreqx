"""Tests for `sigmoid.py`."""

from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Sigmoid, Tanh


RTOL = 1e-5


class SigmoidTest(TestCase):
    def setUp(self):
        self.seed = jax.random.PRNGKey(1234)

    def test_properties(self):
        bijector = Sigmoid()
        self.assertFalse(bijector.is_constant_jacobian)
        self.assertFalse(bijector.is_constant_log_det)

    @parameterized.expand(
        [("x_shape", (2,)), ("x_shape", (2, 3)), ("x_shape", (2, 3, 4))]
    )
    def test_forward_shapes(self, name, x_shape):
        x = jnp.zeros(shape=x_shape)
        bijector = Sigmoid()
        y1 = bijector.forward(x)
        logdet1 = bijector.forward_log_det_jacobian(x)
        y2, logdet2 = bijector.forward_and_log_det(x)
        self.assertEqual(y1.shape, x_shape)
        self.assertEqual(y2.shape, x_shape)
        self.assertEqual(logdet1.shape, x_shape)
        self.assertEqual(logdet2.shape, x_shape)

    @parameterized.expand(
        [("y_shape", (2,)), ("y_shape", (2, 3)), ("y_shape", (2, 3, 4))]
    )
    def test_inverse_shapes(self, name, y_shape):
        y = jnp.zeros(shape=y_shape)
        bijector = Sigmoid()
        x1 = bijector.inverse(y)
        logdet1 = bijector.inverse_log_det_jacobian(y)
        x2, logdet2 = bijector.inverse_and_log_det(y)
        self.assertEqual(x1.shape, y_shape)
        self.assertEqual(x2.shape, y_shape)
        self.assertEqual(logdet1.shape, y_shape)
        self.assertEqual(logdet2.shape, y_shape)

    def test_forward(self):
        prng = jax.random.PRNGKey(42)
        x = jax.random.normal(prng, (100,))
        bijector = Sigmoid()
        y = bijector.forward(x)
        np.testing.assert_allclose(y, jax.nn.sigmoid(x), rtol=RTOL)

    def test_forward_log_det_jacobian(self):
        prng = jax.random.PRNGKey(42)
        x = jax.random.normal(prng, (100,))
        bijector = Sigmoid()
        fwd_logdet = bijector.forward_log_det_jacobian(x)
        actual = jnp.log(jax.vmap(jax.grad(bijector.forward))(x))
        np.testing.assert_allclose(fwd_logdet, actual, rtol=1e-3)

    def test_forward_and_log_det(self):
        prng = jax.random.PRNGKey(42)
        x = jax.random.normal(prng, (100,))
        bijector = Sigmoid()
        y1 = bijector.forward(x)
        logdet1 = bijector.forward_log_det_jacobian(x)
        y2, logdet2 = bijector.forward_and_log_det(x)
        np.testing.assert_allclose(y1, y2, rtol=RTOL)
        np.testing.assert_allclose(logdet1, logdet2, rtol=RTOL)

    def test_inverse(self):
        prng = jax.random.PRNGKey(42)
        x = jax.random.normal(prng, (100,))
        bijector = Sigmoid()
        y = bijector.forward(x)
        x_rec = bijector.inverse(y)
        np.testing.assert_allclose(x_rec, x, rtol=1e-3)

    def test_inverse_log_det_jacobian(self):
        prng = jax.random.PRNGKey(42)
        x = jax.random.normal(prng, (100,))
        bijector = Sigmoid()
        y = bijector.forward(x)
        fwd_logdet = bijector.forward_log_det_jacobian(x)
        inv_logdet = bijector.inverse_log_det_jacobian(y)
        np.testing.assert_allclose(inv_logdet, -fwd_logdet, rtol=1e-4)

    def test_inverse_and_log_det(self):
        prng = jax.random.PRNGKey(42)
        y = jax.random.normal(prng, (100,))
        bijector = Sigmoid()
        x1 = bijector.inverse(y)
        logdet1 = bijector.inverse_log_det_jacobian(y)
        x2, logdet2 = bijector.inverse_and_log_det(y)
        np.testing.assert_allclose(x1, x2, rtol=RTOL)
        np.testing.assert_allclose(logdet1, logdet2, rtol=RTOL)

    def test_jittable(self):
        @jax.jit
        def f(x, b):
            return b.forward(x)

        bijector = Sigmoid()
        x = jnp.zeros(())
        y = f(x, bijector)
        self.assertIsInstance(y, jax.Array)

    def test_same_as(self):
        bijector = Sigmoid()
        self.assertTrue(bijector.same_as(bijector))
        self.assertTrue(bijector.same_as(Sigmoid()))
        self.assertFalse(bijector.same_as(Tanh()))
