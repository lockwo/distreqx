"""Tests for `shift.py`."""

from parameterized import parameterized  # type: ignore
from unittest import TestCase
from distreqx.bijectors import Tanh, Shift
import jax
import jax.numpy as jnp
import numpy as np


class ShiftTest(TestCase):

    def test_jacobian_is_constant_property(self):
        bijector = Shift(jnp.ones((4,)))
        self.assertTrue(bijector.is_constant_jacobian)
        self.assertTrue(bijector.is_constant_log_det)

    def test_properties(self):
        bijector = Shift(jnp.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(bijector.shift, np.array([1.0, 2.0, 3.0]))

    @parameterized.expand(
        [
            ("no shape", {"param_shape": ()}),
            ("no shape", {"param_shape": ()}),
            ("3D shape", {"param_shape": (3,)}),
            ("2x3 shape", {"param_shape": (2, 3)}),
        ]
    )
    def test_forward_methods(self, name, kwargs):
        param_shape = kwargs["param_shape"]
        bijector = Shift(jnp.ones(param_shape))
        prng = jax.random.PRNGKey(42)
        x = jax.random.normal(prng, param_shape)
        output_shape = param_shape
        y1 = bijector.forward(x)
        logdet1 = bijector.forward_log_det_jacobian(x)
        y2, logdet2 = bijector.forward_and_log_det(x)
        self.assertEqual(y1.shape, output_shape)
        self.assertEqual(y2.shape, output_shape)
        self.assertEqual(logdet1.shape, output_shape)
        self.assertEqual(logdet2.shape, output_shape)
        np.testing.assert_allclose(y1, x + 1.0, 1e-6)
        np.testing.assert_allclose(y2, x + 1.0, 1e-6)
        np.testing.assert_allclose(logdet1, 0.0, 1e-6)
        np.testing.assert_allclose(logdet2, 0.0, 1e-6)

    @parameterized.expand(
        [
            ("no shape", ()),
            ("3D param", (3,)),
            ("2x3 batch and param", (2, 3)),
        ]
    )
    def test_inverse_methods(self, name, param_shape):
        bijector = Shift(jnp.ones(param_shape))
        prng = jax.random.PRNGKey(42)
        y = jax.random.normal(prng, param_shape)
        output_shape = param_shape
        x1 = bijector.inverse(y)
        logdet1 = bijector.inverse_log_det_jacobian(y)
        x2, logdet2 = bijector.inverse_and_log_det(y)
        self.assertEqual(x1.shape, output_shape)
        self.assertEqual(x2.shape, output_shape)
        self.assertEqual(logdet1.shape, output_shape)
        self.assertEqual(logdet2.shape, output_shape)
        np.testing.assert_allclose(x1, y - 1.0, 1e-6)
        np.testing.assert_allclose(x2, y - 1.0, 1e-6)
        np.testing.assert_allclose(logdet1, 0.0, 1e-6)
        np.testing.assert_allclose(logdet2, 0.0, 1e-6)

    def test_jittable(self):
        @jax.jit
        def f(x, b):
            return b.forward(x)

        bij = Shift(jnp.ones((4,)))
        x = np.zeros((4,))
        z = f(x, bij)
        self.assertIsInstance(z, jnp.ndarray)

    def test_same_as_itself(self):
        bij = Shift(jnp.ones((4,)))
        self.assertTrue(bij.same_as(bij))

    def test_not_same_as_others(self):
        bij = Shift(jnp.ones((4,)))
        other = Shift(jnp.zeros((4,)))
        self.assertFalse(bij.same_as(other))
        self.assertFalse(bij.same_as(Tanh()))

