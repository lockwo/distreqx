"""Tests for `triangular_linear.py`."""

from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from distreqx.bijectors import Tanh, TriangularLinear


class TriangularLinearTest(TestCase):
    def test_static_properties(self):
        bij = TriangularLinear(matrix=jnp.eye(4))
        self.assertTrue(bij.is_constant_jacobian)
        self.assertTrue(bij.is_constant_log_det)

    def test_properties(self):
        for is_lower in [True, False]:
            bij = TriangularLinear(matrix=jnp.ones((4, 4)), is_lower=is_lower)
            self.assertEqual(bij.event_dims, 4)
            self.assertEqual(bij.matrix.shape, (4, 4))
            tri = np.tril if is_lower else np.triu
            np.testing.assert_allclose(bij.matrix, tri(np.ones((4, 4))), atol=1e-6)
            self.assertEqual(bij.is_lower, is_lower)

    def test_raises_with_invalid_parameters(self):
        with self.assertRaises(ValueError):
            TriangularLinear(matrix=jnp.zeros(()))
        with self.assertRaises(ValueError):
            TriangularLinear(matrix=jnp.zeros((4,)))
        with self.assertRaises(ValueError):
            TriangularLinear(matrix=jnp.zeros((3, 4)))

    def test_parameters(self):
        prng = jax.random.PRNGKey(42)
        prng = jax.random.split(prng, 2)
        matrix = jax.random.uniform(prng[0], (4, 4)) + jnp.eye(4)
        bijector = TriangularLinear(matrix)

        x = jax.random.normal(prng[1], (4,))
        y, logdet_fwd = bijector.forward_and_log_det(x)
        z, logdet_inv = bijector.inverse_and_log_det(x)

        self.assertEqual(y.shape, (4,))
        self.assertEqual(z.shape, (4,))
        self.assertEqual(logdet_fwd.shape, ())
        self.assertEqual(logdet_inv.shape, ())

    def test_identity_initialization(self):
        for is_lower in [True, False]:
            bijector = TriangularLinear(matrix=jnp.eye(4), is_lower=is_lower)
            prng = jax.random.PRNGKey(42)
            x = jax.random.normal(prng, (4,))

            # Forward methods.
            y, logdet = bijector.forward_and_log_det(x)
            np.testing.assert_array_equal(y, x)
            np.testing.assert_array_equal(logdet, jnp.zeros(1))

            # Inverse methods.
            x_rec, logdet = bijector.inverse_and_log_det(y)
            np.testing.assert_array_equal(x_rec, y)
            np.testing.assert_array_equal(logdet, jnp.zeros(1))

    def test_inverse_methods(self):
        for is_lower in [True, False]:
            prng = jax.random.PRNGKey(42)
            prng = jax.random.split(prng, 2)
            matrix = jax.random.uniform(prng[0], (4, 4)) + jnp.eye(4)
            bijector = TriangularLinear(matrix=matrix, is_lower=is_lower)
            x = jax.random.normal(prng[1], (4,))
            y, logdet_fwd = bijector.forward_and_log_det(x)
            x_rec, logdet_inv = bijector.inverse_and_log_det(y)
            np.testing.assert_allclose(x_rec, x, atol=1e-6)
            np.testing.assert_allclose(logdet_fwd, -logdet_inv, atol=1e-6)

    def test_forward_jacobian_det(self):
        for is_lower in [True, False]:
            prng = jax.random.PRNGKey(42)
            prng = jax.random.split(prng, 3)
            matrix = jax.random.uniform(prng[0], (4, 4)) + jnp.eye(4)
            bijector = TriangularLinear(matrix, is_lower)

            batched_x = jax.random.normal(prng[1], (10, 4))
            single_x = jax.random.normal(prng[2], (4,))
            batched_logdet = eqx.filter_vmap(bijector.forward_log_det_jacobian)(
                batched_x
            )

            jacobian_fn = jax.jacfwd(bijector.forward)
            logdet_numerical = jnp.linalg.slogdet(jacobian_fn(single_x))[1]
            for logdet in batched_logdet:
                np.testing.assert_allclose(logdet, logdet_numerical, atol=5e-3)

    def test_inverse_jacobian_det(self):
        for is_lower in [True, False]:
            prng = jax.random.PRNGKey(42)
            prng = jax.random.split(prng, 3)
            matrix = jax.random.uniform(prng[0], (4, 4)) + jnp.eye(4)
            bijector = TriangularLinear(matrix, is_lower)

            batched_y = jax.random.normal(prng[1], (10, 4))
            single_y = jax.random.normal(prng[2], (4,))
            batched_logdet = eqx.filter_vmap(bijector.inverse_log_det_jacobian)(
                batched_y
            )

            jacobian_fn = jax.jacfwd(bijector.inverse)
            logdet_numerical = jnp.linalg.slogdet(jacobian_fn(single_y))[1]
            for logdet in batched_logdet:
                np.testing.assert_allclose(logdet, logdet_numerical, atol=5e-5)

    def test_jittable(self):
        @eqx.filter_jit
        def f(x, b):
            return b.forward(x)

        bij = TriangularLinear(matrix=jnp.eye(4))
        x = jnp.zeros((4,))
        f(x, bij)

    def test_same_as_itself(self):
        bij = TriangularLinear(matrix=jnp.eye(4))
        self.assertTrue(bij.same_as(bij))

    def test_not_same_as_others(self):
        bij = TriangularLinear(matrix=jnp.eye(4))
        other = TriangularLinear(matrix=jnp.ones((4, 4)))
        self.assertFalse(bij.same_as(other))
        self.assertFalse(bij.same_as(Tanh()))
