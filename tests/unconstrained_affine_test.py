"""Tests for `unconstrained_affine.py`."""

from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from distreqx.bijectors import Tanh, UnconstrainedAffine


class UnconstrainedAffineTest(TestCase):
    def test_static_properties(self):
        bij = UnconstrainedAffine(matrix=jnp.eye(4), bias=jnp.zeros((4,)))
        self.assertTrue(bij.is_constant_jacobian)
        self.assertTrue(bij.is_constant_log_det)

    def test_properties(self):
        bijector = UnconstrainedAffine(matrix=jnp.eye(4), bias=jnp.zeros((4,)))
        np.testing.assert_allclose(bijector.matrix, np.eye(4))
        np.testing.assert_allclose(bijector.bias, np.zeros((4,)))

    def test_raises_with_invalid_parameters(self):
        # matrix is 0d
        with self.assertRaises(ValueError):
            UnconstrainedAffine(matrix=jnp.zeros(()), bias=jnp.zeros((4,)))
        # matrix is 1d
        with self.assertRaises(ValueError):
            UnconstrainedAffine(matrix=jnp.zeros((4,)), bias=jnp.zeros((4,)))
        # bias is 0d
        with self.assertRaises(ValueError):
            UnconstrainedAffine(matrix=jnp.zeros((4, 4)), bias=jnp.zeros(()))
        # matrix is not square
        with self.assertRaises(ValueError):
            UnconstrainedAffine(matrix=jnp.zeros((3, 4)), bias=jnp.zeros((4,)))
        # matrix and bias shapes do not agree
        with self.assertRaises(ValueError):
            UnconstrainedAffine(matrix=jnp.zeros((4, 4)), bias=jnp.zeros((3,)))

    def test_shapes(self):
        prng = jax.random.split(jax.random.key(42), 3)
        matrix = jax.random.uniform(prng[0], (4, 4)) + jnp.eye(4)
        bias = jax.random.normal(prng[1], (4,))
        bijector = UnconstrainedAffine(matrix, bias)

        x = jax.random.normal(prng[2], (4,))
        y, logdet_fwd = bijector.forward_and_log_det(x)
        z, logdet_inv = bijector.inverse_and_log_det(x)

        self.assertEqual(y.shape, (4,))
        self.assertEqual(z.shape, (4,))
        self.assertEqual(logdet_fwd.shape, ())
        self.assertEqual(logdet_inv.shape, ())

    def test_identity_initialization(self):
        bijector = UnconstrainedAffine(matrix=jnp.eye(4), bias=jnp.zeros((4,)))
        prng = jax.random.key(42)
        x = jax.random.normal(prng, (4,))

        y, logdet = bijector.forward_and_log_det(x)
        np.testing.assert_allclose(y, x, atol=1e-6)
        np.testing.assert_allclose(logdet, 0.0, atol=1e-6)

        x_rec, logdet = bijector.inverse_and_log_det(y)
        np.testing.assert_allclose(x_rec, y, atol=1e-6)
        np.testing.assert_allclose(logdet, 0.0, atol=1e-6)

    def test_inverse_methods(self):
        prng = jax.random.split(jax.random.key(42), 3)
        matrix = jax.random.uniform(prng[0], (4, 4)) + jnp.eye(4)
        bias = jax.random.normal(prng[1], (4,))
        bijector = UnconstrainedAffine(matrix, bias)
        x = jax.random.normal(prng[2], (4,))
        y, logdet_fwd = bijector.forward_and_log_det(x)
        x_rec, logdet_inv = bijector.inverse_and_log_det(y)
        np.testing.assert_allclose(x_rec, x, atol=1e-5)
        np.testing.assert_allclose(logdet_fwd, -logdet_inv, atol=1e-6)

    def test_forward_jacobian_det(self):
        prng = jax.random.split(jax.random.key(42), 4)
        matrix = jax.random.uniform(prng[0], (4, 4)) + jnp.eye(4)
        bias = jax.random.normal(prng[1], (4,))
        bijector = UnconstrainedAffine(matrix, bias)

        batched_x = jax.random.normal(prng[2], (10, 4))
        single_x = jax.random.normal(prng[3], (4,))
        batched_logdet = eqx.filter_vmap(bijector.forward_log_det_jacobian)(batched_x)

        jacobian_fn = jax.jacfwd(bijector.forward)
        logdet_numerical = jnp.linalg.slogdet(jacobian_fn(single_x))[1]
        for logdet in batched_logdet:
            np.testing.assert_allclose(logdet, logdet_numerical, atol=5e-3)

    def test_inverse_jacobian_det(self):
        prng = jax.random.split(jax.random.key(42), 4)
        matrix = jax.random.uniform(prng[0], (4, 4)) + jnp.eye(4)
        bias = jax.random.normal(prng[1], (4,))
        bijector = UnconstrainedAffine(matrix, bias)

        batched_y = jax.random.normal(prng[2], (10, 4))
        single_y = jax.random.normal(prng[3], (4,))
        batched_logdet = eqx.filter_vmap(bijector.inverse_log_det_jacobian)(batched_y)

        jacobian_fn = jax.jacfwd(bijector.inverse)
        logdet_numerical = jnp.linalg.slogdet(jacobian_fn(single_y))[1]
        for logdet in batched_logdet:
            np.testing.assert_allclose(logdet, logdet_numerical, atol=5e-5)

    def test_jittable(self):
        @eqx.filter_jit
        def f(x, b):
            return b.forward(x)

        bijector = UnconstrainedAffine(matrix=jnp.eye(4), bias=jnp.zeros((4,)))
        x = jnp.zeros((4,))
        f(x, bijector)

    def test_same_as_itself(self):
        bij = UnconstrainedAffine(matrix=jnp.eye(4), bias=jnp.zeros((4,)))
        self.assertTrue(bij.same_as(bij))

    def test_not_same_as_others(self):
        bij = UnconstrainedAffine(matrix=jnp.eye(4), bias=jnp.zeros((4,)))
        other = UnconstrainedAffine(matrix=jnp.eye(4), bias=jnp.ones((4,)))
        self.assertFalse(bij.same_as(other))
        self.assertFalse(bij.same_as(Tanh()))
