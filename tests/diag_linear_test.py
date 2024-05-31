"""Tests for `diag_linear.py`."""

from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from distreqx.bijectors import DiagLinear, Tanh


class DiagLinearTest(TestCase):
    def test_static_properties(self):
        bij = DiagLinear(diag=jnp.ones((4,)))
        self.assertTrue(bij.is_constant_jacobian)
        self.assertTrue(bij.is_constant_log_det)
        self.assertEqual(bij.event_dims, 4)

    def test_properties(self):
        bij = DiagLinear(diag=jnp.ones((4,)))
        self.assertEqual(bij.event_dims, 4)
        self.assertEqual(bij.diag.shape, (4,))
        self.assertEqual(bij.matrix.shape, (4, 4))
        np.testing.assert_allclose(bij.diag, 1.0, atol=1e-6)
        np.testing.assert_allclose(bij.matrix, np.eye(4), atol=1e-6)

    def test_raises_with_invalid_parameters(self):
        with self.assertRaises(ValueError):
            DiagLinear(diag=jnp.ones(()))

    def test_parameters(self):
        prng = jax.random.PRNGKey(42)
        prng = jax.random.split(prng, 2)
        diag = jax.random.uniform(prng[0], (4,)) + 0.5
        bij = DiagLinear(diag)

        x = jax.random.normal(prng[1], (4,))
        y, logdet_fwd = bij.forward_and_log_det(x)
        z, logdet_inv = bij.inverse_and_log_det(x)

        self.assertEqual(y.shape, (4,))
        self.assertEqual(z.shape, (4,))
        self.assertEqual(logdet_fwd.shape, ())
        self.assertEqual(logdet_inv.shape, ())

    def test_identity_initialization(self):
        bij = DiagLinear(diag=jnp.ones((4,)))
        prng = jax.random.PRNGKey(42)
        x = jax.random.normal(prng, (4,))

        # Forward methods.
        y, logdet = bij.forward_and_log_det(x)
        np.testing.assert_array_equal(y, x)
        np.testing.assert_array_equal(logdet, jnp.zeros(1))

        # Inverse methods.
        x_rec, logdet = bij.inverse_and_log_det(y)
        np.testing.assert_array_equal(x_rec, y)
        np.testing.assert_array_equal(logdet, jnp.zeros(1))

    def test_inverse_methods(self):
        prng = jax.random.PRNGKey(42)
        prng = jax.random.split(prng, 2)
        diag = jax.random.uniform(prng[0], (4,)) + 0.5
        bij = DiagLinear(diag)
        x = jax.random.normal(prng[1], (4,))
        y, logdet_fwd = bij.forward_and_log_det(x)
        x_rec, logdet_inv = bij.inverse_and_log_det(y)
        np.testing.assert_allclose(x_rec, x, atol=1e-6)
        np.testing.assert_allclose(logdet_fwd, -logdet_inv, atol=1e-6)

    def test_forward_jacobian_det(self):
        prng = jax.random.PRNGKey(42)
        prng = jax.random.split(prng, 3)
        diag = jax.random.uniform(prng[0], (4,)) + 0.5
        bij = DiagLinear(diag)

        batched_x = jax.random.normal(prng[1], (10, 4))
        single_x = jax.random.normal(prng[2], (4,))
        batched_logdet = eqx.filter_vmap(bij.forward_log_det_jacobian)(batched_x)

        jacobian_fn = jax.jacfwd(bij.forward)
        logdet_numerical = jnp.linalg.slogdet(jacobian_fn(single_x))[1]
        for logdet in batched_logdet:
            np.testing.assert_allclose(logdet, logdet_numerical, atol=5e-4)

    def test_inverse_jacobian_det(self):
        prng = jax.random.PRNGKey(42)
        prng = jax.random.split(prng, 3)
        diag = jax.random.uniform(prng[0], (4,)) + 0.5
        bij = DiagLinear(diag)

        batched_y = jax.random.normal(prng[1], (10, 4))
        single_y = jax.random.normal(prng[2], (4,))
        batched_logdet = eqx.filter_vmap(bij.inverse_log_det_jacobian)(batched_y)

        jacobian_fn = jax.jacfwd(bij.inverse)
        logdet_numerical = jnp.linalg.slogdet(jacobian_fn(single_y))[1]
        for logdet in batched_logdet:
            np.testing.assert_allclose(logdet, logdet_numerical, atol=5e-4)

    def test_jittable(self):
        @eqx.filter_jit
        def f(x, b):
            return b.forward(x)

        bij = DiagLinear(diag=jnp.ones((4,)))
        x = jnp.zeros((4,))
        f(x, bij)

    def test_same_as_itself(self):
        bij = DiagLinear(diag=jnp.ones((4,)))
        self.assertTrue(bij.same_as(bij))

    def test_not_same_as_others(self):
        bij = DiagLinear(diag=jnp.ones((4,)))
        other = DiagLinear(diag=2.0 * jnp.ones((4,)))
        self.assertFalse(bij.same_as(other))
        self.assertFalse(bij.same_as(Tanh()))
