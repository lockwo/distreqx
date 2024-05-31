"""Tests for `scalar_affine.py`."""

from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np

from distreqx.bijectors import ScalarAffine


class ScalarAffineTest(TestCase):
    def test_properties(self):
        bij = ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(1.0))
        self.assertTrue(bij.is_constant_jacobian)
        self.assertTrue(bij.is_constant_log_det)
        np.testing.assert_allclose(bij.shift, 0.0)
        np.testing.assert_allclose(bij.scale, 1.0)
        np.testing.assert_allclose(bij.log_scale, 0.0)

    def test_raises_if_both_scale_and_log_scale_are_specified(self):
        with self.assertRaises(ValueError):
            ScalarAffine(
                shift=jnp.array(0.0), scale=jnp.array(1.0), log_scale=jnp.array(0.0)
            )

    def test_shapes_are_correct(self):
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
        x = jax.random.normal(k1, (3, 4, 5))
        shift = jax.random.normal(k2, (3, 4, 5))
        scale = jax.random.uniform(k3, (3, 4, 5)) + 0.1
        log_scale = jax.random.normal(k4, (3, 4, 5))
        bij_no_scale = ScalarAffine(shift)
        bij_with_scale = ScalarAffine(shift, scale=scale)
        bij_with_log_scale = ScalarAffine(shift, log_scale=log_scale)
        for bij in [bij_no_scale, bij_with_scale, bij_with_log_scale]:
            # Forward methods.
            y, logdet = bij.forward_and_log_det(x)
            self.assertEqual(y.shape, (3, 4, 5))
            self.assertEqual(logdet.shape, (3, 4, 5))
            # Inverse methods.
            x, logdet = bij.inverse_and_log_det(y)
            self.assertEqual(x.shape, (3, 4, 5))
            self.assertEqual(logdet.shape, (3, 4, 5))

    def test_forward_methods_are_correct(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (2, 3, 4, 5))
        bij_no_scale = ScalarAffine(shift=jnp.array(3.0))
        bij_with_scale = ScalarAffine(shift=jnp.array(3.0), scale=jnp.array(1.0))
        bij_with_log_scale = ScalarAffine(
            shift=jnp.array(3.0), log_scale=jnp.array(0.0)
        )
        for bij in [bij_no_scale, bij_with_scale, bij_with_log_scale]:
            y, logdet = bij.forward_and_log_det(x)
            np.testing.assert_allclose(y, x + 3.0, atol=1e-8)
            np.testing.assert_allclose(logdet, 0.0, atol=1e-8)

    def test_inverse_methods_are_correct(self):
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
        x = jax.random.normal(k1, (2, 3, 4, 5))
        shift = jax.random.normal(k2, (4, 5))
        scale = jax.random.uniform(k3, (3, 4, 5)) + 0.1
        log_scale = jax.random.normal(k4, (3, 4, 5))
        bij_no_scale = ScalarAffine(shift)
        bij_with_scale = ScalarAffine(shift, scale=scale)
        bij_with_log_scale = ScalarAffine(shift, log_scale=log_scale)
        for bij in [bij_no_scale, bij_with_scale, bij_with_log_scale]:
            y, logdet_fwd = bij.forward_and_log_det(x)
            x_rec, logdet_inv = bij.inverse_and_log_det(y)
            np.testing.assert_allclose(x_rec, x, atol=1e-5)
            np.testing.assert_allclose(logdet_fwd, -logdet_inv, atol=3e-6)

    def test_composite_methods_are_consistent(self):
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
        bij = ScalarAffine(
            shift=jax.random.normal(k1, (4, 5)), log_scale=jax.random.normal(k2, (4, 5))
        )
        # Forward methods.
        x = jax.random.normal(k3, (2, 3, 4, 5))
        y1 = bij.forward(x)
        logdet1 = bij.forward_log_det_jacobian(x)
        y2, logdet2 = bij.forward_and_log_det(x)
        np.testing.assert_allclose(y1, y2, atol=1e-12)
        np.testing.assert_allclose(logdet1, logdet2, atol=1e-12)
        # Inverse methods.
        y = jax.random.normal(k4, (2, 3, 4, 5))
        x1 = bij.inverse(y)
        logdet1 = bij.inverse_log_det_jacobian(y)
        x2, logdet2 = bij.inverse_and_log_det(y)
        np.testing.assert_allclose(x1, x2, atol=1e-12)
        np.testing.assert_allclose(logdet1, logdet2, atol=1e-12)

    def test_jittable(self):
        @jax.jit
        def f(x, b):
            return b.forward(x)

        bijector = ScalarAffine(jnp.array(0.0), jnp.array(1.0))
        x = jnp.zeros(())
        y = f(x, bijector)
        self.assertIsInstance(y, jax.Array)
