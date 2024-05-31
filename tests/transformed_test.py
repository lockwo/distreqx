"""Tests for `transformed.py`."""

from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import ScalarAffine, Sigmoid
from distreqx.distributions import Normal, Transformed


class TransformedTest(TestCase):
    def setUp(self):
        self.seed = jax.random.PRNGKey(1234)

    @parameterized.expand(
        [
            ("int16", jnp.array([0, 0], dtype=np.int16), Normal),
            ("int32", jnp.array([0, 0], dtype=np.int32), Normal),
            ("int64", jnp.array([0, 0], dtype=np.int64), Normal),
        ]
    )
    def test_integer_inputs(self, name, inputs, base_dist):
        base = base_dist(
            jnp.zeros_like(inputs, dtype=jnp.float32),
            jnp.ones_like(inputs, dtype=jnp.float32),
        )
        bijector = ScalarAffine(shift=jnp.array(0.0))
        dist = Transformed(base, bijector)

        log_prob = dist.log_prob(inputs)

        standard_normal_log_prob_of_zero = jnp.array(-0.9189385)
        expected_log_prob = jnp.full_like(
            inputs, standard_normal_log_prob_of_zero, dtype=jnp.float32
        )

        np.testing.assert_array_equal(log_prob, expected_log_prob)

    @parameterized.expand(
        [
            ("kl distreqx_to_distreqx", "distreqx_to_distreqx"),
        ]
    )
    def test_kl_divergence(self, name, mode_string):
        base_dist1 = Normal(
            loc=jnp.array([0.1, 0.5, 0.9]), scale=jnp.array([0.1, 1.1, 2.5])
        )
        base_dist2 = Normal(
            loc=jnp.array([-0.1, -0.5, 0.9]), scale=jnp.array([0.1, -1.1, 2.5])
        )
        bij_distreqx1 = ScalarAffine(shift=jnp.array(0.0))
        bij_distreqx2 = ScalarAffine(shift=jnp.array(0.0))
        distreqx_dist1 = Transformed(base_dist1, bij_distreqx1)
        distreqx_dist2 = Transformed(base_dist2, bij_distreqx2)

        expected_result_fwd = base_dist1.kl_divergence(base_dist2)
        expected_result_inv = base_dist2.kl_divergence(base_dist1)

        if mode_string == "distreqx_to_distreqx":
            result_fwd = distreqx_dist1.kl_divergence(distreqx_dist2)
            result_inv = distreqx_dist2.kl_divergence(distreqx_dist1)
        else:
            raise ValueError(f"Unsupported mode string: {mode_string}")

        np.testing.assert_allclose(result_fwd, expected_result_fwd, rtol=1e-2)
        np.testing.assert_allclose(result_inv, expected_result_inv, rtol=1e-2)

    def test_kl_divergence_on_same_instance_of_distreqx_bijector(self):
        base_dist1 = Normal(
            loc=jnp.array([0.1, 0.5, 0.9]), scale=jnp.array([0.1, 1.1, 2.5])
        )
        base_dist2 = Normal(
            loc=jnp.array([-0.1, -0.5, 0.9]), scale=jnp.array([0.1, -1.1, 2.5])
        )
        bij_distreqx = Sigmoid()
        distreqx_dist1 = Transformed(base_dist1, bij_distreqx)
        distreqx_dist2 = Transformed(base_dist2, bij_distreqx)
        expected_result_fwd = base_dist1.kl_divergence(base_dist2)
        expected_result_inv = base_dist2.kl_divergence(base_dist1)
        result_fwd = distreqx_dist1.kl_divergence(distreqx_dist2)
        result_inv = distreqx_dist2.kl_divergence(distreqx_dist1)
        np.testing.assert_allclose(result_fwd, expected_result_fwd, rtol=1e-2)
        np.testing.assert_allclose(result_inv, expected_result_inv, rtol=1e-2)

    def test_jittable(self):
        @jax.jit
        def f(x, d):
            return d.log_prob(x)

        base = Normal(jnp.array(0.0), jnp.array(1.0))
        bijector = ScalarAffine(jnp.array(0.0), jnp.array(1.0))
        dist = Transformed(base, bijector)
        x = jnp.zeros(())
        y = f(x, dist)
        self.assertIsInstance(y, jax.Array)
