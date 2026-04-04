from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from distreqx.bijectors import RationalQuadraticSpline


class RationalQuadraticSplineTest(TestCase):
    def assertion_fn(self, rtol=1e-4, atol=1e-7):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)

    def test_identity_initialization(self):
        # A params array of all zeros should result in exactly the identity function
        num_bins = 4
        params = jnp.zeros(3 * num_bins + 1)
        bij = RationalQuadraticSpline(params, range_min=-2.0, range_max=2.0)

        x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        y, logdet = bij.forward_and_log_det(x)

        self.assertion_fn()(y, x)
        self.assertion_fn()(logdet, jnp.zeros_like(x))

    def test_forward_and_inverse(self):
        key = jax.random.key(42)
        params = jax.random.normal(key, (10,))  # 3 bins = 3*3 + 1 = 10 params
        bij = RationalQuadraticSpline(params, range_min=-5.0, range_max=5.0)

        # Test values inside and outside the defined range
        x = jnp.array([-10.0, -2.5, 0.0, 4.5, 10.0])
        y, log_det_fwd = bij.forward_and_log_det(x)
        x_rec, log_det_inv = bij.inverse_and_log_det(y)

        self.assertion_fn()(x_rec, x)
        self.assertion_fn()(log_det_fwd, -log_det_inv)

    def test_jittable(self):
        @eqx.filter_jit
        def f(bij, x):
            return bij.forward_and_log_det(x)

        params = jnp.ones(10)
        bij = RationalQuadraticSpline(params, range_min=-1.0, range_max=1.0)
        y, logdet = f(bij, jnp.array(0.5))

        self.assertIsInstance(y, jax.Array)
