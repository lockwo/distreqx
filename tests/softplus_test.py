from unittest import TestCase
import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from distreqx.bijectors import Softplus

class SoftplusTest(TestCase):
    def setUp(self):
        self.bij = Softplus()

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_forward_and_inverse(self):
        x = jnp.array([-5.0, 0.0, 5.0])
        y, log_det_fwd = self.bij.forward_and_log_det(x)
        
        expected_y = jax.nn.softplus(x)
        self.assertion_fn()(y, expected_y)
        self.assertion_fn()(log_det_fwd, -jax.nn.softplus(-x))

        x_rec, log_det_inv = self.bij.inverse_and_log_det(y)
        self.assertion_fn()(x_rec, x)
        self.assertion_fn()(log_det_inv, -log_det_fwd)

    def test_jittable(self):
        @eqx.filter_jit
        def f(bij, x):
            return bij.forward_and_log_det(x)
        
        y, log_det = f(self.bij, jnp.array(1.0))
        self.assertIsInstance(y, jax.Array)