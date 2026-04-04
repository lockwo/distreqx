from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Indexed, Tanh


class IndexedTest(TestCase):
    def setUp(self):
        # We apply an Tanh bijector only to indices 1 and 3
        self.inner_bij = Tanh()
        self.bij = Indexed(bijector=self.inner_bij, indices=[1, 3])

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_forward_and_inverse(self, name, dtype):
        x = jnp.array([10.0, 0.0, 20.0, 1.0], dtype=dtype)

        y, log_det_fwd = self.bij.forward_and_log_det(x)

        # Indices 0 and 2 unchanged. Indices 1 and 3 tanh'd.
        expected_y = jnp.array([10.0, 0.0, 20.0, jnp.tanh(1.0)], dtype=dtype)
        self.assertion_fn()(y, expected_y)

        # The log_det of tanh(x) is log(1 - tanh(x)^2)
        # We can write it as jnp.log(1.0 - jnp.tanh(1.0)**2)
        # or the numerically stabler -2.0 * jnp.log(jnp.cosh(1.0))
        expected_logdet = jnp.array(
            [0.0, jnp.log(1.0 - jnp.tanh(1.0) ** 2)], dtype=dtype
        )
        self.assertion_fn()(log_det_fwd, expected_logdet)

        x_rec, log_det_inv = self.bij.inverse_and_log_det(y)
        self.assertion_fn()(x_rec, x)
        self.assertion_fn()(log_det_inv, -expected_logdet)

    def test_boolean_mask_indexing(self):
        # Indexed can also accept boolean masks instead of integer indices
        bool_bij = Indexed(bijector=Tanh(), indices=[False, True, False, True])

        x = jnp.array([10.0, 0.0, 20.0, 1.0])
        y, log_det = bool_bij.forward_and_log_det(x)

        expected_y = jnp.array([10.0, 0.0, 20.0, jnp.tanh(1.0)])
        self.assertion_fn()(y, expected_y)

    def test_jittable(self):
        @eqx.filter_jit
        def f(bij, x):
            return bij.forward_and_log_det(x)

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y, logdet = f(self.bij, x)
        self.assertIsInstance(y, jax.Array)
        self.assertIsInstance(logdet, jax.Array)
