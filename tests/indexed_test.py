from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Indexed, ScalarAffine


class IndexedTest(TestCase):
    def setUp(self):
        # We apply a ScalarAffine (y = 2x) only to indices 1 and 3
        self.inner_bij = ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(2.0))
        self.bij = Indexed(bijector=self.inner_bij, indices=[1, 3])

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_forward_and_inverse(self, name, dtype):
        x = jnp.array([10.0, 0.0, 20.0, 1.0], dtype=dtype)

        y, log_det_fwd = self.bij.forward_and_log_det(x)

        # Indices 0 and 2 unchanged. Indices 1 and 3 doubled.
        expected_y = jnp.array([10.0, 0.0, 20.0, 2.0], dtype=dtype)
        self.assertion_fn()(y, expected_y)

        # The total log_det is the sum of the log_dets of the
        # transformed indices (log(2) for each of index 1 and 3)
        expected_logdet = jnp.array([jnp.log(2.0), jnp.log(2.0)], dtype=dtype)
        self.assertion_fn()(log_det_fwd, expected_logdet)

        x_rec, log_det_inv = self.bij.inverse_and_log_det(y)
        self.assertion_fn()(x_rec, x)
        self.assertion_fn()(log_det_inv, -expected_logdet)

    def test_boolean_mask_indexing(self):
        # Indexed can also accept boolean masks instead of integer indices
        affine = ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(2.0))
        bool_bij = Indexed(bijector=affine, indices=[False, True, False, True])

        x = jnp.array([10.0, 0.0, 20.0, 1.0])
        y, log_det = bool_bij.forward_and_log_det(x)

        expected_y = jnp.array([10.0, 0.0, 20.0, 2.0])
        self.assertion_fn()(y, expected_y)

    def test_jittable(self):
        @eqx.filter_jit
        def f(bij, x):
            return bij.forward_and_log_det(x)

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y, logdet = f(self.bij, x)
        self.assertIsInstance(y, jax.Array)
        self.assertIsInstance(logdet, jax.Array)
