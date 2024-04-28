"""Tests for `bijector.py`."""
from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import AbstractBijector


class DummyBijector(AbstractBijector):
    def forward_and_log_det(self, x):
        return x, jnp.zeros(x.shape[:-1], jnp.float_)

    def inverse_and_log_det(self, y):
        return y, jnp.zeros(y.shape[:-1], jnp.float_)


class BijectorTest(TestCase):
    @parameterized.expand(
        [
            ("non-consistent constant properties", True, False),
        ]
    )
    def test_invalid_parameters(self, name, cnst_jac, cnst_logdet):
        with self.assertRaises(ValueError):
            DummyBijector(cnst_jac, cnst_logdet)

    def test_jittable(self):
        @jax.jit
        def forward(bij, x):
            return bij.forward(x)

        bij = DummyBijector(True, True)
        x = jnp.zeros((4,))
        np.testing.assert_allclose(forward(bij, x), x)
