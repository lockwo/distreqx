"""Tests for `bijector.py`."""

from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np

from distreqx.bijectors import (
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class DummyBijector(
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
    strict=True,
):
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def forward_and_log_det(self, x):
        return x, jnp.zeros(x.shape[:-1], jnp.float_)

    def inverse_and_log_det(self, y):
        return y, jnp.zeros(y.shape[:-1], jnp.float_)

    def same_as(self, other):
        raise NotImplementedError


class BijectorTest(TestCase):
    def test_jittable(self):
        @jax.jit
        def forward(bij, x):
            return bij.forward(x)

        bij = DummyBijector(True, True)
        x = jnp.zeros((4,))
        np.testing.assert_allclose(forward(bij, x), x)
