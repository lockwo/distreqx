"""Tests for `chain.py`."""

from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np

from distreqx.bijectors import (
    AbstractBijector,
    AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
    Chain,
    ScalarAffine,
    Tanh,
)

RTOL = 1e-2


class DummyPytreeBijector(
    AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
):
    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def forward_and_log_det(self, x):
        return {"a": x[:1], "b": x[1:]}, jnp.zeros(())

    def inverse_and_log_det(self, y):
        return jnp.concatenate([y["a"], y["b"]]), jnp.zeros(())

    def same_as(self, other):
        return type(other) is DummyPytreeBijector


class ChainTest(TestCase):
    def setUp(self):
        self.seed = jax.random.key(1234)

    def test_properties(self):
        bijector = Chain([Tanh()])
        for bij in bijector.bijectors:
            assert isinstance(bij, AbstractBijector)

    def test_raises_on_empty_list(self):
        with self.assertRaises(ValueError):
            Chain([])

    def test_pytree_valued_bijector(self):
        bijector = Chain([DummyPytreeBijector()])
        x = jnp.arange(2.0)
        y = bijector.forward(x)
        self.assertEqual(set(y.keys()), {"a", "b"})
        np.testing.assert_array_equal(bijector.inverse(y), x)

    def test_jittable(self):
        @jax.jit
        def f(x, b):
            return b.forward(x)

        bijector = Chain([ScalarAffine(jnp.array(0.0), jnp.array(1.0))])
        x = np.zeros(())
        y = f(x, bijector)
        self.assertIsInstance(y, jax.Array)
