"""Tests for `chain.py`."""

from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np

from distreqx.bijectors import AbstractBijector, Chain, ScalarAffine, Tanh


RTOL = 1e-2


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

    def test_jittable(self):
        @jax.jit
        def f(x, b):
            return b.forward(x)

        bijector = Chain([ScalarAffine(jnp.array(0.0), jnp.array(1.0))])
        x = np.zeros(())
        y = f(x, bijector)
        self.assertIsInstance(y, jax.Array)
