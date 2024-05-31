"""Tests for `independent.py`."""

from unittest import TestCase

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from distreqx.distributions import Independent, Normal


class IndependentTest(TestCase):
    """Class to test miscellaneous methods of the `Independent` distribution."""

    def setUp(self):
        self.loc = jnp.array(np.random.randn(2, 3, 4))
        self.scale = jnp.array(np.abs(np.random.randn(2, 3, 4)))
        self.base = Normal(loc=self.loc, scale=self.scale)
        self.dist = Independent(self.base)

    def assertion_fn(self, rtol):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_constructor_is_jittable_given_ndims(self):
        constructor = lambda d: Independent(d)
        model = eqx.filter_jit(constructor)(self.base)
        self.assertIsInstance(model, Independent)
