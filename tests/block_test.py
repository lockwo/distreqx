"""Tests for `block.py`."""

from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import AbstractBijector, Block, ScalarAffine, Tanh


RTOL = 1e-6
seed = jax.random.key(1234)


class BlockTest(TestCase):
    def test_properties(self):
        bijct = Tanh()
        block = Block(bijct, 1)
        self.assertEqual(block.ndims, 1)
        self.assertIsInstance(block.bijector, AbstractBijector)

    def test_invalid_properties(self):
        bijct = Tanh()
        with self.assertRaises(ValueError):
            Block(bijct, -1)

    @parameterized.expand(
        [
            ("dx_tanh_0", Tanh, 0),
            ("dx_tanh_1", Tanh, 1),
            ("dx_tanh_2", Tanh, 2),
        ]
    )
    def test_forward_inverse_work_as_expected(self, name, bijector_fn, ndims):
        bijct = bijector_fn()
        x = jax.random.normal(seed, [2, 3])
        block = Block(bijct, ndims)
        np.testing.assert_array_equal(bijct.forward(x), block.forward(x))
        np.testing.assert_array_equal(bijct.inverse(x), block.inverse(x))
        np.testing.assert_allclose(
            bijct.forward_and_log_det(x)[0], block.forward_and_log_det(x)[0], atol=2e-7
        )
        np.testing.assert_array_equal(
            bijct.inverse_and_log_det(x)[0], block.inverse_and_log_det(x)[0]
        )

    @parameterized.expand(
        [
            ("dx_tanh_0", Tanh, 0),
            ("dx_tanh_1", Tanh, 1),
            ("dx_tanh_2", Tanh, 2),
        ]
    )
    def test_log_det_jacobian_works_as_expected(self, name, bijector_fn, ndims):
        bijct = bijector_fn()
        x = jax.random.normal(seed, [2, 3])
        block = Block(bijct, ndims)
        axes = tuple(range(-ndims, 0))
        np.testing.assert_allclose(
            bijct.forward_log_det_jacobian(x).sum(axes),
            block.forward_log_det_jacobian(x),
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            bijct.inverse_log_det_jacobian(x).sum(axes),
            block.inverse_log_det_jacobian(x),
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            bijct.forward_and_log_det(x)[1].sum(axes),
            block.forward_and_log_det(x)[1],
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            bijct.inverse_and_log_det(x)[1].sum(axes),
            block.inverse_and_log_det(x)[1],
            rtol=RTOL,
        )

    def test_jittable(self):
        @jax.jit
        def f(x, b):
            return b.forward(x)

        bijector = Block(ScalarAffine(jnp.array(0)), 1)
        x = jnp.zeros((2, 3))
        y = f(x, bijector)
        self.assertIsInstance(y, jax.Array)
