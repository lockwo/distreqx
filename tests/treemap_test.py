"""Tests for `tree_map.py`."""

from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Shift, Tanh, TreeMap
from distreqx.bijectors._bijector import AbstractBijector


def _is_bijector(node):
    return isinstance(node, AbstractBijector)


class TreeMapTest(TestCase):
    def test_empty_tree_raises(self):
        with self.assertRaisesRegex(
            ValueError, "The pytree of bijectors cannot be empty"
        ):
            TreeMap({})
        with self.assertRaisesRegex(
            ValueError, "The pytree of bijectors cannot be empty"
        ):
            TreeMap([])

    def test_jacobian_is_constant_property(self):
        # All bijectors have constant jacobians
        const_bij = TreeMap({"a": Shift(jnp.ones((4,))), "b": Shift(jnp.ones((2,)))})
        self.assertTrue(const_bij.is_constant_jacobian)
        self.assertTrue(const_bij.is_constant_log_det)

        # Mixed bijectors (Tanh does not have a constant jacobian)
        mixed_bij = TreeMap({"a": Shift(jnp.ones((4,))), "b": Tanh()})
        self.assertFalse(mixed_bij.is_constant_jacobian)
        self.assertFalse(mixed_bij.is_constant_log_det)

    @parameterized.expand(
        [
            (
                "dict_tree",
                {"a": Shift(jnp.array(1.0)), "b": Tanh()},
                {"a": jnp.array(0.0), "b": jnp.array(0.5)},
            ),
            (
                "tuple_tree",
                (Shift(jnp.array(2.0)), Tanh()),
                (jnp.array(0.0), jnp.array(-0.5)),
            ),
            (
                "nested_tree",
                {"a": (Shift(jnp.array(1.0)),), "b": Tanh()},
                {"a": (jnp.array(0.0),), "b": jnp.array(0.5)},
            ),
        ]
    )
    def test_forward_methods(self, name, bijectors, x):
        tree_bij = TreeMap(bijectors)

        y1 = tree_bij.forward(x)
        logdet1 = tree_bij.forward_log_det_jacobian(x)
        y2, logdet2 = tree_bij.forward_and_log_det(x)

        # Verify tree structures match
        self.assertEqual(
            jax.tree_util.tree_structure(y1), jax.tree_util.tree_structure(x)
        )
        self.assertEqual(
            jax.tree_util.tree_structure(y2), jax.tree_util.tree_structure(x)
        )

        # Manually compute expected values via tree_map
        # using is_leaf to stop at bijectors
        expected_y = jax.tree_util.tree_map(
            lambda b, v: b.forward(v), bijectors, x, is_leaf=_is_bijector
        )
        expected_logdets_tree = jax.tree_util.tree_map(
            lambda b, v: b.forward_log_det_jacobian(v),
            bijectors,
            x,
            is_leaf=_is_bijector,
        )
        expected_logdet = sum(jax.tree_util.tree_leaves(expected_logdets_tree))

        # Assert shapes and values
        jax.tree_util.tree_map(
            lambda res, exp: self.assertEqual(res.shape, exp.shape), y1, expected_y
        )
        jax.tree_util.tree_map(
            lambda res, exp: np.testing.assert_allclose(res, exp, 1e-6), y1, expected_y
        )
        jax.tree_util.tree_map(
            lambda res, exp: np.testing.assert_allclose(res, exp, 1e-6), y2, expected_y
        )

        np.testing.assert_allclose(logdet1, expected_logdet, 1e-6)
        np.testing.assert_allclose(logdet2, expected_logdet, 1e-6)

    @parameterized.expand(
        [
            (
                "dict_tree",
                {"a": Shift(jnp.array(1.0)), "b": Tanh()},
                {"a": jnp.array(1.0), "b": jnp.array(0.2)},
            ),
            (
                "tuple_tree",
                (Shift(jnp.array(2.0)), Tanh()),
                (jnp.array(2.0), jnp.array(-0.2)),
            ),
            (
                "nested_tree",
                {"a": (Shift(jnp.array(1.0)),), "b": Tanh()},
                {"a": (jnp.array(1.0),), "b": jnp.array(0.2)},
            ),
        ]
    )
    def test_inverse_methods(self, name, bijectors, y):
        tree_bij = TreeMap(bijectors)

        x1 = tree_bij.inverse(y)
        logdet1 = tree_bij.inverse_log_det_jacobian(y)
        x2, logdet2 = tree_bij.inverse_and_log_det(y)

        # Verify tree structures match
        self.assertEqual(
            jax.tree_util.tree_structure(x1), jax.tree_util.tree_structure(y)
        )
        self.assertEqual(
            jax.tree_util.tree_structure(x2), jax.tree_util.tree_structure(y)
        )

        # Manually compute expected values via tree_map
        # using is_leaf to stop at bijectors
        expected_x = jax.tree_util.tree_map(
            lambda b, v: b.inverse(v), bijectors, y, is_leaf=_is_bijector
        )
        expected_logdets_tree = jax.tree_util.tree_map(
            lambda b, v: b.inverse_log_det_jacobian(v),
            bijectors,
            y,
            is_leaf=_is_bijector,
        )
        expected_logdet = sum(jax.tree_util.tree_leaves(expected_logdets_tree))

        # Assert shapes and values
        jax.tree_util.tree_map(
            lambda res, exp: self.assertEqual(res.shape, exp.shape), x1, expected_x
        )
        jax.tree_util.tree_map(
            lambda res, exp: np.testing.assert_allclose(res, exp, 1e-6), x1, expected_x
        )
        jax.tree_util.tree_map(
            lambda res, exp: np.testing.assert_allclose(res, exp, 1e-6), x2, expected_x
        )

        np.testing.assert_allclose(logdet1, expected_logdet, 1e-6)
        np.testing.assert_allclose(logdet2, expected_logdet, 1e-6)

    def test_jittable(self):
        @jax.jit
        def f(x, b):
            return b.forward(x)

        bij = TreeMap({"a": Shift(jnp.ones((4,))), "b": Tanh()})
        x = {"a": np.zeros((4,)), "b": np.zeros((4,))}

        z = f(x, bij)

        self.assertIsInstance(z, dict)
        self.assertIsInstance(z["a"], jnp.ndarray)
        self.assertIsInstance(z["b"], jnp.ndarray)

    def test_same_as_itself(self):
        bij = TreeMap({"a": Shift(jnp.ones((4,))), "b": Tanh()})
        # Distreqx bijectors often evaluate False for different object
        # instances, so checking the exact same instance is the correct test.
        self.assertTrue(bij.same_as(bij))

    def test_not_same_as_others(self):
        bij = TreeMap({"a": Shift(jnp.ones((4,))), "b": Tanh()})

        # Completely different bijector
        other_type = Shift(jnp.zeros((4,)))
        self.assertFalse(bij.same_as(other_type))

        # Same structure, different bijector parameters
        different_params = TreeMap({"a": Shift(jnp.zeros((4,))), "b": Tanh()})
        self.assertFalse(bij.same_as(different_params))

        # Different structure
        different_structure = TreeMap({"a": Shift(jnp.ones((4,)))})
        self.assertFalse(bij.same_as(different_structure))
