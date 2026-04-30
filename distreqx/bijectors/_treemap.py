"""TreeMap Bijector for applying a pytree of bijectors to a pytree of inputs."""

import functools

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from ._bijector import (
    AbstractBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


def _is_bijector(node: PyTree) -> bool:
    return isinstance(node, AbstractBijector)


class TreeMap(AbstractFwdLogDetJacBijector, AbstractInvLogDetJacBijector, strict=True):
    """Applies a pytree of bijectors to a pytree of inputs.

    This behaves analogously to TensorFlow Probability's `JointMap`. It allows
    applying independent bijectors to a structured input (e.g., a tuple or dict
    of arrays) and aggregates the log-determinants across the structure.
    `None` values in the bijector pytree act as identity transformations.
    """

    bijectors: PyTree[AbstractBijector]
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, bijectors: PyTree[AbstractBijector]):
        """Initializes a TreeMap bijector."""
        leaves = [
            b
            for b in jax.tree_util.tree_leaves(bijectors, is_leaf=_is_bijector)
            if b is not None
        ]
        if not leaves:
            raise ValueError(
                "The pytree of bijectors must contain at least one valid bijector."
            )

        self.bijectors = bijectors

        is_constant_jacobian = all(b.is_constant_jacobian for b in leaves)
        is_constant_log_det = all(b.is_constant_log_det for b in leaves)

        if is_constant_log_det is None:
            is_constant_log_det = is_constant_jacobian
        if is_constant_jacobian and not is_constant_log_det:
            raise ValueError(
                "The Jacobian is said to be constant, but its "
                "determinant is said not to be, which is impossible."
            )
        self._is_constant_jacobian = is_constant_jacobian
        self._is_constant_log_det = is_constant_log_det

    def forward(self, x: PyTree) -> PyTree:
        """Computes y = f(x)."""
        return jax.tree_util.tree_map(
            lambda b, v: b.forward(v) if b is not None else v,
            self.bijectors,
            x,
            is_leaf=_is_bijector,
        )

    def inverse(self, y: PyTree) -> PyTree:
        """Computes x = f^{-1}(y)."""
        return jax.tree_util.tree_map(
            lambda b, v: b.inverse(v) if b is not None else v,
            self.bijectors,
            y,
            is_leaf=_is_bijector,
        )

    def forward_and_log_det(self, x: PyTree) -> tuple[PyTree, PyTree]:
        """Computes y = f(x) and sum of log|det J(f)(x)|."""
        ys_and_log_dets = jax.tree_util.tree_map(
            lambda b, v: (
                b.forward_and_log_det(v) if b is not None else (v, jnp.array(0.0))
            ),
            self.bijectors,
            x,
            is_leaf=_is_bijector,
        )

        y = jax.tree_util.tree_map(
            lambda b, res: res[0], self.bijectors, ys_and_log_dets, is_leaf=_is_bijector
        )
        log_dets = jax.tree_util.tree_map(
            lambda b, res: res[1], self.bijectors, ys_and_log_dets, is_leaf=_is_bijector
        )

        log_det_leaves = jax.tree_util.tree_leaves(log_dets)
        total_log_det = functools.reduce(jnp.add, log_det_leaves)
        return y, total_log_det

    def inverse_and_log_det(self, y: PyTree) -> tuple[PyTree, PyTree]:
        """Computes x = f^{-1}(y) and sum of log|det J(f^{-1})(y)|."""
        xs_and_log_dets = jax.tree_util.tree_map(
            lambda b, v: (
                b.inverse_and_log_det(v) if b is not None else (v, jnp.array(0.0))
            ),
            self.bijectors,
            y,
            is_leaf=_is_bijector,
        )

        x = jax.tree_util.tree_map(
            lambda b, res: res[0], self.bijectors, xs_and_log_dets, is_leaf=_is_bijector
        )
        log_dets = jax.tree_util.tree_map(
            lambda b, res: res[1], self.bijectors, xs_and_log_dets, is_leaf=_is_bijector
        )

        log_det_leaves = jax.tree_util.tree_leaves(log_dets)
        total_log_det = functools.reduce(jnp.add, log_det_leaves)
        return x, total_log_det

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is TreeMap:
            if jax.tree_util.tree_structure(
                self.bijectors, is_leaf=_is_bijector
            ) != jax.tree_util.tree_structure(other.bijectors, is_leaf=_is_bijector):
                return False

            def _check_same(b1, b2):
                if b1 is None and b2 is None:
                    return True
                if b1 is None or b2 is None:
                    return False
                return b1.same_as(b2)

            match_tree = jax.tree_util.tree_map(
                _check_same,
                self.bijectors,
                other.bijectors,
                is_leaf=_is_bijector,
            )
            return all(jax.tree_util.tree_leaves(match_tree))
        return False
