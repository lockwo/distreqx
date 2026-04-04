import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from ._bijector import (
    AbstractBijector,
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class Identity(
    AbstractFowardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """Identity bijector: y = x."""

    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def forward_and_log_det(self, x: PyTree) -> tuple[PyTree, PyTree]:
        """Computes y = x and log|det J(f)(x)| = 0."""
        log_det = jax.tree_util.tree_map(jnp.zeros_like, x)
        return x, log_det

    def inverse_and_log_det(self, y: PyTree) -> tuple[PyTree, PyTree]:
        """Computes x = y and log|det J(f^{-1})(y)| = 0."""
        log_det = jax.tree_util.tree_map(jnp.zeros_like, y)
        return y, log_det

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is Identity
