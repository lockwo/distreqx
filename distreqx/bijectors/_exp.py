import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import (
    AbstractBijector,
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class Exp(
    AbstractFowardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """Exponential bijector: y = exp(x)."""

    _is_constant_jacobian: bool = False
    _is_constant_log_det: bool = False

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = exp(x) and log|det J(f)(x)| = x."""
        return jnp.exp(x), x

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = log(y) and log|det J(f^{-1})(y)| = -log(y)."""
        x = jnp.log(y)
        # Optimization: since x = log(y), the log det is simply -x
        return x, -x

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is Exp