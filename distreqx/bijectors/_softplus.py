import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import (
    AbstractBijector,
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class Softplus(
    AbstractFowardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """
    Transforms the real line to the positive domain using
    softplus y = log(1 + exp(x)).
    """

    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = softplus(x) and log|det J(f)(x)|."""
        y = jnn.softplus(x)
        logdet = -jnn.softplus(-x)
        return y, logdet

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = softplus^{-1}(y) and log|det J(f^{-1})(y)|."""
        x = jnp.log(-jnp.expm1(-y)) + y
        logdet = jnn.softplus(-x)
        return x, logdet

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is Softplus
