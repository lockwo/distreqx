import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import (
    AbstractBijector,
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class Reshape(
    AbstractFowardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """A bijector that reshapes the input array."""

    in_shape: tuple[int, ...] = eqx.field(static=True)
    out_shape: tuple[int, ...] = eqx.field(static=True)

    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def __init__(self, in_shape: tuple[int, ...], out_shape: tuple[int, ...]):
        """Initializes a Reshape bijector.

        **Arguments:**

        - `in_shape`: The shape of the input event.
        - `out_shape`: The desired shape of the output event.
        """
        in_size = math.prod(in_shape)
        out_size = math.prod(out_shape)

        if in_size != out_size:
            raise ValueError(
                f"Shapes are incompatible: in_shape {in_shape} (size {in_size}) and "
                f"out_shape {out_shape} (size {out_size}) must have the same total "
                f"number of elements."
            )

        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = reshape(x) and log|det J(f)(x)| = 0."""
        y = jnp.reshape(x, self.out_shape)
        return y, jnp.zeros((), dtype=x.dtype)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = reshape(y) and log|det J(f^{-1})(y)| = 0."""
        x = jnp.reshape(y, self.in_shape)
        return x, jnp.zeros((), dtype=y.dtype)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return (
            type(other) is Reshape
            and self.in_shape == other.in_shape
            and self.out_shape == other.out_shape
        )
