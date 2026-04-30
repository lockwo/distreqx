from typing import Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import (
    AbstractBijector,
    AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class Split(
    AbstractForwardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """A bijector that splits a single array into a tuple of arrays along an axis.

    This operates as a wrapper around `jax.numpy.split`.
    """

    indices_or_sections: Union[int, tuple[int, ...]] = eqx.field(static=True)
    axis: int = eqx.field(static=True)

    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def __init__(
        self,
        indices_or_sections: Union[int, tuple[int, ...], list[int]],
        axis: int = -1,
    ):
        """Initializes a Split bijector.

        **Arguments:**

        - `indices_or_sections`: If an integer `N`, the array will be divided into
          `N` equal arrays along axis. If a tuple/list of sorted integers, the entries
          indicate where along axis the array is split.
        - `axis`: The axis along which to split. Defaults to -1 (last axis).
        """
        # Ensure lists are converted to tuples so they remain hashable for JAX JIT
        if isinstance(indices_or_sections, list):
            indices_or_sections = tuple(indices_or_sections)

        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def forward_and_log_det(self, x: Array) -> tuple[tuple[Array, ...], Array]:
        """Computes y = tuple(split(x)) and log|det J(f)(x)| = 0."""
        y = tuple(jnp.split(x, self.indices_or_sections, axis=self.axis))
        return y, jnp.zeros((), dtype=x.dtype)

    def inverse_and_log_det(self, y: tuple[Array, ...]) -> tuple[Array, Array]:
        """Computes x = concatenate(y) and log|det J(f^{-1})(y)| = 0."""
        x = jnp.concatenate(y, axis=self.axis)
        dtype = y[0].dtype if y else jnp.float32
        return x, jnp.zeros((), dtype=dtype)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return (
            type(other) is Split
            and self.indices_or_sections == other.indices_or_sections
            and self.axis == other.axis
        )
