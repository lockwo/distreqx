from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import (
    AbstractBijector,
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class SplitCoupling(
    AbstractFowardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """Split coupling bijector, with arbitrary conditioner & inner bijector."""

    split_index: int = eqx.field(static=True)
    split_axis: int = eqx.field(static=True)
    swap: bool = eqx.field(static=True)
    conditioner: Any  # Usually an eqx.nn.MLP or similar PyTree module
    bijector_fn: Callable[[Any], AbstractBijector] = eqx.field(static=True)

    _is_constant_jacobian: bool = eqx.field(static=True, init=False)
    _is_constant_log_det: bool = eqx.field(static=True, init=False)

    def __init__(
        self,
        split_index: int,
        conditioner: Any,
        bijector: Callable[[Any], AbstractBijector],
        swap: bool = False,
        split_axis: int = -1,
    ):
        """Initializes a SplitCoupling bijector.

        **Arguments:**

        - `split_index`: The index used to split the input array along the `split_axis`.
        - `conditioner`: A callable (usually an Equinox module) that takes the
            unchanged slice and outputs parameters for the inner bijector.
        - `bijector`: A callable that takes the parameters and returns an
            instantiated `distreqx` bijector.
        - `swap`: If True, the second half of the split remains unchanged and
            conditions the first half.
        - `split_axis`: The axis along which to split the input.
        """
        if split_index < 0:
            raise ValueError(
                f"The split index must be non-negative; got {split_index}."
            )
        if split_axis >= 0:
            raise ValueError(f"The split axis must be negative; got {split_axis}.")

        self.split_index = split_index
        self.conditioner = conditioner
        self.bijector_fn = bijector
        self.swap = swap
        self.split_axis = split_axis
        self._is_constant_jacobian = False
        self._is_constant_log_det = False

    def _split(self, x: Array) -> tuple[Array, Array]:
        x1, x2 = jnp.split(x, [self.split_index], self.split_axis)
        if self.swap:
            x1, x2 = x2, x1
        return x1, x2

    def _recombine(self, x1: Array, x2: Array) -> Array:
        if self.swap:
            x1, x2 = x2, x1
        return jnp.concatenate([x1, x2], self.split_axis)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        x1, x2 = self._split(x)
        params = self.conditioner(x1)
        inner_bij = self.bijector_fn(params)

        y2, logdet2 = inner_bij.forward_and_log_det(x2)
        y = self._recombine(x1, y2)

        # If the inner logdet is elementwise, pad it. If it's a scalar, return it.
        if jnp.shape(logdet2) == jnp.shape(x2):
            logdet = self._recombine(jnp.zeros_like(x1), logdet2)
        else:
            logdet = logdet2

        return y, logdet

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        y1, y2 = self._split(y)
        params = self.conditioner(y1)
        inner_bij = self.bijector_fn(params)

        x2, logdet2 = inner_bij.inverse_and_log_det(y2)
        x = self._recombine(y1, x2)

        if jnp.shape(logdet2) == jnp.shape(x2):
            logdet = self._recombine(jnp.zeros_like(y1), logdet2)
        else:
            logdet = logdet2

        return x, logdet

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return False  # Too complex to guarantee identity for arbitrary neural nets
