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


class MaskedCoupling(
    AbstractFowardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """Coupling bijector that uses a mask to specify which inputs are transformed."""

    mask: Array
    conditioner: Any
    bijector_fn: Callable[[Any], AbstractBijector] = eqx.field(static=True)

    _is_constant_jacobian: bool = eqx.field(static=True, init=False)
    _is_constant_log_det: bool = eqx.field(static=True, init=False)

    def __init__(
        self,
        mask: Array,
        conditioner: Any,
        bijector: Callable[[Any], AbstractBijector],
    ):
        """Initializes a MaskedCoupling bijector.

        **Arguments:**

        - `mask`: A boolean array where True indicates the element remains
            unchanged, and False indicates the element is transformed.
        - `conditioner`: A callable (usually an Equinox module) that takes the
            masked input and outputs parameters.
        - `bijector`: A callable that takes the parameters and returns an
            instantiated `distreqx` bijector.
        """
        self.mask = jnp.asarray(mask, dtype=bool)
        self.conditioner = conditioner
        self.bijector_fn = bijector
        self._is_constant_jacobian = False
        self._is_constant_log_det = False

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        masked_x = jnp.where(self.mask, x, 0.0)
        params = self.conditioner(masked_x)
        inner_bij = self.bijector_fn(params)

        y0, log_d = inner_bij.forward_and_log_det(x)
        y = jnp.array(jnp.where(self.mask, x, y0))
        logdet = jnp.array(jnp.where(self.mask, 0.0, log_d))

        return y, logdet

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        masked_y = jnp.where(self.mask, y, 0.0)
        params = self.conditioner(masked_y)
        inner_bij = self.bijector_fn(params)

        x0, log_d = inner_bij.inverse_and_log_det(y)
        x = jnp.array(jnp.where(self.mask, y, x0))
        logdet = jnp.array(jnp.where(self.mask, 0.0, log_d))

        return x, logdet

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return False
