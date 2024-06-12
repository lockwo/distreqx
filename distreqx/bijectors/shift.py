"""Shift bijector."""

from jax import numpy as jnp
from jaxtyping import Array

from ._bijector import AbstractBijector
from .scalar_affine import AbstractScalarAffine


class Shift(AbstractScalarAffine, strict=True):
    """Bijector that translates its input elementwise.

    The bijector is defined as follows:

    - Forward: `y = x + shift`
    - Forward Jacobian determinant: `log|det J(x)| = 0`
    - Inverse: `x = y - shift`
    - Inverse Jacobian determinant: `log|det J(y)| = 0`

    where `shift` parameterizes the bijector.
    """

    _shift: Array
    _scale: Array
    _inv_scale: Array
    _log_scale: Array
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, shift: Array):
        """Initializes a `Shift` bijector.

        **Arguments:**

        - `shift`: the bijector's shift parameter.
        """
        self._is_constant_jacobian = True
        self._is_constant_log_det = True
        self._shift = shift
        self._scale = jnp.ones_like(shift)
        self._inv_scale = jnp.ones_like(shift)
        self._log_scale = jnp.zeros_like(shift)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Shift:
            return self.shift is other.shift
        return False
