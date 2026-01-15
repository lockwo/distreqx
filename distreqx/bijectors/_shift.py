"""Shift bijector."""

from jax import numpy as jnp
from jaxtyping import Array

from ._bijector import AbstractBijector
from ._scalar_affine import AbstractScalarAffine


class Shift(AbstractScalarAffine, strict=True):
    r"""Bijector that translates its input elementwise.

    The bijector is defined as follows:

    - Forward: $y = x + \text{shift}$
    - Forward Jacobian determinant: $\log|\det J(x)| = 0$
    - Inverse: $x = y - \text{shift}$
    - Inverse Jacobian determinant: $\log|\det J(y)| = 0$

    where `shift` parameterizes the bijector.
    """

    shift: Array
    scale: Array
    inv_scale: Array
    log_scale: Array
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, shift: Array):
        """Initializes a `Shift` bijector.

        **Arguments:**

        - `shift`: the bijector's shift parameter.
        """
        self._is_constant_jacobian = True
        self._is_constant_log_det = True
        self.shift = shift
        self.scale = jnp.ones_like(shift)
        self.inv_scale = jnp.ones_like(shift)
        self.log_scale = jnp.zeros_like(shift)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Shift:
            return self.shift is other.shift
        return False
