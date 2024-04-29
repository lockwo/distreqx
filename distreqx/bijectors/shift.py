"""Shift bijector."""

from jaxtyping import Array

from ._bijector import AbstractBijector
from .scalar_affine import ScalarAffine


class Shift(ScalarAffine):
    """Bijector that translates its input elementwise.

    The bijector is defined as follows:

    - Forward: `y = x + shift`
    - Forward Jacobian determinant: `log|det J(x)| = 0`
    - Inverse: `x = y - shift`
    - Inverse Jacobian determinant: `log|det J(y)| = 0`

    where `shift` parameterizes the bijector.
    """

    def __init__(self, shift: Array):
        """Initializes a `Shift` bijector.

        Args:
          shift: the bijector's shift parameter. Can also be batched.
        """
        super().__init__(shift=shift)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Shift:  # pylint: disable=unidiomatic-typecheck
            return self.shift is other.shift
        return False
