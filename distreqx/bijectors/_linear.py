"""Linear bijector."""

import equinox as eqx
from jaxtyping import Array

from ._bijector import AbstractBijector


class AbstractLinearBijector(AbstractBijector, strict=True):
    """Base class for linear bijectors.

    This class provides a base class for bijectors defined as `f(x) = Ax`,
    where `A` is a `DxD` matrix and `x` is a `D`-dimensional vector.
    """

    event_dims: eqx.AbstractVar[int]

    @property
    def matrix(self) -> Array:
        """The matrix `A` of the transformation.

        To be optionally implemented in a subclass.

        **Returns:**

        - An array of shape `(D, D)`.
        """
        raise NotImplementedError(
            f"Linear bijector {self.name} does not implement `matrix`."
        )
