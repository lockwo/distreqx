"""Linear bijector."""
from ._bijector import AbstractBijector
from jaxtyping import Array


class AbstractLinearBijector(AbstractBijector):
    """Base class for linear bijectors.

    This class provides a base class for bijectors defined as `f(x) = Ax`,
    where `A` is a `DxD` matrix and `x` is a `D`-dimensional vector.
    """

    _event_dims: int

    def __init__(self, event_dims: int):
        """Initializes a `Linear` bijector.

        **Arguments:**
        - `event_dims`: the dimensionality of the vector `D`
        """
        super().__init__(event_ndims_in=1, is_constant_jacobian=True)

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

    @property
    def event_dims(self) -> int:
        """The dimensionality `D` of the event `x`."""
        return self._event_dims
