import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import (
    AbstractBijector,
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class Indexed(
    AbstractFowardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """Applies a bijector to a specific subset of indices of an input array."""

    bijector: AbstractBijector
    indices: Array

    _is_constant_jacobian: bool = eqx.field(init=False)
    _is_constant_log_det: bool = eqx.field(init=False)

    def __init__(self, bijector: AbstractBijector, indices: Array | list | tuple):
        """Initializes an Indexed bijector.

        **Arguments:**

        - `bijector`: The bijector to apply to the specified subset.
        - `indices`: An array of integer indices or boolean masks indicating
            which elements of the input should be transformed.
        """
        self.bijector = bijector
        self.indices = jnp.asarray(indices)

        self._is_constant_jacobian = self.bijector.is_constant_jacobian
        self._is_constant_log_det = self.bijector.is_constant_log_det

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Transforms x at the target indices and leaves the rest unchanged."""
        y_subset, log_det = self.bijector.forward_and_log_det(x[self.indices])
        y = x.at[self.indices].set(y_subset)
        return y, log_det

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """
        Inversely transforms y at the target indices and leaves the rest
        unchanged.
        """
        x_subset, log_det = self.bijector.inverse_and_log_det(y[self.indices])
        x = y.at[self.indices].set(x_subset)
        return x, log_det

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return bool(
            type(other) is Indexed
            and self.bijector.same_as(other.bijector)
            and jnp.array_equal(self.indices, other.indices)
        )
