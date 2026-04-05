import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import (
    AbstractBijector,
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class Transpose(
    AbstractFowardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """A bijector that transposes the dimensions of the input array."""

    permutation: tuple[int, ...] = eqx.field(static=True)
    _inverse_permutation: tuple[int, ...] = eqx.field(static=True)

    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def __init__(self, permutation: tuple[int, ...]):
        """Initializes a Transpose bijector.

        **Arguments:**

        - `permutation`: A tuple of integers representing
                        the desired permutation of the axes.
        """
        perm_length = len(permutation)

        # Validate that the permutation is a valid sequence of axes
        if sorted(permutation) != list(range(perm_length)):
            raise ValueError(
                f"Permutation must be a valid combination of axes 0 "
                f"to {perm_length - 1}, but got {permutation}."
            )

        self.permutation = tuple(permutation)

        # Compute and store the inverse permutation for O(1) retrieval
        # during the inverse pass
        inv_perm = [0] * perm_length
        for i, p in enumerate(self.permutation):
            inv_perm[p] = i
        self._inverse_permutation = tuple(inv_perm)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = transpose(x) and log|det J(f)(x)| = 0."""
        y = jnp.transpose(x, self.permutation)
        return y, jnp.zeros((), dtype=x.dtype)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = transpose(y) and log|det J(f^{-1})(y)| = 0."""
        x = jnp.transpose(y, self._inverse_permutation)
        return x, jnp.zeros((), dtype=y.dtype)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is Transpose and self.permutation == other.permutation
