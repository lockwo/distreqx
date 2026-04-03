import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import (
    AbstractBijector,
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class R2ToComplex(
    AbstractFowardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """Maps a real array of shape (..., 2) to a complex array of shape (...)."""

    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = x[..., 0] + 1j * x[..., 1] and log|det J(f)(x)| = 0.0."""
        if x.shape[-1] != 2:
            raise ValueError(
                f"Expected the last dimension of x to be 2 for R2 coordinates, "
                f"but got {x.shape[-1]}."
            )
            
        y = x[..., 0] + 1j * x[..., 1]
        return y, jnp.zeros((), dtype=x.dtype)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = [Re(y), Im(y)] and log|det J(f^{-1})(y)| = 0.0."""
        if not jnp.iscomplexobj(y):
            raise ValueError(f"Expected input to inverse to be a complex array, got {y.dtype}.")
            
        x = jnp.stack([jnp.real(y), jnp.imag(y)], axis=-1)
        # The log-det uses the real dtype so type-checkers and JAX don't complain
        return x, jnp.zeros((), dtype=jnp.real(y).dtype)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is R2ToComplex