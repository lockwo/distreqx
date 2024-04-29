"""Diagonal linear bijector."""

from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import AbstractBijector
from ._linear import AbstractLinearBijector
from .block import Block
from .scalar_affine import ScalarAffine


class DiagLinear(AbstractLinearBijector):
    """Linear bijector with a diagonal weight matrix.

    The bijector is defined as `f(x) = Ax` where `A` is a `DxD` diagonal matrix.
    Additional dimensions, if any, index batches.

    The Jacobian determinant is trivially computed by taking the product of the
    diagonal entries in `A`. The inverse transformation `x = f^{-1}(y)` is
    computed element-wise.

    The bijector is invertible if and only if the diagonal entries of `A` are all
    non-zero. It is the responsibility of the user to make sure that this is the
    case; the class will make no attempt to verify that the bijector is
    invertible.
    """

    _diag: Array
    _bijector: AbstractBijector

    def __init__(self, diag: Array):
        """Initializes the bijector.

        **Arguments:**

        - `diag`: a vector of length D, the diagonal of matrix `A`.
        """
        if diag.ndim != 1:
            raise ValueError("`diag` must have one dimension.")
        self._bijector = Block(
            ScalarAffine(shift=jnp.zeros_like(diag), scale=diag), ndims=diag.ndim
        )
        super().__init__(event_dims=diag.shape[-1])
        self._diag = diag

    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        return self._bijector.forward(x)

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        return self._bijector.inverse(y)

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        return self._bijector.forward_log_det_jacobian(x)

    def inverse_log_det_jacobian(self, y: Array) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        return self._bijector.inverse_log_det_jacobian(y)

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self._bijector.inverse_and_log_det(y)

    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self._bijector.forward_and_log_det(x)

    @property
    def diag(self) -> Array:
        """Vector of length D, the diagonal of matrix `A`."""
        return self._diag

    @property
    def matrix(self) -> Array:
        """The full matrix `A`."""
        # TODO: vectorize -> vmap
        return jnp.vectorize(jnp.diag, signature="(k)->(k,k)")(self.diag)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is DiagLinear:  # pylint: disable=unidiomatic-typecheck
            return self.diag is other.diag
        return False
