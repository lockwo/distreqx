"""Unconstrained affine bijector."""

import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import AbstractBijector


class UnconstrainedAffine(AbstractBijector, strict=True):
    """An unconstrained affine bijection.

    This bijector is a linear-plus-bias transformation `f(x) = Ax + b`, where `A`
    is a `D x D` square matrix and `b` is a `D`-dimensional vector.

    The bijector is invertible if and only if `A` is an invertible matrix. It is
    the responsibility of the user to make sure that this is the case; the class
    will make no attempt to verify that the bijector is invertible.

    The Jacobian determinant is equal to `det(A)`. The inverse is computed by
    solving the linear system `Ax = y - b`.

    WARNING: Both the determinant and the inverse cost `O(D^3)` to compute. Thus,
    this bijector is recommended only for small `D`.
    """

    matrix: Array
    bias: Array
    logdet: Array
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, matrix: Array, bias: Array):
        """Initializes an `UnconstrainedAffine` bijector.

        **Arguments:**

        - `matrix`: the matrix `A` in `Ax + b`. Must be square and invertible.
        - `bias`: the vector `b` in `Ax + b`.
        """
        if matrix.ndim != 2:
            raise ValueError(
                f"`matrix` must have exactly 2 dimensions, got {matrix.ndim}."
            )
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"`matrix` must be square; instead, it has shape {matrix.shape}."
            )
        if bias.ndim != 1:
            raise ValueError(f"`bias` must have exactly 1 dimension, got {bias.ndim}.")
        if matrix.shape[0] != bias.shape[0]:
            raise ValueError(
                f"`matrix` and `bias` have inconsistent shapes: `matrix` is "
                f"{matrix.shape}, `bias` is {bias.shape}."
            )
        self.matrix = matrix
        self.bias = bias
        self.logdet = jnp.linalg.slogdet(matrix)[1]
        self._is_constant_jacobian = True
        self._is_constant_log_det = True

    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        return self.matrix @ x + self.bias

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        return self.logdet

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        return jnp.linalg.solve(self.matrix, y - self.bias)

    def inverse_log_det_jacobian(self, y: Array) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        return -self.logdet

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is UnconstrainedAffine:  # pylint: disable=unidiomatic-typecheck
            return self.matrix is other.matrix and self.bias is other.bias
        return False
