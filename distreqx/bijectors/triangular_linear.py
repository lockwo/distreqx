"""Triangular linear bijector."""

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import AbstractBijector
from ._linear import AbstractLinearBijector


def _triangular_logdet(matrix: Array) -> Array:
    """Computes the log absolute determinant of a triangular matrix."""
    return jnp.sum(jnp.log(jnp.abs(jnp.diag(matrix))))


class TriangularLinear(AbstractLinearBijector, strict=True):
    """A linear bijector whose weight matrix is triangular.

    The bijector is defined as `f(x) = Ax` where `A` is a DxD triangular matrix.

    The Jacobian determinant can be computed in O(D) as follows:

    log|det J(x)| = log|det A| = sum(log|diag(A)|)

    The inverse is computed in O(D^2) by solving the triangular system `Ax = y`.

    The bijector is invertible if and only if all diagonal elements of `A` are
    non-zero. It is the responsibility of the user to make sure that this is the
    case; the class will make no attempt to verify that the bijector is
    invertible.
    """

    _matrix: Array
    _is_lower: bool
    _is_constant_jacobian: bool
    _is_constant_log_det: bool
    _event_dims: int

    def __init__(self, matrix: Array, is_lower: bool = True):
        """Initializes a `TriangularLinear` bijector.

        **Arguments:**

        - `matrix`: a square matrix whose triangular part defines `A`. Can also be a
            batch of matrices. Whether `A` is the lower or upper triangular part of
            `matrix` is determined by `is_lower`.
        - `is_lower`: if True, `A` is set to the lower triangular part of `matrix`. If
            False, `A` is set to the upper triangular part of `matrix`.
        """
        self._is_constant_jacobian = True
        self._is_constant_log_det = True
        if matrix.ndim < 2:
            raise ValueError(
                f"`matrix` must have at least 2 dimensions, got {matrix.ndim}."
            )
        if matrix.shape[-2] != matrix.shape[-1]:
            raise ValueError(
                f"`matrix` must be square; instead, it has shape {matrix.shape[-2:]}."
            )
        self._event_dims = matrix.shape[-1]
        self._matrix = jnp.tril(matrix) if is_lower else jnp.triu(matrix)
        self._is_lower = is_lower

    @property
    def matrix(self) -> Array:
        """The triangular matrix `A` of the transformation."""
        return self._matrix

    @property
    def is_lower(self) -> bool:
        """True if `A` is lower triangular, False if upper triangular."""
        return self._is_lower

    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        return self._matrix @ x

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        triangular_logdet = jax.vmap(_triangular_logdet)
        return triangular_logdet(jnp.expand_dims(self._matrix, axis=0))[0]

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        return jax.scipy.linalg.solve_triangular(self._matrix, y, lower=self.is_lower)

    def inverse_log_det_jacobian(self, y: Array) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        return -self.forward_log_det_jacobian(y)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is TriangularLinear:  # pylint: disable=unidiomatic-typecheck
            return all(
                (
                    self.matrix is other.matrix,
                    self.is_lower is other.is_lower,
                )
            )
        return False
