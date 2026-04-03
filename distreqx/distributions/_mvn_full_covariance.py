from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array

from ._mvn_tri import MultivariateNormalTri


def _check_parameters(
    loc: Optional[Array], covariance_matrix: Optional[Array]
) -> None:
    """Checks that the inputs are correct for a single event."""
    if loc is None and covariance_matrix is None:
        raise ValueError(
            "At least one of `loc` and `covariance_matrix` must be specified."
        )

    if loc is not None and loc.ndim != 1:
        raise ValueError(
            f"The parameter `loc` must be a 1D vector, but "
            f"`loc.shape = {loc.shape}`."
        )

    if covariance_matrix is not None and covariance_matrix.ndim != 2:
        raise ValueError(
            f"The `covariance_matrix` must be a 2D matrix, but "
            f"`covariance_matrix.shape = {covariance_matrix.shape}`."
        )

    if covariance_matrix is not None and (
        covariance_matrix.shape[-1] != covariance_matrix.shape[-2]
    ):
        raise ValueError(
            f"The `covariance_matrix` must be a square matrix, but "
            f"`covariance_matrix.shape = {covariance_matrix.shape}`."
        )

    if loc is not None and covariance_matrix is not None:
        num_dims = loc.shape[-1]
        if covariance_matrix.shape[-1] != num_dims:
            raise ValueError(
                f"Shapes are not compatible: `loc.shape = {loc.shape}` and "
                f"`covariance_matrix.shape = {covariance_matrix.shape}`."
            )


class MultivariateNormalFullCovariance(MultivariateNormalTri, strict=True):
    r"""Multivariate normal distribution on $\mathbb{R}^k$.

    The `MultivariateNormalFullCovariance` distribution is parameterized by a
    $k$-length location (mean) vector $b$ and a covariance matrix $C$ of size
    $k \times k$ that must be positive definite and symmetric.

    !!! note

        This class makes no attempt to verify that the covariance matrix is
        positive definite or symmetric. The underlying Cholesky decomposition
        will simply fail if these conditions are not met.
    """

    _covariance_matrix: Array

    def __init__(
        self,
        loc: Optional[Array] = None,
        covariance_matrix: Optional[Array] = None,
    ):
        """Initializes a MultivariateNormalFullCovariance distribution.

        **Arguments:**

        - `loc`: Mean vector of the distribution of shape `k`.
            If not specified, it defaults to zeros.
        - `covariance_matrix`: The covariance matrix `C`. It must be a `k x k`
            symmetric positive definite matrix. If not specified, it defaults
            to the identity matrix.
        """
        _check_parameters(loc, covariance_matrix)

        if loc is not None:
            num_dims = loc.shape[-1]
        elif covariance_matrix is not None:
            num_dims = covariance_matrix.shape[-1]
        else:
            raise ValueError

        dtype = jnp.result_type(
            *[x for x in [loc, covariance_matrix] if x is not None]
        )

        if covariance_matrix is None:
            covariance_matrix = jnp.eye(num_dims, dtype=dtype)
            scale_tri = jnp.eye(num_dims, dtype=dtype)
        else:
            scale_tri = jnp.linalg.cholesky(covariance_matrix)

        # Let MultivariateNormalTri handle the bijector and std_mvn_dist logic
        super().__init__(loc=loc, scale_tri=scale_tri, is_lower=True)
        
        self._covariance_matrix = covariance_matrix

    @property
    def covariance_matrix(self) -> Array:
        """Covariance matrix `C`."""
        return self._covariance_matrix

    def covariance(self) -> Array:
        """Calculates the covariance matrix."""
        return self.covariance_matrix

    def variance(self) -> Array:
        """Calculates the variance of all one-dimensional marginals."""
        return jnp.diag(self.covariance_matrix)