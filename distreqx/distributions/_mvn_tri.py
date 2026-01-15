"""MultivariateNormalTri distribution."""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..bijectors import (
    AbstractBijector,
    AbstractLinearBijector,
    Block,
    Chain,
    DiagLinear,
    Shift,
    TriangularLinear,
)
from ._distribution import AbstractDistribution
from ._independent import Independent
from ._mvn_from_bijector import AbstractMultivariateNormalFromBijector
from ._normal import Normal


def _check_parameters(loc: Optional[Array], scale_tri: Optional[Array]) -> None:
    """Checks that the inputs are correct."""
    if loc is None and scale_tri is None:
        raise ValueError("At least one of `loc` and `scale_tri` must be specified.")

    if loc is not None and loc.ndim < 1:
        raise ValueError("The parameter `loc` must have at least one dimension.")

    if scale_tri is not None and scale_tri.ndim < 2:
        raise ValueError(
            f"The parameter `scale_tri` must have at least two dimensions, but "
            f"`scale_tri.shape = {scale_tri.shape}`."
        )

    if scale_tri is not None and scale_tri.shape[-1] != scale_tri.shape[-2]:
        raise ValueError(
            f"The parameter `scale_tri` must be a square matrix, but "
            f"`scale_tri.shape = {scale_tri.shape}`."
        )

    if loc is not None:
        num_dims = loc.shape[-1]
        if scale_tri is not None and scale_tri.shape[-1] != num_dims:
            raise ValueError(
                f"Shapes are not compatible: `loc.shape = {loc.shape}` and "
                f"`scale_tri.shape = {scale_tri.shape}`."
            )


class MultivariateNormalTri(AbstractMultivariateNormalFromBijector, strict=True):
    r"""Multivariate normal distribution on $\mathbb{R}^k$.

    The `MultivariateNormalTri` distribution is parameterized by a $k$-length
    location (mean) vector $b$ and a (lower or upper) triangular scale matrix $S$
    of size $k \times k$. The covariance matrix is $C = SS^T$.

    !!! note

        The `scale_tri` matrix must have non-zero diagonal elements for the
        distribution to be valid. This class does not verify this condition.
    """

    scale_tri: Array
    is_lower: bool
    loc: Array
    scale: AbstractLinearBijector
    distribution: AbstractDistribution
    bijector: AbstractBijector

    def __init__(
        self,
        loc: Optional[Array] = None,
        scale_tri: Optional[Array] = None,
        is_lower: bool = True,
    ):
        """Initializes a MultivariateNormalTri distribution.

        **Arguments:**

        - `loc`: Mean vector of the distribution of shape `k`.
            If not specified, it defaults to zeros.
        - `scale_tri`: The scale matrix `S`. It must be a `k x k` triangular matrix.
            If `scale_tri` is not triangular, the entries above or below the main
            diagonal will be ignored. The parameter `is_lower` specifies if `scale_tri`
            is lower or upper triangular. It is the responsibility of the user to make
            sure that `scale_tri` only contains non-zero elements in its diagonal;
            this class makes no attempt to verify that. If `scale_tri` is not specified,
            it defaults to the identity.
        - `is_lower`: Indicates if `scale_tri` is lower (if True) or upper (if False)
            triangular.
        """
        _check_parameters(loc, scale_tri)

        if loc is not None:
            num_dims = loc.shape[-1]
        elif scale_tri is not None:
            num_dims = scale_tri.shape[-1]
        else:
            raise ValueError

        dtype = jnp.result_type(*[x for x in [loc, scale_tri] if x is not None])

        if loc is None:
            loc = jnp.zeros((num_dims,), dtype=dtype)

        if scale_tri is None:
            scale_tri_val = jnp.eye(num_dims, dtype=dtype)
            scale = DiagLinear(diag=jnp.ones(loc.shape[-1:], dtype=dtype))
        else:
            tri_fn = jnp.tril if is_lower else jnp.triu
            scale_tri_val = tri_fn(scale_tri)
            scale = TriangularLinear(matrix=scale_tri_val, is_lower=is_lower)

        # Build a standard multivariate Gaussian.
        std_mvn_dist = Independent(
            distribution=eqx.filter_vmap(Normal)(
                jnp.zeros_like(loc), jnp.ones_like(loc)
            ),
        )
        # Form the bijector `f(x) = Ax + b`.
        bijector = Chain([Block(Shift(loc), ndims=loc.ndim), scale])
        self.distribution = std_mvn_dist
        self.bijector = bijector
        self.scale = scale
        self.loc = loc
        self.scale_tri = scale_tri_val
        self.is_lower = is_lower

    def log_cdf(self, value: PyTree[Array]) -> PyTree[Array]:
        raise NotImplementedError

    def cdf(self, value: PyTree[Array]) -> PyTree[Array]:
        raise NotImplementedError
