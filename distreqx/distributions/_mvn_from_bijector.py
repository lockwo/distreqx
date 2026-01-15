"""MultivariateNormalFromBijector distribution."""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..bijectors import (
    AbstractBijector,
    AbstractLinearBijector,
    Block,
    Chain,
    DiagLinear,
    Shift,
)
from ._distribution import AbstractDistribution
from ._independent import Independent
from ._normal import Normal
from ._transformed import AbstractTransformed


def _check_input_parameters_are_valid(
    scale: AbstractLinearBijector, loc: Array
) -> None:
    """Raises an error if `scale` and `loc` are not valid."""
    if loc.ndim < 1:
        raise ValueError("`loc` must have at least 1 dimension.")
    if scale.event_dims != loc.shape[-1]:
        raise ValueError(
            f"`scale` and `loc` have inconsistent dimensionality: "
            f"`scale.event_dims = {scale.event_dims} and "
            f"`loc.shape[-1] = {loc.shape[-1]}."
        )


class AbstractMultivariateNormalFromBijector(AbstractTransformed, strict=True):
    loc: eqx.AbstractVar[Array]
    scale: eqx.AbstractVar[AbstractLinearBijector]

    def mean(self) -> Array:
        """Calculates the mean."""
        return self.loc

    def median(self) -> Array:
        """Calculates the median."""
        return self.loc

    def mode(self) -> Array:
        """Calculates the mode."""
        return self.loc

    def covariance(self) -> Array:
        """Calculates the covariance matrix.

        **Returns:**
        - The covariance matrix, of shape `k x k`.
        """
        if isinstance(self.scale, DiagLinear):
            result = jnp.diag(self.variance())
        else:
            result = jax.vmap(self.scale.forward, in_axes=-2, out_axes=-2)(
                self.scale.matrix
            )
        return result

    def variance(self) -> Array:
        """Calculates the variance of all one-dimensional marginals."""
        if isinstance(self.scale, DiagLinear):
            result = jnp.square(self.scale.diag)
        else:
            scale_matrix = self.scale.matrix
            result = jnp.sum(scale_matrix * scale_matrix, axis=-1)
        return result

    def stddev(self) -> Array:
        """Calculates the standard deviation (the square root of the variance)."""
        if isinstance(self.scale, DiagLinear):
            result = jnp.abs(self.scale.diag)
        else:
            result = jnp.sqrt(self.variance())
        return result

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible disteqx distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The KL divergence `KL(self || other_dist)`.
        """
        dist1 = self
        dist2 = other_dist

        num_dims = dist1.event_shape[-1]

        # Calculation is based on:
        # https://github.com/tensorflow/probability/blob/v0.12.1/tensorflow_probability/python/distributions/mvn_linear_operator.py#L384
        # If C_1 = AA.T, C_2 = BB.T, then
        #   tr[inv(C_2) C_1] = ||inv(B) A||_F^2
        # where ||.||_F^2 is the squared Frobenius norm.
        diff_lob_abs_det = _log_abs_determinant(dist2) - _log_abs_determinant(dist1)
        if _has_diagonal_scale(dist1) and _has_diagonal_scale(dist2):
            # This avoids instantiating the full scale matrix when it is diagonal.
            b_inv_a = jnp.expand_dims(dist1.stddev() / dist2.stddev(), axis=-1)
        else:
            b_inv_a = _inv_scale_operator(dist2)(_scale_matrix(dist1))
        diff_mean_expanded = jnp.expand_dims(dist2.mean() - dist1.mean(), axis=-1)
        b_inv_diff_mean = _inv_scale_operator(dist2)(diff_mean_expanded)
        kl_divergence = diff_lob_abs_det + 0.5 * (
            -num_dims
            + _squared_frobenius_norm(b_inv_a)
            + _squared_frobenius_norm(b_inv_diff_mean)
        )
        return kl_divergence


class MultivariateNormalFromBijector(AbstractMultivariateNormalFromBijector):
    r"""Multivariate normal distribution on $\mathbb{R}^k$.

    The multivariate normal over $x$ is characterized by an invertible affine
    transformation $x = f(z) = Az + b$, where $z$ is a random variable that
    follows a standard multivariate normal on $\mathbb{R}^k$, i.e.,
    $p(z) = \mathcal{N}(0, I_k)$,
    $A$ is a $k \times k$ transformation matrix, and $b$ is a $k$-dimensional vector.

    The resulting PDF on $x$ is a multivariate normal, $p(x) = \mathcal{N}(b, C)$, where
    $C = AA^T$ is the covariance matrix.

    The transformation $x = f(z)$ must be specified by a linear scale bijector
    implementing the operation $Az$ and a shift (or location) term $b$.
    """

    loc: Array
    scale: AbstractLinearBijector
    distribution: AbstractDistribution
    bijector: AbstractBijector

    def __init__(self, loc: Array, scale: AbstractLinearBijector):
        """Initializes the distribution.

        **Arguments:**

        - `loc`: The term `b`, i.e., the mean of the multivariate normal distribution.
        - `scale`: The bijector specifying the linear transformation `A @ z`, as
            described in the class docstring.
        """
        _check_input_parameters_are_valid(scale, loc)

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

    def log_cdf(self, value: PyTree[Array]) -> PyTree[Array]:
        raise NotImplementedError

    def cdf(self, value: PyTree[Array]) -> PyTree[Array]:
        raise NotImplementedError


def _squared_frobenius_norm(x: Array) -> Array:
    """Computes the squared Frobenius norm of a matrix."""
    return jnp.sum(jnp.square(x), axis=[-2, -1])


def _log_abs_determinant(d: AbstractMultivariateNormalFromBijector) -> Array:
    """Obtains `log|det(A)|`."""
    return d.scale.forward_log_det_jacobian(jnp.zeros(d.event_shape, dtype=d.dtype))


def _inv_scale_operator(
    d: AbstractMultivariateNormalFromBijector,
) -> Callable[[Array], Array]:
    """Gets the operator that performs `A^-1 * x`."""
    return jax.vmap(d.scale.inverse, in_axes=-1, out_axes=-1)


def _scale_matrix(d: AbstractMultivariateNormalFromBijector) -> Array:
    """Gets the full scale matrix `A`."""
    return d.scale.matrix


def _has_diagonal_scale(d: AbstractMultivariateNormalFromBijector) -> bool:
    """Determines if the scale matrix `A` is diagonal."""
    if isinstance(d, AbstractMultivariateNormalFromBijector) and isinstance(
        d.scale, DiagLinear
    ):
        return True
    return False
