"""MultivariateNormalDiag distribution."""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..bijectors import (
    AbstractBijector,
    AbstractLinearBijector,
    Block,
    Chain,
    DiagLinear,
    Shift,
)
from ._distribution import AbstractDistribution
from .independent import Independent
from .mvn_from_bijector import (
    _check_input_parameters_are_valid,
    AbstractMultivariateNormalFromBijector,
)
from .normal import Normal


def _check_parameters(loc: Optional[Array], scale_diag: Optional[Array]) -> None:
    """Checks that the `loc` and `scale_diag` parameters are correct."""
    if scale_diag is not None and not scale_diag.shape:
        raise ValueError(
            "If provided, argument `scale_diag` must have at least 1 dimension."
        )
    if loc is not None and not loc.shape:
        raise ValueError("If provided, argument `loc` must have at least 1 dimension.")
    if (
        loc is not None
        and scale_diag is not None
        and (loc.shape[-1] != scale_diag.shape[-1])
    ):
        raise ValueError(
            f"The last dimension of arguments `loc` and "
            f"`scale_diag` must coincide, but {loc.shape[-1]} != "
            f"{scale_diag.shape[-1]}."
        )


class MultivariateNormalDiag(AbstractMultivariateNormalFromBijector, strict=True):
    """Multivariate normal distribution on `R^k` with diagonal covariance."""

    _loc: Array
    _scale: AbstractLinearBijector
    _event_shape: tuple[int, ...]
    _distribution: AbstractDistribution
    _bijector: AbstractBijector
    _scale_diag: Array

    def __init__(self, loc: Optional[Array] = None, scale_diag: Optional[Array] = None):
        """Initializes a MultivariateNormalDiag distribution.

        **Arguments:**

        - `loc`: Mean vector of the distribution. If not specified, it defaults
            to zeros. At least one of `loc` and `scale_diag` must be specified.
        - `scale_diag`: Vector of standard deviations.  If not specified, it
            defaults to ones. At least one of `loc` and`scale_diag` must be specified.
        """
        _check_parameters(loc, scale_diag)

        if scale_diag is None and loc is not None:
            scale_diag = jnp.ones(loc.shape[-1], loc.dtype)
        elif loc is None and scale_diag is not None:
            loc = jnp.zeros(scale_diag.shape[-1], scale_diag.dtype)

        if loc is None:
            raise ValueError("loc is None")
        if scale_diag is None:
            raise ValueError("scale_diag is None")
        if scale_diag.ndim != 1:
            raise ValueError("scale_diag must be a vector!")

        scale = DiagLinear(scale_diag)
        _check_input_parameters_are_valid(scale, loc)

        # Build a standard multivariate Gaussian.
        std_mvn_dist = Independent(
            distribution=eqx.filter_vmap(Normal)(
                jnp.zeros_like(loc), jnp.ones_like(loc)
            ),
        )
        # Form the bijector `f(x) = Ax + b`.
        bijector = Chain([Block(Shift(loc), ndims=loc.ndim), scale])
        self._distribution = std_mvn_dist
        self._bijector = bijector
        self._scale = scale
        self._loc = loc
        self._event_shape = loc.shape[-1:]
        self._scale_diag = scale_diag

    @property
    def scale_diag(self) -> Array:
        """Scale of the distribution."""
        return jnp.broadcast_to(self._scale_diag, self.event_shape)

    def _standardize(self, value: Array) -> Array:
        return (value - self._loc) / self._scale_diag

    def cdf(self, value: Array) -> Array:
        """See `Distribution.cdf`."""
        return jnp.prod(jax.scipy.special.ndtr(self._standardize(value)), axis=-1)

    def log_cdf(self, value: Array) -> Array:
        """See `Distribution.log_cdf`."""
        # TODO: in normal and here we have a pyright ignore,
        # jax has a weird return value for log_ndtr
        return jnp.sum(jax.scipy.special.log_ndtr(self._standardize(value)), axis=-1)  # pyright: ignore[reportGeneralTypeIssues]
