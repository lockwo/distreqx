"""MultivariateNormalDiag distribution."""

from typing import Optional
from ..bijectors import DiagLinear
from .mvn_from_bijector import MultivariateNormalFromBijector
from jaxtyping import Array
import jax
import jax.numpy as jnp


def _check_parameters(loc: Optional[Array], scale_diag: Optional[Array]) -> None:
    """Checks that the `loc` and `scale_diag` parameters are correct."""
    if scale_diag is not None and not scale_diag.shape:
        raise ValueError(
            "If provided, argument `scale_diag` must have at least " "1 dimension."
        )
    if loc is not None and not loc.shape:
        raise ValueError(
            "If provided, argument `loc` must have at least " "1 dimension."
        )
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


class MultivariateNormalDiag(MultivariateNormalFromBijector):
    """Multivariate normal distribution on `R^k` with diagonal covariance."""

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

        if scale_diag is None:
            scale_diag = jnp.ones(loc.shape[-1], loc.dtype)
        elif loc is None:
            loc = jnp.zeros(scale_diag.shape[-1], scale_diag.dtype)

        broadcasted_shapes = jnp.broadcast_shapes(loc.shape, scale_diag.shape)
        loc = jnp.expand_dims(loc, axis=list(range(len(broadcasted_shapes) - loc.ndim)))
        scale_diag = jnp.expand_dims(
            scale_diag, axis=list(range(len(broadcasted_shapes) - scale_diag.ndim))
        )

        scale = DiagLinear(scale_diag)
        super().__init__(loc=loc, scale=scale)
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
        return jnp.sum(jax.scipy.special.log_ndtr(self._standardize(value)), axis=-1)
