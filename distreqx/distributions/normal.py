"""Normal distribution."""

import math
from typing import Tuple, Union
from jaxtyping import PRNGKeyArray
from ._distribution import AbstractDistribution
from .._custom_types import RealScalarLike
import jax
import jax.numpy as jnp


_half_log2pi = 0.5 * math.log(2 * math.pi)


class Normal(AbstractDistribution):
    """Normal distribution with location `loc` and `scale` parameters."""

    _loc: RealScalarLike
    _scale: RealScalarLike

    def __init__(self, loc: RealScalarLike, scale: RealScalarLike):
        """Initializes a Normal distribution.

        **Arguments:**

        - `loc`: Mean of the distribution.
        - `scale`: Standard deviation of the distribution.
        """
        self._loc = loc
        self._scale = scale

    @property
    def event_shape(self) -> Tuple[int, ...]:
        """Shape of event of distribution samples."""
        return self._loc.shape

    @property
    def loc(self) -> RealScalarLike:
        """Mean of the distribution."""
        return self._loc

    @property
    def scale(self) -> RealScalarLike:
        """Scale of the distribution."""
        return self._scale

    def _sample_from_std_normal(self, key: PRNGKeyArray) -> RealScalarLike:
        dtype = jnp.result_type(self._loc, self._scale)
        return jax.random.normal(key, shape=self.event_shape, dtype=dtype)

    def sample(self, key: PRNGKeyArray) -> RealScalarLike:
        """See `Distribution.sample`."""
        rnd = self._sample_from_std_normal(key)
        return self._scale * rnd + self._loc

    def sample_and_log_prob(
        self, key: PRNGKeyArray
    ) -> Tuple[RealScalarLike, RealScalarLike]:
        """See `Distribution.sample_and_log_prob`."""
        rnd = self._sample_from_std_normal(key)
        samples = self._scale * rnd + self._loc
        log_prob = -0.5 * jnp.square(rnd) - _half_log2pi - jnp.log(self._scale)
        return samples, log_prob

    def log_prob(self, value: RealScalarLike) -> RealScalarLike:
        """See `Distribution.log_prob`."""
        log_unnormalized = -0.5 * jnp.square(self._standardize(value))
        log_normalization = _half_log2pi + jnp.log(self._scale)
        return log_unnormalized - log_normalization

    def cdf(self, value: RealScalarLike) -> RealScalarLike:
        """See `Distribution.cdf`."""
        return jax.scipy.special.ndtr(self._standardize(value))

    def log_cdf(self, value: RealScalarLike) -> RealScalarLike:
        """See `Distribution.log_cdf`."""
        return jax.scipy.special.log_ndtr(self._standardize(value))

    def survival_function(self, value: RealScalarLike) -> RealScalarLike:
        """See `Distribution.survival_function`."""
        return jax.scipy.special.ndtr(-self._standardize(value))

    def log_survival_function(self, value: RealScalarLike) -> RealScalarLike:
        """See `Distribution.log_survival_function`."""
        return jax.scipy.special.log_ndtr(-self._standardize(value))

    def _standardize(self, value: RealScalarLike) -> RealScalarLike:
        return (value - self._loc) / self._scale

    def entropy(self) -> RealScalarLike:
        """Calculates the Shannon entropy (in nats)."""
        log_normalization = _half_log2pi + jnp.log(self.scale)
        entropy = 0.5 + log_normalization
        return entropy

    def mean(self) -> RealScalarLike:
        """Calculates the mean."""
        return self.loc

    def variance(self) -> RealScalarLike:
        """Calculates the variance."""
        return jnp.square(self.scale)

    def stddev(self) -> RealScalarLike:
        """Calculates the standard deviation."""
        return self.scale

    def mode(self) -> RealScalarLike:
        """Calculates the mode."""
        return self.mean()

    def median(self) -> RealScalarLike:
        """Calculates the median."""
        return self.mean()

    def kl_divergence(self, other_dist, **kwargs) -> RealScalarLike:
        """Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible disteqx distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The KL divergence `KL(self || other_dist)`.
        """
        return _kl_divergence_normal_normal(self, other_dist)


def _kl_divergence_normal_normal(
    dist1: Normal,
    dist2: Normal,
    *unused_args,
    **unused_kwargs,
) -> RealScalarLike:
    """Obtain the batched KL divergence KL(dist1 || dist2) between two Normals.

    **Arguments:**
    - `dist1`: A Normal distribution.
    - `dist2`: A Normal distribution.

    **Returns:**
    - `KL(dist1 || dist2)`.
    """
    diff_log_scale = jnp.log(dist1.scale) - jnp.log(dist2.scale)
    return (
        0.5 * jnp.square(dist1.loc / dist2.scale - dist2.loc / dist2.scale)
        + 0.5 * jnp.expm1(2.0 * diff_log_scale)
        - diff_log_scale
    )
