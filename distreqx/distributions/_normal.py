"""Normal distribution."""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ._distribution import AbstractProbDistribution


_half_log2pi = 0.5 * math.log(2 * math.pi)


class Normal(AbstractProbDistribution, strict=True):
    """Normal distribution with location `loc` and `scale` parameters."""

    loc: Array
    scale: Array

    def __init__(self, loc: Array, scale: Array):
        """Initializes a Normal distribution.

        **Arguments:**

        - `loc`: Mean of the distribution.
        - `scale`: Standard deviation of the distribution.
        """
        self.loc = jnp.array(loc)
        self.scale = jnp.array(scale)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of event of distribution samples."""
        return self.loc.shape

    def _sample_from_std_normal(self, key: PRNGKeyArray) -> Array:
        dtype = jnp.result_type(self.loc, self.scale)
        return jax.random.normal(key, shape=self.event_shape, dtype=dtype)

    def sample(self, key: PRNGKeyArray) -> Array:
        """See `Distribution.sample`."""
        rnd = self._sample_from_std_normal(key)
        return self.scale * rnd + self.loc

    def sample_and_log_prob(self, key: PRNGKeyArray) -> tuple[Array, Array]:
        """See `Distribution.sample_and_log_prob`."""
        rnd = self._sample_from_std_normal(key)
        samples = self.scale * rnd + self.loc
        log_prob = -0.5 * jnp.square(rnd) - _half_log2pi - jnp.log(self.scale)
        return samples, log_prob

    def log_prob(self, value: Array) -> Array:
        """See `Distribution.log_prob`."""
        log_unnormalized = -0.5 * jnp.square(self._standardize(value))
        log_normalization = _half_log2pi + jnp.log(self.scale)
        return log_unnormalized - log_normalization

    def cdf(self, value: Array) -> Array:
        """See `Distribution.cdf`."""
        return jax.scipy.special.ndtr(self._standardize(value))

    def log_cdf(self, value: Array) -> Array:
        """See `Distribution.log_cdf`."""
        return jax.scipy.special.log_ndtr(self._standardize(value))  # pyright: ignore[reportGeneralTypeIssues]

    def survival_function(self, value: Array) -> Array:
        """See `Distribution.survival_function`."""
        return jax.scipy.special.ndtr(-self._standardize(value))

    def log_survival_function(self, value: Array) -> Array:
        """See `Distribution.log_survival_function`."""
        return jax.scipy.special.log_ndtr(-self._standardize(value))  # pyright: ignore[reportGeneralTypeIssues]

    def _standardize(self, value: Array) -> Array:
        return (value - self.loc) / self.scale

    def entropy(self) -> Array:
        """Calculates the Shannon entropy (in nats)."""
        log_normalization = _half_log2pi + jnp.log(self.scale)
        entropy = 0.5 + log_normalization
        return entropy

    def mean(self) -> Array:
        """Calculates the mean."""
        return self.loc

    def variance(self) -> Array:
        """Calculates the variance."""
        return jnp.square(self.scale)

    def stddev(self) -> Array:
        """Calculates the standard deviation."""
        return self.scale

    def mode(self) -> Array:
        """Calculates the mode."""
        return self.mean()

    def median(self) -> Array:
        """Calculates the median."""
        return self.mean()

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
        diff_log_scale = jnp.log(dist1.scale) - jnp.log(dist2.scale)
        return (
            0.5 * jnp.square(dist1.loc / dist2.scale - dist2.loc / dist2.scale)
            + 0.5 * jnp.expm1(2.0 * diff_log_scale)
            - diff_log_scale
        )
