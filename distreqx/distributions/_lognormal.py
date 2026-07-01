"""LogNormal distribution."""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from ._distribution import AbstractProbDistribution

_half_log2pi = 0.5 * math.log(2 * math.pi)


class LogNormal(AbstractProbDistribution):
    """
    LogNormal distribution parameterized by
    `loc` and `scale` of the underlying Normal.
    """

    loc: Array
    scale: Array

    def __init__(self, loc: Array, scale: Array):
        """Initializes a LogNormal distribution.

        **Arguments:**

        - `loc`: Mean of the underlying Normal distribution.
        - `scale`: Standard deviation of the underlying Normal distribution.
        """
        self.loc = jnp.array(loc)
        self.scale = jnp.array(scale)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of event of distribution samples."""
        return self.loc.shape

    @property
    def support(self) -> tuple[Array, Array]:
        """See `Distribution.support`."""
        dtype = jnp.result_type(self.loc, self.scale)
        return (jnp.array(0.0, dtype=dtype), jnp.array(jnp.inf, dtype=dtype))

    def _sample_from_std_normal(self, key: Key[Array, ""]) -> Array:
        dtype = jnp.result_type(self.loc, self.scale)
        return jax.random.normal(key, shape=self.event_shape, dtype=dtype)

    def sample(self, key: Key[Array, ""]) -> Array:
        """See `Distribution.sample`."""
        rnd = self._sample_from_std_normal(key)
        normal_samples = self.scale * rnd + self.loc
        return jnp.exp(normal_samples)

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[Array, Array]:
        """See `Distribution.sample_and_log_prob`."""
        rnd = self._sample_from_std_normal(key)
        normal_samples = self.scale * rnd + self.loc
        samples = jnp.exp(normal_samples)

        # Change of variables: log(P(Y=y)) = log(P(X=x)) - log(y), where x = log(y)
        normal_log_prob = -0.5 * jnp.square(rnd) - _half_log2pi - jnp.log(self.scale)
        log_prob = normal_log_prob - normal_samples
        return samples, log_prob

    def log_prob(self, value: Array) -> Array:
        """See `Distribution.log_prob`."""
        log_value = jnp.log(value)
        log_unnormalized = -0.5 * jnp.square(self._standardize(log_value))
        log_normalization = _half_log2pi + jnp.log(self.scale) + log_value
        return log_unnormalized - log_normalization

    def icdf(self, value: Array) -> Array:
        """See `Distribution.icdf`."""
        return jnp.exp(jax.scipy.special.ndtri(value) * self.scale + self.loc)

    def cdf(self, value: Array) -> Array:
        """See `Distribution.cdf`."""
        return jax.scipy.special.ndtr(self._standardize(jnp.log(value)))

    def log_cdf(self, value: Array) -> Array:
        """See `Distribution.log_cdf`."""
        return jax.scipy.special.log_ndtr(
            self._standardize(jnp.log(value))
        )  # pyright: ignore[reportGeneralTypeIssues]

    def survival_function(self, value: Array) -> Array:
        """See `Distribution.survival_function`."""
        return jax.scipy.special.ndtr(-self._standardize(jnp.log(value)))

    def log_survival_function(self, value: Array) -> Array:
        """See `Distribution.log_survival_function`."""
        return jax.scipy.special.log_ndtr(
            -self._standardize(jnp.log(value))
        )  # pyright: ignore[reportGeneralTypeIssues]

    def _standardize(self, value: Array) -> Array:
        return (value - self.loc) / self.scale

    def entropy(self) -> Array:
        """Calculates the Shannon entropy (in nats)."""
        log_normalization = _half_log2pi + jnp.log(self.scale)
        entropy = 0.5 + log_normalization + self.loc
        return entropy

    def mean(self) -> Array:
        """Calculates the mean."""
        return jnp.exp(self.loc + 0.5 * jnp.square(self.scale))

    def variance(self) -> Array:
        """Calculates the variance."""
        var_scale = jnp.square(self.scale)
        return jnp.expm1(var_scale) * jnp.exp(2.0 * self.loc + var_scale)

    def stddev(self) -> Array:
        """Calculates the standard deviation."""
        return jnp.sqrt(self.variance())

    def mode(self) -> Array:
        """Calculates the mode."""
        return jnp.exp(self.loc - jnp.square(self.scale))

    def median(self) -> Array:
        """Calculates the median."""
        return jnp.exp(self.loc)

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible disteqx LogNormal distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The KL divergence `KL(self || other_dist)`.
        """
        # KL divergence is invariant under strictly monotonic transformations.
        # Thus, KL(LogNormal(m1, s1) || LogNormal(m2, s2)) is identical to
        # KL(Normal(m1, s1) || Normal(m2, s2)).
        dist1 = self
        dist2 = other_dist
        diff_log_scale = jnp.log(dist1.scale) - jnp.log(dist2.scale)
        return (
            0.5 * jnp.square(dist1.loc / dist2.scale - dist2.loc / dist2.scale)
            + 0.5 * jnp.expm1(2.0 * diff_log_scale)
            - diff_log_scale
        )
