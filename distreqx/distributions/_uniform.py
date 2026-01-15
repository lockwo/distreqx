"""Uniform distribution."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ._distribution import (
    AbstractSTDDistribution,
    AbstractSurvivalDistribution,
)


class Uniform(
    AbstractSTDDistribution,
    AbstractSurvivalDistribution,
    strict=True,
):
    """Uniform distribution with `low` and `high` parameters."""

    low: Float[Array, "..."]
    high: Float[Array, "..."]

    @property
    def range(self) -> Array:
        return self.high - self.low

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.low.shape

    def sample(self, key: Key[Array, ""]) -> Array:
        """See `Distribution.sample`."""
        uniform = jax.random.uniform(
            key=key,
            shape=self.low.shape,
            dtype=self.range.dtype,
            minval=0.0,
            maxval=1.0,
        )
        return self.low + self.range * uniform

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[Array, Array]:
        samples = self.sample(key)
        # broadcast this?
        log_prob = -jnp.log(self.range)
        # log_prob = jnp.broadcast_to(log_prob, samples.shape)
        return samples, log_prob

    def log_prob(self, value: Array) -> Array:
        """See `Distribution.log_prob`."""
        return jnp.log(self.prob(value))

    def prob(self, value: Array) -> Array:
        """See `Distribution.prob`."""
        return jnp.where(
            jnp.logical_or(value < self.low, value > self.high),
            jnp.zeros_like(value),
            jnp.ones_like(value) / self.range,
        )

    def cdf(self, value: Array) -> Array:
        """See `Distribution.cdf`."""
        ones = jnp.ones_like(self.range)
        zeros = jnp.zeros_like(ones)
        result_if_not_big = jnp.where(
            value < self.low, zeros, (value - self.low) / self.range
        )
        return jnp.where(value > self.high, ones, result_if_not_big)

    def log_cdf(self, value: Array) -> Array:
        """See `Distribution.log_cdf`."""
        return jnp.log(self.cdf(value))

    def entropy(self) -> Array:
        """See `Distribution.entropy`."""
        return jnp.log(self.range)

    def mean(self) -> Array:
        """See `Distribution.mean`."""
        return (self.low + self.high) / 2.0

    def median(self) -> Array:
        """See `Distribution.median`."""
        return self.mean()

    def variance(self) -> Array:
        """See `Distribution.variance`."""
        return jnp.square(self.range) / 12.0

    def mode(self) -> Array:
        """See `Distribution.probs`."""
        raise NotImplementedError

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible disteqx distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The KL divergence `KL(self || other_dist)`.
        """
        return jnp.where(
            jnp.logical_and(other_dist.low <= self.low, self.high <= other_dist.high),
            jnp.log(other_dist.high - other_dist.low) - jnp.log(self.high - self.low),
            jnp.inf,
        )
