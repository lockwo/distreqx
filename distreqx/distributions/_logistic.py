"""Logistic distribution."""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ._distribution import AbstractProbDistribution


class Logistic(AbstractProbDistribution, strict=True):
    """Logistic distribution with location `loc` and `scale` parameters."""

    loc: Float[Array, "..."]
    scale: Float[Array, "..."]

    def __init__(self, loc: Array, scale: Array):
        """Initializes a Logistic distribution.

        **Arguments:**

        - `loc`: Mean of the distribution.
        - `scale`: Spread of the distribution. Must be positive.
        """
        self.loc = jnp.array(loc)
        self.scale = jnp.array(scale)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of event of distribution samples."""
        return self.loc.shape

    def _standardize(self, value: Array) -> Array:
        return (value - self.loc) / self.scale

    def sample(self, key: Key[Array, ""]) -> Array:
        """See `Distribution.sample`."""
        dtype = jnp.result_type(self.loc, self.scale)
        uniform = jax.random.uniform(
            key,
            shape=self.event_shape,
            dtype=dtype,
            minval=jnp.finfo(dtype).tiny,
            maxval=1.0,
        )
        rnd = jnp.log(uniform) - jnp.log1p(-uniform)
        return self.scale * rnd + self.loc

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[Array, Array]:
        """See `Distribution.sample_and_log_prob`."""
        dtype = jnp.result_type(self.loc, self.scale)
        uniform = jax.random.uniform(
            key,
            shape=self.event_shape,
            dtype=dtype,
            minval=jnp.finfo(dtype).tiny,
            maxval=1.0,
        )
        rnd = jnp.log(uniform) - jnp.log1p(-uniform)
        samples = self.scale * rnd + self.loc
        log_prob = -rnd - 2.0 * jax.nn.softplus(-rnd) - jnp.log(self.scale)
        return samples, log_prob

    def log_prob(self, value: Array) -> Array:
        """See `Distribution.log_prob`."""
        z = self._standardize(value)
        return -z - 2.0 * jax.nn.softplus(-z) - jnp.log(self.scale)

    def icdf(self, value: Array) -> Array:
        """See `Distribution.icdf`."""
        return self.loc + self.scale * jax.scipy.special.logit(value)

    def cdf(self, value: Array) -> Array:
        """See `Distribution.cdf`."""
        return jax.nn.sigmoid(self._standardize(value))

    def log_cdf(self, value: Array) -> Array:
        """See `Distribution.log_cdf`."""
        return -jax.nn.softplus(-self._standardize(value))

    def survival_function(self, value: Array) -> Array:
        """See `Distribution.survival_function`."""
        return jax.nn.sigmoid(-self._standardize(value))

    def log_survival_function(self, value: Array) -> Array:
        """See `Distribution.log_survival_function`."""
        return -jax.nn.softplus(self._standardize(value))

    def entropy(self) -> Array:
        """Calculates the Shannon entropy (in nats)."""
        return 2.0 + jnp.log(self.scale)

    def mean(self) -> Array:
        """Calculates the mean."""
        return self.loc

    def variance(self) -> Array:
        """Calculates the variance."""
        return jnp.square(self.scale * math.pi) / 3.0

    def stddev(self) -> Array:
        """Calculates the standard deviation."""
        return self.scale * math.pi / math.sqrt(3.0)

    def mode(self) -> Array:
        """Calculates the mode."""
        return self.loc

    def median(self) -> Array:
        """Calculates the median."""
        return self.loc

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible distreqx distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The KL divergence `KL(self || other_dist)`.
        """
        raise NotImplementedError(
            "Logistic distribution does not have a closed-form KL divergence."
        )
