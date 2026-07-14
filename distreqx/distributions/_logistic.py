"""Logistic distribution."""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ._distribution import AbstractProbDistribution


class Logistic(AbstractProbDistribution):
    r"""Logistic distribution with location $\mu$ and scale $s$.

    The probability density function is

    $$
    p(x) = \frac{\exp(-(x - \mu) / s)}
      {s \left(1 + \exp(-(x - \mu) / s)\right)^2},
    $$

    where $s > 0$.
    """

    loc: Float[Array, "..."]
    scale: Float[Array, "..."]

    def __init__(self, loc: Array, scale: Array):
        r"""Initializes a Logistic distribution.

        **Arguments:**

        - `loc`: Location parameter $\mu$, equal to the mean, median, and mode.
        - `scale`: Positive scale parameter $s$.
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
        return (jnp.array(-jnp.inf, dtype=dtype), jnp.array(jnp.inf, dtype=dtype))

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
        r"""Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible distreqx distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The divergence $D_{\mathrm{KL}}(P \parallel Q)$, where $P$ is this
        distribution and $Q$ is `other_dist`.
        """
        raise NotImplementedError(
            "Logistic distribution does not have a closed-form KL divergence."
        )
