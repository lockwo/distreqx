"""Gamma distribution."""

from typing import Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ._distribution import (
    AbstractCDFDistribution,
    AbstractProbDistribution,
    AbstractSampleLogProbDistribution,
    AbstractSurvivalDistribution,
)


class Gamma(
    AbstractSampleLogProbDistribution,
    AbstractProbDistribution,
    AbstractSurvivalDistribution,
    AbstractCDFDistribution,
    strict=True,
):
    r"""Gamma distribution with parameters `concentration` and `rate`.

    The PDF of a Gamma distributed random variable $X$ is defined on the interval
    $X > 0$ and has the form:

    $$p(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}
        x^{\alpha - 1} e^{-\beta x}$$

    where $\alpha > 0$ is the concentration (shape) parameter and $\beta > 0$ is
    the rate (inverse scale) parameter.
    """

    concentration: Float[Array, "..."]
    rate: Float[Array, "..."]

    def __init__(
        self,
        concentration: Union[float, Float[Array, "..."]],
        rate: Union[float, Float[Array, "..."]],
    ):
        """Initializes a Gamma distribution.

        **Arguments:**

        - `concentration`: Concentration (shape) parameter. Must be positive.
        - `rate`: Rate (inverse scale) parameter. Must be positive.
        """
        super().__init__()
        self.concentration = jnp.asarray(concentration)
        self.rate = jnp.asarray(rate)

    @property
    def event_shape(self) -> tuple:
        return ()

    def sample(self, key: Key[Array, ""]) -> Array:
        """See `Distribution.sample`."""
        dtype = jnp.result_type(self.concentration, self.rate)
        rnd = jax.random.gamma(key, a=self.concentration, dtype=dtype)
        return rnd / self.rate

    def log_prob(self, value: Array) -> Array:
        """See `Distribution.log_prob`."""
        return (
            self.concentration * jnp.log(self.rate)
            + (self.concentration - 1.0) * jnp.log(value)
            - self.rate * value
            - jax.lax.lgamma(self.concentration)
        )

    def cdf(self, value: Array) -> Array:
        """See `Distribution.cdf`."""
        return jax.lax.igamma(self.concentration, self.rate * value)

    def log_cdf(self, value: Array) -> Array:
        """See `Distribution.log_cdf`."""
        return jnp.log(self.cdf(value))

    def entropy(self) -> Array:
        """Calculates the Shannon entropy (in nats)."""
        return (
            self.concentration
            - jnp.log(self.rate)
            + jax.lax.lgamma(self.concentration)
            + (1.0 - self.concentration) * jax.lax.digamma(self.concentration)
        )

    def mean(self) -> Array:
        """Calculates the mean."""
        return self.concentration / self.rate

    def variance(self) -> Array:
        """Calculates the variance."""
        return self.concentration / jnp.square(self.rate)

    def stddev(self) -> Array:
        """Calculates the standard deviation."""
        return jnp.sqrt(self.concentration) / self.rate

    def mode(self) -> Array:
        """Calculates the mode."""
        mode = (self.concentration - 1.0) / self.rate
        return jnp.where(self.concentration >= 1.0, mode, jnp.nan)

    def median(self):
        raise NotImplementedError

    def kl_divergence(
        self,
        other_dist,
        *unused_args,
        **unused_kwargs,
    ) -> Array:
        """KL divergence KL(self || other_dist) between two Gamma distributions.

        **Arguments:**

        - `other_dist`: A Gamma distribution.

        **Returns:**

        - `KL(self || other_dist)`.
        """
        t1 = other_dist.concentration * (jnp.log(self.rate) - jnp.log(other_dist.rate))
        t2 = jax.lax.lgamma(other_dist.concentration) - jax.lax.lgamma(
            self.concentration
        )
        t3 = (self.concentration - other_dist.concentration) * jax.lax.digamma(
            self.concentration
        )
        t4 = (other_dist.rate - self.rate) * (self.concentration / self.rate)
        return t1 + t2 + t3 + t4
