"""Beta distribution."""

from typing import Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ..utils.math import log_beta
from ._distribution import (
    AbstractProbDistribution,
    AbstractSampleLogProbDistribution,
    AbstractSTDDistribution,
    AbstractSurvivalDistribution,
)


class Beta(
    AbstractSampleLogProbDistribution,
    AbstractSTDDistribution,
    AbstractProbDistribution,
    AbstractSurvivalDistribution,
    strict=True,
):
    """
    Beta distribution with parameters `alpha` and `beta`.

    The PDF of a Beta distributed random variable `X` is defined on the interval
    `0 <= X <= 1` and has the form:
    ```
    p(x; alpha, beta) = x ** {alpha - 1} * (1 - x) ** (beta - 1) / B(alpha, beta)
    ```
    where `B(alpha, beta)` is the beta function, and the `alpha, beta > 0` are the
    shape parameters.

    Note that the support of the distribution does not include `x = 0` or `x = 1`
    if `alpha < 1` or `beta < 1`, respectively.
    """

    alpha: Float[Array, "..."]
    beta: Float[Array, "..."]
    _log_normalization_constant: Float[Array, "..."]

    def __init__(
        self,
        alpha: Union[float, Float[Array, "..."]],
        beta: Union[float, Float[Array, "..."]],
    ):
        """Initializes a Beta distribution.

        **Arguments:**

        - `alpha`: Shape parameter `alpha` of the distribution. Must be positive.
        - `beta`: Shape parameter `beta` of the distribution. Must be positive.
        """
        super().__init__()
        self.alpha = jnp.array(alpha)
        self.beta = jnp.array(beta)
        self._log_normalization_constant = log_beta(self.alpha, self.beta)

    @property
    def event_shape(self) -> tuple:
        return ()

    def sample(self, key: Key[Array, ""]) -> Array:
        """See `Distribution.sample`."""
        dtype = jnp.result_type(self.alpha, self.beta)
        return jax.random.beta(key, a=self.alpha, b=self.beta, dtype=dtype)

    def log_prob(self, value: Array) -> Array:
        """See `Distribution.log_prob`."""
        result = (
            (self.alpha - 1.0) * jnp.log(value)
            + (self.beta - 1.0) * jnp.log(1.0 - value)
            - self._log_normalization_constant
        )
        return jnp.where(
            jnp.logical_or(
                jnp.logical_and(self.alpha == 1.0, value == 0.0),
                jnp.logical_and(self.beta == 1.0, value == 1.0),
            ),
            -self._log_normalization_constant,
            result,
        )

    def cdf(self, value: Array) -> Array:
        """See `Distribution.cdf`."""
        return jax.scipy.special.betainc(self.alpha, self.beta, value)

    def log_cdf(self, value: Array) -> Array:
        """See `Distribution.log_cdf`."""
        return jnp.log(self.cdf(value))

    def entropy(self) -> Array:
        """Calculates the Shannon entropy (in nats)."""
        return (
            self._log_normalization_constant
            - (self.alpha - 1.0) * jax.lax.digamma(self.alpha)
            - (self.beta - 1.0) * jax.lax.digamma(self.beta)
            + (self.alpha + self.beta - 2.0) * jax.lax.digamma(self.alpha + self.beta)
        )

    def mean(self) -> Array:
        """Calculates the mean."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> Array:
        """Calculates the variance."""
        sumalphabeta = self.alpha + self.beta
        return (
            self.alpha * self.beta / (jnp.square(sumalphabeta) * (sumalphabeta + 1.0))
        )

    def mode(self) -> Array:
        """Calculates the mode."""
        return jnp.where(
            jnp.logical_and(self.alpha > 1.0, self.beta > 1.0),
            (self.alpha - 1.0) / (self.alpha + self.beta - 2.0),
            jnp.where(
                jnp.logical_and(self.alpha <= 1.0, self.beta > 1.0),
                0.0,
                jnp.where(
                    jnp.logical_and(self.alpha > 1.0, self.beta <= 1.0), 1.0, jnp.nan
                ),
            ),
        )

    def median(self):
        raise NotImplementedError

    def kl_divergence(
        self,
        other_dist,
        *unused_args,
        **unused_kwargs,
    ) -> Array:
        """KL divergence KL(dist1 || dist2) between two Beta distributions.

        Args:
            dist1: A Beta distribution.
            dist2: A Beta distribution.

        Returns:
            `KL(dist1 || dist2)`.
        """
        alpha1, beta1 = self.alpha, self.beta
        alpha2, beta2 = other_dist.alpha, other_dist.beta
        t1 = log_beta(alpha2, beta2) - log_beta(alpha1, beta1)
        t2 = (alpha1 - alpha2) * jax.lax.digamma(alpha1)
        t3 = (beta1 - beta2) * jax.lax.digamma(beta1)
        t4 = (alpha2 - alpha1 + beta2 - beta1) * jax.lax.digamma(alpha1 + beta1)
        return t1 + t2 + t3 + t4
