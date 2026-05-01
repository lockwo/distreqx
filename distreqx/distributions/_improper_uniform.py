import jax.numpy as jnp
from jaxtyping import Array, Key

from ._distribution import AbstractDistribution


class ImproperUniform(
    AbstractDistribution,
    strict=True,
):
    """Improper Uniform distribution over the entire real line.

    This distribution has an unnormalized probability density of 1 everywhere,
    meaning its `log_prob` evaluates to 0. As an improper distribution, it
    does not integrate to 1 and cannot be sampled from.
    """

    shape: tuple[int, ...] = ()

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.shape

    def sample(self, key: Key[Array, ""]) -> Array:
        """Sampling is not defined for improper distributions."""
        raise NotImplementedError("Cannot sample from an ImproperUniform distribution.")

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[Array, Array]:
        raise NotImplementedError("Cannot sample from an ImproperUniform distribution.")

    def log_prob(self, value: Array) -> Array:
        """Returns the unnormalized log probability (constant 0.0)."""
        return jnp.zeros_like(value)

    def prob(self, value: Array) -> Array:
        """Returns the unnormalized probability (constant 1.0)."""
        return jnp.ones_like(value)

    def entropy(self) -> Array:
        """Entropy of an improper uniform over the reals is infinite."""
        return jnp.full(self.shape, jnp.inf)

    def icdf(self, value: Array) -> Array:
        raise NotImplementedError("icdf is undefined for an improper distribution.")

    def log_cdf(self, value: Array) -> Array:
        raise NotImplementedError("log_cdf is undefined for an improper distribution.")

    def cdf(self, value: Array) -> Array:
        raise NotImplementedError("cdf is undefined for an improper distribution.")

    def survival_function(self, value: Array) -> Array:
        raise NotImplementedError(
            "survival_function is undefined for an improper distribution."
        )

    def log_survival_function(self, value: Array) -> Array:
        raise NotImplementedError(
            "log_survival_function is undefined for an improper distribution."
        )

    def mean(self) -> Array:
        raise NotImplementedError("Mean is undefined for an improper distribution.")

    def median(self) -> Array:
        raise NotImplementedError("Median is undefined for an improper distribution.")

    def variance(self) -> Array:
        raise NotImplementedError("Variance is undefined for an improper distribution.")

    def stddev(self) -> Array:
        raise NotImplementedError(
            "Standard deviation is undefined for an improper distribution."
        )

    def mode(self) -> Array:
        raise NotImplementedError("Mode is undefined for an improper distribution.")

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        raise NotImplementedError(
            "KL divergence is undefined for an improper distribution."
        )
