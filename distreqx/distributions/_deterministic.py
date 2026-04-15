import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from ._distribution import AbstractDistribution


class Deterministic(AbstractDistribution, strict=True):
    """Scalar Deterministic distribution on the real line.

    Represents a distribution that places all its probability mass on a single
    point (the `loc`).
    """

    loc: Array
    atol: Array
    rtol: Array

    def __init__(self, loc: Array | float, atol: float = 0.0, rtol: float = 0.0):
        """Initializes a Deterministic distribution.

        **Arguments:**

        - `loc`: The single point on which the distribution is supported.
        - `atol`: Absolute tolerance for comparing closeness to `loc` to account
            for floating-point inaccuracies.
        - `rtol`: Relative tolerance for comparing closeness to `loc`.
        """
        self.loc = jnp.asarray(loc)
        self.atol = jnp.asarray(atol)
        self.rtol = jnp.asarray(rtol)

        if self.loc.ndim != 0:
            raise ValueError(
                f"The parameter `loc` must be a scalar for a single event, "
                f"but got shape {self.loc.shape}."
            )

    @property
    def event_shape(self) -> tuple:
        """Shape of the event."""
        return ()

    @property
    def slack(self) -> Array:
        """Calculates the tolerance window around the location."""
        return jnp.where(
            self.rtol == 0, self.atol, self.atol + self.rtol * jnp.abs(self.loc)
        )

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[Array, Array]:
        """Returns sample and its log prob."""
        samples = self.sample(key)
        return samples, jnp.zeros_like(samples)

    def sample(self, key: Key[Array, ""]) -> Array:
        """Samples an event."""
        return self.loc

    def prob(self, value: PyTree[Array]) -> Array:
        """Calculates the probability of an event."""
        return jnp.where(jnp.abs(value - self.loc) <= self.slack, 1.0, 0.0)

    def log_prob(self, value: PyTree[Array]) -> Array:
        """Calculates the log probability of an event."""
        return jnp.log(self.prob(value))

    def entropy(self) -> Array:
        """Calculates the Shannon entropy (in nats)."""
        return jnp.zeros_like(self.loc)

    def mean(self) -> Array:
        """Calculates the mean."""
        return self.loc

    def mode(self) -> Array:
        """Calculates the mode."""
        return self.loc

    def median(self) -> Array:
        """Calculates the median."""
        return self.loc

    def variance(self) -> Array:
        """Calculates the variance."""
        return jnp.zeros_like(self.loc)

    def stddev(self) -> Array:
        """Calculates the standard deviation."""
        return jnp.zeros_like(self.loc)

    def cdf(self, value: PyTree[Array]) -> Array:
        """Evaluates the cumulative distribution function at `value`."""
        return jnp.where(value >= self.loc - self.slack, 1.0, 0.0)

    def log_cdf(self, value: PyTree[Array]) -> Array:
        """Evaluates the log cumulative distribution function at `value`."""
        return jnp.log(self.cdf(value))

    def survival_function(self, value: PyTree[Array]) -> Array:
        """Evaluates the survival function at `value`."""
        return 1.0 - self.cdf(value)

    def log_survival_function(self, value: PyTree[Array]) -> Array:
        """Evaluates the log of the survival function at `value`."""
        return jnp.log1p(-self.cdf(value))

    def icdf(self, value: PyTree[Array]) -> Array:
        """Evaluates the inverse cumulative distribution function at `value`."""
        return jnp.where((value >= 0.0) & (value <= 1.0), self.loc, jnp.nan)

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence to another distribution."""
        if not isinstance(other_dist, Deterministic):
            raise NotImplementedError(
                "KL divergence is only implemented for two Deterministic distributions."
            )

        slack2 = other_dist.atol + other_dist.rtol * jnp.abs(other_dist.loc)
        return -jnp.log(
            jnp.where(jnp.abs(self.loc - other_dist.loc) <= slack2, 1.0, 0.0)
        )
