from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from ._distribution import AbstractDistribution


class Empirical(AbstractDistribution, strict=True):
    """Scalar Empirical distribution on the real line.
    
    Represents a distribution defined by a finite set of samples. The 
    probability of any given value is the proportion of times it appears 
    in the sample set.
    """

    samples: Array
    atol: Array
    rtol: Array

    def __init__(
        self, 
        samples: Array, 
        atol: float = 0.0, 
        rtol: float = 0.0
    ):
        """Initializes an Empirical distribution.

        **Arguments:**

        - `samples`: A 1D array of observations defining the distribution.
        - `atol`: Absolute tolerance for comparing closeness to samples.
        - `rtol`: Relative tolerance for comparing closeness to samples.
        """
        self.samples = jnp.asarray(samples)
        self.atol = jnp.asarray(atol)
        self.rtol = jnp.asarray(rtol)

        if self.samples.ndim != 1:
            raise ValueError(
                f"The parameter `samples` must be a 1D array for a scalar "
                f"event distribution, but got shape {self.samples.shape}."
            )
            
        if self.samples.shape[0] == 0:
            raise ValueError("The `samples` array cannot be empty.")

    @property
    def event_shape(self) -> tuple:
        """Shape of the event."""
        return ()

    def _slack(self, value: Array) -> Array:
        """Calculates the tolerance window around a specific value."""
        return jnp.where(
            self.rtol == 0,
            self.atol,
            self.atol + self.rtol * jnp.abs(value)
        )

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[Array, Array]:
        """Returns sample and its log prob."""
        sample = self.sample(key)
        return sample, self.log_prob(sample)

    def sample(self, key: Key[Array, ""]) -> Array:
        """Samples an event uniformly at random from the dataset."""
        return jax.random.choice(key, self.samples)

    def prob(self, value: PyTree[Array]) -> Array:
        """Calculates the empirical probability of an event."""
        # Check how many samples fall within the tolerance window of the value
        matches = jnp.abs(self.samples - value) <= self._slack(value)
        return jnp.mean(matches)

    def log_prob(self, value: PyTree[Array]) -> Array:
        """Calculates the log probability of an event."""
        return jnp.log(self.prob(value))

    def entropy(self) -> Array:
        """Calculates the Shannon entropy (in nats).
        
        Uses the identity: H = -1/N * sum(log(P(x_i))) over all samples, 
        which avoids dynamically sized arrays from jnp.unique.
        """
        # Map the prob function over all our samples
        probs = jax.vmap(self.prob)(self.samples)
        return -jnp.mean(jnp.log(probs))

    def mean(self) -> Array:
        """Calculates the empirical mean."""
        return jnp.mean(self.samples)

    def mode(self) -> Array:
        """Calculates the mode (most frequent sample)."""
        probs = jax.vmap(self.prob)(self.samples)
        return self.samples[jnp.argmax(probs)]

    def median(self) -> Array:
        """Calculates the median of the samples."""
        return jnp.median(self.samples)

    def variance(self) -> Array:
        """Calculates the empirical variance."""
        return jnp.var(self.samples)

    def stddev(self) -> Array:
        """Calculates the standard deviation."""
        return jnp.std(self.samples)

    def cdf(self, value: PyTree[Array]) -> Array:
        """Evaluates the cumulative distribution function at `value`."""
        # Proportion of samples less than or equal to the value (with slack)
        return jnp.mean(self.samples <= (value + self._slack(value)))

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
        """Evaluates the inverse cumulative distribution function (quantile)."""
        return jnp.quantile(self.samples, value)

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence to another distribution."""
        raise NotImplementedError(
            "KL divergence is not well-defined for Empirical distributions "
            "unless they share the exact same support."
        )