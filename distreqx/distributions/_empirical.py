from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from ._distribution import AbstractDistribution


class AbstractEmpirical(AbstractDistribution, strict=True):
    """Abstract base class for Empirical and WeightedEmpirical distributions."""

    samples: eqx.AbstractVar[Array]
    atol: eqx.AbstractVar[Array]
    rtol: eqx.AbstractVar[Array]

    @property
    @abstractmethod
    def normalized_weights(self) -> Array:
        """Returns the weights of the samples, normalized to sum to 1."""
        raise NotImplementedError

    @property
    def event_shape(self) -> tuple:
        """Shape of the event (all dimensions after the sample count)."""
        return self.samples.shape[1:]

    def _event_axes(self) -> tuple:
        """Returns the axes corresponding to the event dimensions."""
        return tuple(range(1, self.samples.ndim))

    def _expand_weights(self) -> Array:
        """Expands weights for broadcasting over multivariate events."""
        return jnp.reshape(
            self.normalized_weights, (-1,) + (1,) * len(self.event_shape)
        )

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
        """Samples an event based on the normalized weights."""
        idx = jax.random.choice(
            key, a=self.samples.shape[0], p=self.normalized_weights
        )
        return self.samples[idx]

    def prob(self, value: PyTree[Array]) -> Array:
        """Calculates the probability of an event.
        
        For multivariate events, all dimensions must fall within the tolerance
        slack to be considered a match.
        """
        matches = jnp.abs(self.samples - value) <= self._slack(value)
        if self._event_axes():
            matches = jnp.all(matches, axis=self._event_axes())
        return jnp.sum(jnp.where(matches, self.normalized_weights, 0.0))

    def log_prob(self, value: PyTree[Array]) -> Array:
        """Calculates the log probability of an event."""
        return jnp.log(self.prob(value))

    def mean(self) -> Array:
        """Calculates the weighted empirical mean."""
        return jnp.sum(self.samples * self._expand_weights(), axis=0)

    def variance(self) -> Array:
        """Calculates the weighted empirical variance."""
        mu = self.mean()
        sq_diff = jnp.square(self.samples - mu)
        return jnp.sum(sq_diff * self._expand_weights(), axis=0)

    def stddev(self) -> Array:
        """Calculates the standard deviation."""
        return jnp.sqrt(self.variance())

    def entropy(self) -> Array:
        """Calculates the Shannon entropy (in nats)."""
        probs = jax.vmap(self.prob)(self.samples)
        # Avoid log(0) NaNs by masking out zero-probability samples
        safe_probs = jnp.where(probs > 0, probs, 1.0)
        return -jnp.sum(self.normalized_weights * jnp.log(safe_probs))

    def mode(self) -> Array:
        """Calculates the mode (sample with the highest combined probability)."""
        probs = jax.vmap(self.prob)(self.samples)
        return self.samples[jnp.argmax(probs)]

    def median(self) -> Array:
        """Calculates the median (50th percentile)."""
        if self.event_shape != ():
            raise NotImplementedError(
                "Median is intractable and undefined for multivariate events."
            )
        return self.icdf(jnp.array(0.5, dtype=self.samples.dtype))

    def cdf(self, value: PyTree[Array]) -> Array:
        """Evaluates the joint cumulative distribution function."""
        matches = self.samples <= (value + self._slack(value))
        if self._event_axes():
            matches = jnp.all(matches, axis=self._event_axes())
        return jnp.sum(jnp.where(matches, self.normalized_weights, 0.0))

    def log_cdf(self, value: PyTree[Array]) -> Array:
        """Evaluates the log cumulative distribution function."""
        return jnp.log(self.cdf(value))

    def survival_function(self, value: PyTree[Array]) -> Array:
        """Evaluates the survival function."""
        return 1.0 - self.cdf(value)

    def log_survival_function(self, value: PyTree[Array]) -> Array:
        """Evaluates the log of the survival function."""
        return jnp.log1p(-self.cdf(value))

    def icdf(self, value: PyTree[Array]) -> Array:
        """Evaluates the inverse cumulative distribution function (quantile)."""
        if self.event_shape != ():
            raise NotImplementedError(
                "Inverse CDF is intractable and undefined for multivariate events."
            )
            
        sort_idx = jnp.argsort(self.samples)
        sorted_samples = self.samples[sort_idx]
        sorted_weights = self.normalized_weights[sort_idx]
        
        cum_weights = jnp.cumsum(sorted_weights)
        idx = jnp.searchsorted(cum_weights, value)
        idx = jnp.clip(idx, 0, self.samples.shape[0] - 1)
        
        return jnp.where((value >= 0.0) & (value <= 1.0), sorted_samples[idx], jnp.nan)

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        raise NotImplementedError(
            "KL divergence is not well-defined for Empirical distributions."
        )


class Empirical(AbstractEmpirical, strict=True):
    """Scalar or Multivariate Empirical distribution."""

    samples: Array
    atol: Array
    rtol: Array

    def __init__(self, samples: Array, atol: float = 0.0, rtol: float = 0.0):
        self.samples = jnp.asarray(samples)
        self.atol = jnp.asarray(atol)
        self.rtol = jnp.asarray(rtol)

        if self.samples.ndim < 1:
            raise ValueError("Samples must have at least one dimension (the dataset axis).")
        if self.samples.shape[0] == 0:
            raise ValueError("The `samples` array cannot be empty.")

    @property
    def normalized_weights(self) -> Array:
        """Uniform weights: 1/N for each sample."""
        n = self.samples.shape[0]
        return jnp.ones(n, dtype=self.samples.dtype) / n


class WeightedEmpirical(AbstractEmpirical, strict=True):
    """Scalar or Multivariate Weighted Empirical distribution."""

    samples: Array
    weights: Array
    atol: Array
    rtol: Array

    def __init__(
        self, samples: Array, weights: Array, atol: float = 0.0, rtol: float = 0.0
    ):
        self.samples = jnp.asarray(samples)
        self.weights = jnp.asarray(weights)
        self.atol = jnp.asarray(atol)
        self.rtol = jnp.asarray(rtol)

        if self.samples.ndim < 1:
            raise ValueError("Samples must have at least one dimension (the dataset axis).")
        if self.weights.ndim != 1:
            raise ValueError("Weights must be a 1D array corresponding to the dataset axis.")
        if self.samples.shape[0] != self.weights.shape[0]:
            raise ValueError(
                f"Number of weights ({self.weights.shape[0]}) must match "
                f"number of samples ({self.samples.shape[0]})."
            )
        if jnp.any(self.weights < 0):
            raise ValueError("Weights cannot be negative.")

    @property
    def normalized_weights(self) -> Array:
        """Weights normalized by their sum."""
        return self.weights / jnp.sum(self.weights)