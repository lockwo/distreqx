"""Base class for distributions."""
from abc import abstractmethod
from typing import Tuple

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from .._custom_types import EventT


class AbstractDistribution(eqx.Module, strict=True):
    """Base class for all distreqx distributions."""

    def sample_and_log_prob(
        self,
        key: PRNGKeyArray,
    ) -> Tuple[PyTree[Array], PyTree[Array]]:
        """Returns sample and its log prob.

        By default, it just calls `log_prob` on the generated samples. However, for
        many distributions it's more efficient to compute the log prob of samples
        than of arbitrary events (for example, there's no need to check that a
        sample is within the distribution's domain). If that's the case, a subclass
        may override this method with a more efficient implementation.

        **Arguments:**

        - `key`: PRNG key.

        **Returns:**

            A tuple of a sample and their log probs.
        """
        samples = self.sample(key)
        log_prob = self.log_prob(samples)
        return samples, log_prob

    @abstractmethod
    def log_prob(self, value: PyTree[Array]) -> PyTree[Array]:
        """Calculates the log probability of an event.

        **Arguments:**

        - `value`: An event.

        **Returns:**

        The log probability log P(value).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def event_shape(self) -> EventT:
        """Shape of event of distribution samples."""
        raise NotImplementedError

    @property
    def dtype(self) -> jnp.dtype:
        """Data type of a sample"""
        sample_spec = jax.eval_shape(self.sample, jax.random.PRNGKey(0))
        return jax.tree_util.tree_map(lambda x: x.dtype, sample_spec)

    @property
    def name(self) -> str:
        """Distribution name."""
        return type(self).__name__

    def prob(self, value: PyTree[Array]) -> PyTree[Array]:
        """Calculates the probability of an event.

        **Arguments:**

        - `value`: An event.

        **Returns:**

        The probability P(value).
        """
        return jnp.exp(self.log_prob(value))

    @abstractmethod
    def sample(self, key: PRNGKeyArray) -> PyTree[Array]:
        """Samples an event."""
        raise NotImplementedError

    def entropy(self) -> PyTree[Array]:
        """Calculates the Shannon entropy (in nats)."""
        raise NotImplementedError(
            f"Distribution `{self.name}` does not implement `entropy`."
        )

    def log_cdf(self, value: PyTree[Array]) -> PyTree[Array]:
        """Evaluates the log cumulative distribution function at
        `value` i.e. log P[X <= value]."""
        raise NotImplementedError(
            f"Distribution `{self.name}` does not implement `log_cdf`."
        )

    def cdf(self, value: PyTree[Array]) -> PyTree[Array]:
        """Evaluates the cumulative distribution function at `value`.

        **Arguments:**

        - `value`: An event.

        **Returns:**

        The CDF evaluated at value, i.e. P[X <= value].
        """
        return jnp.exp(self.log_cdf(value))

    def survival_function(self, value: PyTree[Array]) -> PyTree[Array]:
        """Evaluates the survival function at `value`.

        Note that by default we use a numerically not necessarily stable definition
        of the survival function in terms of the CDF.
        More stable definitions should be implemented in subclasses for
        distributions for which they exist.

        **Arguments:**

        - `value`: An event.

        **Returns:**

        The survival function evaluated at `value`, i.e. P[X > value]
        """
        return 1.0 - self.cdf(value)

    def log_survival_function(self, value: PyTree[Array]) -> PyTree[Array]:
        """Evaluates the log of the survival function at `value`.

        Note that by default we use a numerically not necessarily stable definition
        of the log of the survival function in terms of the CDF.
        More stable definitions should be implemented in subclasses for
        distributions for which they exist.

        **Arguments:**

        - `value`: An event.

        **Returns:**

        The log of the survival function evaluated at `value`, i.e.
            log P[X > value]
        """
        return jnp.log1p(-self.cdf(value))

    def mean(self) -> PyTree[Array]:
        """Calculates the mean."""
        raise NotImplementedError(
            f"Distribution `{self.name}` does not implement `mean`."
        )

    def median(self) -> PyTree[Array]:
        """Calculates the median."""
        raise NotImplementedError(
            f"Distribution `{self.name}` does not implement `median`."
        )

    def variance(self) -> PyTree[Array]:
        """Calculates the variance."""
        raise NotImplementedError(
            f"Distribution `{self.name}` does not implement `variance`."
        )

    def stddev(self) -> PyTree[Array]:
        """Calculates the standard deviation."""
        return jnp.sqrt(self.variance())

    def mode(self) -> PyTree[Array]:
        """Calculates the mode."""
        raise NotImplementedError(
            f"Distribution `{self.name}` does not implement `mode`."
        )

    def kl_divergence(self, other_dist, **kwargs) -> PyTree[Array]:
        """Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible distreqx Distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The KL divergence `KL(self || other_dist)`.
        """
        raise NotImplementedError(
            f"Distribution `{self.name}` does not implement `kl_divergence`."
        )

    def cross_entropy(self, other_dist, **kwargs) -> Array:
        """Calculates the cross entropy to another distribution.

        **Arguments:**

        - `other_dist`: A compatible distreqx Distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The cross entropy `H(self || other_dist)`.
        """
        return self.kl_divergence(other_dist, **kwargs) + self.entropy()
