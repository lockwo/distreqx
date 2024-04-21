"""Base class for distributions."""
from abc import abstractmethod
from typing import Tuple

import equinox as eqx
from jaxtyping import Array, Num, PRNGKeyArray, PyTree


class AbstractDistribution(eqx.Module, strict=True):
    """Base class for all distreqx distributions."""

    @abstractmethod
    def _sample_n(self, key: PRNGKeyArray, n: int) -> PyTree[Array]:
        """Returns `n` samples."""
        raise NotImplementedError

    def _sample_n_and_log_prob(
        self,
        key: PRNGKeyArray,
        n: int,
    ) -> Tuple[PyTree[Array], Num[Array, ""]]:
        """Returns `n` samples and their log probs.

        By default, it just calls `log_prob` on the generated samples. However, for
        many distributions it's more efficient to compute the log prob of samples
        than of arbitrary events (for example, there's no need to check that a
        sample is within the distribution's domain). If that's the case, a subclass
        may override this method with a more efficient implementation.

        **Arguments:**

        - `key`: PRNG key.
        - `n`: Number of samples to generate.

        **Returns:**

        A tuple of `n` samples and their log probs.
        """
        samples = self._sample_n(key, n)
        log_prob = self.log_prob(samples)
        return samples, log_prob

    @abstractmethod
    def log_prob(self, value: PyTree[Array]) -> Num[Array, ""]:
        """Calculates the log probability of an event.

        Args:
            value: An event.

        Returns:
            The log probability log P(value).
        """
        raise NotImplementedError
