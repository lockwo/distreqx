"""One hot categorical distribution."""

from typing import Optional, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ..utils.math import mul_exp, multiply_no_nan, normalize
from ._distribution import (
    AbstractSampleLogProbDistribution,
    AbstractSTDDistribution,
    AbstractSurvivalDistribution,
)


class OneHotCategorical(
    AbstractSTDDistribution,
    AbstractSampleLogProbDistribution,
    AbstractSurvivalDistribution,
    strict=True,
):
    """OneHotCategorical distribution."""

    _logits: Union[Array, None]
    _probs: Union[Array, None]

    def __init__(self, logits: Optional[Array] = None, probs: Optional[Array] = None):
        """Initializes a OneHotCategorical distribution.

        **Arguments:**

        - `logits`: Logit transform of the probability of each category. Only one
            of `logits` or `probs` can be specified.
        - `probs`: Probability of each category. Only one of `logits` or `probs` can
            be specified.
        """
        if (logits is None) == (probs is None):
            raise ValueError(
                f"One and exactly one of `logits` and `probs` should be `None`, "
                f"but `logits` is {logits} and `probs` is {probs}."
            )
        if (not isinstance(logits, jax.Array)) and (not isinstance(probs, jax.Array)):
            raise ValueError("`logits` and `probs` are not jax arrays.")

        self._probs = None if probs is None else normalize(probs=probs)
        self._logits = None if logits is None else normalize(logits=logits)

    @property
    def event_shape(self) -> tuple:
        """Shape of event of distribution samples."""
        return (self.num_categories,)

    @property
    def logits(self) -> Array:
        """The logits for each event."""
        if self._logits is not None:
            return self._logits
        if self._probs is None:
            raise ValueError(
                "_probs and _logits are None!"
            )  # TODO: useless but needed for pyright
        return jnp.log(self._probs)

    @property
    def probs(self) -> Array:
        """The probabilities for each event."""
        if self._probs is not None:
            return self._probs
        if self._logits is None:
            raise ValueError(
                "_probs and _logits are None!"
            )  # TODO: useless but needed for pyright
        return jax.nn.softmax(self._logits, axis=-1)

    @property
    def num_categories(self) -> int:
        """Number of categories."""
        if self._probs is not None:
            return self._probs.shape[-1]
        if self._logits is None:
            raise ValueError(
                "_probs and _logits are None!"
            )  # TODO: useless but needed for pyright
        return self._logits.shape[-1]

    def sample(self, key: PRNGKeyArray) -> Array:
        """See `Distribution.sample`."""
        is_valid = jnp.logical_and(
            jnp.all(jnp.isfinite(self.probs), axis=-1, keepdims=True),
            jnp.all(self.probs >= 0, axis=-1, keepdims=True),
        )
        draws = jax.random.categorical(key=key, logits=self.logits, axis=-1)
        draws_one_hot = jax.nn.one_hot(draws, num_classes=self.num_categories)
        return jnp.where(
            is_valid, draws_one_hot, jnp.ones_like(draws_one_hot) * -1
        ).astype(jnp.int8)

    def log_prob(self, value: Array) -> Array:
        """See `Distribution.log_prob`."""
        return jnp.sum(multiply_no_nan(self.logits, value), axis=-1)

    def prob(self, value: Array) -> Array:
        """See `Distribution.prob`."""
        return jnp.sum(multiply_no_nan(self.probs, value), axis=-1)

    def entropy(self) -> Array:
        """See `Distribution.entropy`."""
        if self._logits is None:
            if self._probs is None:
                raise ValueError(
                    "_probs and _logits are None!"
                )  # TODO: useless but needed for pyright
            log_probs = jnp.log(self._probs)
        else:
            log_probs = jax.nn.log_softmax(self._logits)
        return -jnp.sum(mul_exp(log_probs, log_probs), axis=-1)

    def mode(self) -> Array:
        """See `Distribution.mode`."""
        preferences = self._probs if self._logits is None else self._logits
        assert preferences is not None
        greedy_index = jnp.argmax(preferences, axis=-1)
        return jax.nn.one_hot(greedy_index, self.num_categories)

    def cdf(self, value: Array) -> Array:
        """See `Distribution.cdf`."""
        return jnp.sum(multiply_no_nan(jnp.cumsum(self.probs, axis=-1), value), axis=-1)

    def log_cdf(self, value: Array) -> Array:
        """See `Distribution.log_cdf`."""
        return jnp.log(self.cdf(value))

    def median(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Obtains the KL divergence `KL(dist1 || dist2)` between two Categoricals.

        The KL computation takes into account that `0 * log(0) = 0`; therefore,
        `dist1` may have zeros in its probability vector.

        **Arguments:**

            - `other_dist`: A Categorical distribution.

        **Returns:**

        `KL(dist1 || dist2)`.

        **Raises:**

        ValueError if the two distributions have different number of categories.
        """
        if not isinstance(other_dist, OneHotCategorical):
            raise TypeError("Only valid KL for both categoricals.")
        logits1 = self.logits
        logits2 = other_dist.logits

        num_categories1 = logits1.shape[-1]
        num_categories2 = logits2.shape[-1]

        if num_categories1 != num_categories2:
            raise ValueError(
                f"Cannot obtain the KL between two Categorical distributions "
                f"with different number of categories: the first distribution has "
                f"{num_categories1} categories, while the second distribution has "
                f"{num_categories2} categories."
            )

        log_probs1 = jax.nn.log_softmax(logits1, axis=-1)
        log_probs2 = jax.nn.log_softmax(logits2, axis=-1)
        return jnp.sum(mul_exp(log_probs1 - log_probs2, log_probs1), axis=-1)
