"""Categorical distribution."""

from typing import Optional, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ..utils.math import mul_exp, multiply_no_nan, normalize
from ._distribution import (
    AbstractSampleLogProbDistribution,
    AbstractSTDDistribution,
    AbstractSurivialDistribution,
)


class Categorical(
    AbstractSTDDistribution,
    AbstractSampleLogProbDistribution,
    AbstractSurivialDistribution,
    strict=True,
):
    """Categorical distribution over integers.

    The Categorical distribution is parameterized by either probabilities (`probs`) or
    unormalized log-probabilities (`logits`) of a set of `K` classes.
    It is defined over the integers `{0, 1, ..., K-1}`.
    """

    _logits: Union[Array, None]
    _probs: Union[Array, None]

    def __init__(self, logits: Optional[Array] = None, probs: Optional[Array] = None):
        """Initializes a Categorical distribution.

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
        return ()

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
            jnp.all(jnp.isfinite(self.probs), axis=-1),
            jnp.all(self.probs >= 0, axis=-1),
        )
        draws = jax.random.categorical(key=key, logits=self.logits, axis=-1).astype(
            "int8"
        )
        return jnp.where(is_valid, draws, jnp.ones_like(draws) * -1)

    def log_prob(self, value: Array) -> Array:
        """See `Distribution.log_prob`."""
        value_one_hot = jax.nn.one_hot(
            value, self.num_categories, dtype=self.logits.dtype
        )
        mask_outside_domain = jnp.logical_or(value < 0, value > self.num_categories - 1)
        return jnp.where(
            mask_outside_domain,
            -jnp.inf,
            jnp.sum(multiply_no_nan(self.logits, value_one_hot), axis=-1),
        )

    def prob(self, value: Array) -> Array:
        """See `Distribution.prob`."""
        value_one_hot = jax.nn.one_hot(
            value, self.num_categories, dtype=self.probs.dtype
        )
        return jnp.sum(multiply_no_nan(self.probs, value_one_hot), axis=-1)

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
        if self._logits is None:
            if self._probs is None:
                raise ValueError(
                    "_probs and _logits are None!"
                )  # TODO: useless but needed for pyright
            parameter = self.probs
        else:
            parameter = self.logits
        return jnp.argmax(parameter, axis=-1).astype("int8")

    def cdf(self, value: Array) -> Array:
        """See `Distribution.cdf`."""
        # For value < 0 the output should be zero because support = {0, ..., K-1}.
        should_be_zero = value < 0
        # For value >= K-1 the output should be one. Explicitly accounting for this
        # case addresses potential numerical issues that may arise when evaluating
        # derived methods (mainly, `log_survival_function`) for `value >= K-1`.
        should_be_one = value >= self.num_categories - 1
        # Will use value as an index below, so clip it to {0, ..., K-1}.
        value = jnp.clip(value, 0, self.num_categories - 1)
        value_one_hot = jax.nn.one_hot(
            value, self.num_categories, dtype=self.probs.dtype
        )
        cdf = jnp.sum(
            multiply_no_nan(jnp.cumsum(self.probs, axis=-1), value_one_hot), axis=-1
        )
        return jnp.where(should_be_zero, 0.0, jnp.where(should_be_one, 1.0, cdf))

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
        """Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible disteqx distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The KL divergence `KL(self || other_dist)`.
        """
        return _kl_divergence_categorical_categorical(self, other_dist)


def _kl_divergence_categorical_categorical(
    dist1: Categorical,
    dist2: Categorical,
    *unused_args,
    **unused_kwargs,
) -> Array:
    """Obtains the KL divergence `KL(dist1 || dist2)` between two Categoricals.

    The KL computation takes into account that `0 * log(0) = 0`; therefore,
    `dist1` may have zeros in its probability vector.

    **Arguments:**

        - `dist1`: A Categorical distribution.
        - `dist2`: A Categorical distribution.

    **Returns:**

    Batchwise `KL(dist1 || dist2)`.

    **Raises:**

    ValueError if the two distributions have different number of categories.
    """
    logits1 = dist1.logits
    logits2 = dist2.logits

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
