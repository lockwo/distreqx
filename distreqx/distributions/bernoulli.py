"""Bernoulli distribution."""

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ..utils.math import multiply_no_nan
from ._distribution import AbstractDistribution


class Bernoulli(AbstractDistribution):
    """Bernoulli distribution of shape dims.

    Bernoulli distribution with parameter `probs`, the probability of outcome `1`.
    """

    _logits: Union[Array, None]
    _probs: Union[Array, None]

    def __init__(
        self,
        logits: Optional[Array] = None,
        probs: Optional[Array] = None,
    ):
        """Initializes a Bernoulli distribution.

        **Arguments:**

        - `logits`: Logit transform of the probability of a `1` event (`0` otherwise),
            i.e. `probs = sigmoid(logits)`. Only one of `logits` or `probs` can be
            specified.
        - `probs`: Probability of a `1` event (`0` otherwise). Only one of `logits` or
            `probs` can be specified.
        """
        # Validate arguments.
        if (logits is None) == (probs is None):
            raise ValueError(
                f"One and exactly one of `logits` and `probs` should be `None`, "
                f"but `logits` is {logits} and `probs` is {probs}."
            )
        if (not isinstance(logits, jax.Array)) and (not isinstance(probs, jax.Array)):
            raise ValueError("`logits` and `probs` are not jax arrays.")
        # Parameters of the distribution.
        self._probs = None if probs is None else probs
        self._logits = None if logits is None else logits

    @property
    def logits(self) -> Array:
        """The logits of a `1` event."""
        if self._logits is not None:
            return self._logits
        if self._probs is None:
            raise ValueError("_probs is None!")
        return jnp.log(self._probs) - jnp.log(1 - self._probs)

    @property
    def probs(self) -> Array:
        """The probabilities of a `1` event.."""
        if self._probs is not None:
            return self._probs
        if self._logits is None:
            raise ValueError("_logits is None!")
        return jax.nn.sigmoid(self._logits)

    @property
    def event_shape(self) -> Tuple[int]:
        return self.prob.shape

    def _log_probs_parameter(self) -> Tuple[Array, Array]:
        if self._logits is None:
            if self._probs is None:
                raise ValueError("_probs is None!")
            return (jnp.log1p(-1.0 * self._probs), jnp.log(self._probs))
        if self._logits is None:
            raise ValueError("_logits is None!")
        return (-jax.nn.softplus(self._logits), -jax.nn.softplus(-1.0 * self._logits))

    def sample(self, key: PRNGKeyArray) -> Array:
        """See `Distribution.sample`."""
        probs = self.probs
        return jax.random.bernoulli(key=key, p=probs).astype("int8")

    def log_prob(self, value: Array) -> Array:
        """See `Distribution.log_prob`."""
        log_probs0, log_probs1 = self._log_probs_parameter()
        return multiply_no_nan(log_probs0, 1 - value) + multiply_no_nan(
            log_probs1, value
        )

    def prob(self, value: Array) -> Array:
        """See `Distribution.prob`."""
        probs1 = self.probs
        probs0 = 1 - probs1
        return multiply_no_nan(probs0, 1 - value) + multiply_no_nan(probs1, value)

    def cdf(self, value: Array) -> Array:
        """See `Distribution.cdf`."""
        # For value < 0 the output should be zero because support = {0, 1}.
        return jnp.where(
            value < 0,
            jnp.array(0.0, dtype=self.probs.dtype),
            jnp.where(
                value >= 1, jnp.array(1.0, dtype=self.probs.dtype), 1 - self.probs
            ),
        )

    def log_cdf(self, value: Array) -> Array:
        """See `Distribution.log_cdf`."""
        return jnp.log(self.cdf(value))

    def entropy(self) -> Array:
        """See `Distribution.entropy`."""
        (probs0, probs1, log_probs0, log_probs1) = _probs_and_log_probs(self)
        return -1.0 * (
            multiply_no_nan(log_probs0, probs0) + multiply_no_nan(log_probs1, probs1)
        )

    def mean(self) -> Array:
        """See `Distribution.mean`."""
        return self.probs

    def variance(self) -> Array:
        """See `Distribution.variance`."""
        return (1 - self.probs) * self.probs

    def mode(self) -> Array:
        """See `Distribution.probs`."""
        return (self.probs > 0.5).astype("int8")

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible disteqx distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The KL divergence `KL(self || other_dist)`.
        """
        return _kl_divergence_bernoulli_bernoulli(self, other_dist)


def _probs_and_log_probs(
    dist: Bernoulli,
) -> Tuple[
    Array,
    Array,
    Array,
    Array,
]:
    """Calculates both `probs` and `log_probs`."""
    if dist._logits is None:
        if dist._probs is None:
            raise ValueError("_probs is None!")
        probs0 = 1.0 - dist._probs
        probs1 = 1.0 - probs0
        log_probs0 = jnp.log1p(-1.0 * dist._probs)
        log_probs1 = jnp.log(dist._probs)
    else:
        if dist._logits is None:
            raise ValueError("_logits is None!")
        probs0 = jax.nn.sigmoid(-1.0 * dist._logits)
        probs1 = jax.nn.sigmoid(dist._logits)
        log_probs0 = -jax.nn.softplus(dist._logits)
        log_probs1 = -jax.nn.softplus(-1.0 * dist._logits)
    return probs0, probs1, log_probs0, log_probs1


def _kl_divergence_bernoulli_bernoulli(
    dist1: Bernoulli,
    dist2: Bernoulli,
    *unused_args,
    **unused_kwargs,
) -> Array:
    """KL divergence `KL(dist1 || dist2)` between two Bernoulli distributions."""
    one_minus_p1, p1, log_one_minus_p1, log_p1 = _probs_and_log_probs(dist1)
    _, _, log_one_minus_p2, log_p2 = _probs_and_log_probs(dist2)
    # KL[a || b] = Pa * Log[Pa / Pb] + (1 - Pa) * Log[(1 - Pa) / (1 - Pb)]
    # Multiply each factor individually to avoid Inf - Inf
    return (
        multiply_no_nan(log_p1, p1)
        - multiply_no_nan(log_p2, p1)
        + multiply_no_nan(log_one_minus_p1, one_minus_p1)
        - multiply_no_nan(log_one_minus_p2, one_minus_p1)
    )
