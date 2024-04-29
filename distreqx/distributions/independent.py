"""Independent distribution."""

import operator
from typing import Tuple

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree

from .._custom_types import EventT
from ._distribution import AbstractDistribution


def _reduce_helper(pytree: PyTree) -> Array:
    sum_over_leaves = jtu.tree_map(jnp.sum, pytree)
    return jtu.tree_reduce(operator.add, sum_over_leaves)


class Independent(AbstractDistribution):
    """Independent distribution obtained from child distributions."""

    _distribution: AbstractDistribution

    def __init__(
        self,
        distribution: AbstractDistribution,
    ):
        """Initializes an Independent distribution.

        **Arguments:**

        - `distribution`: Base distribution instance.
        """
        self._distribution = distribution

    @property
    def event_shape(self) -> EventT:
        """Shape of event of distribution samples."""
        return self._distribution.event_shape

    @property
    def distribution(self):
        return self._distribution

    def sample(self, key: PRNGKeyArray) -> Array:
        """See `Distribution.sample`."""
        return self._distribution.sample(key)

    def sample_and_log_prob(self, key: PRNGKeyArray) -> Tuple[Array, Array]:
        """See `Distribution.sample_and_log_prob`."""
        samples, log_prob = self._distribution.sample_and_log_prob(key)
        log_prob = _reduce_helper(log_prob)
        return samples, log_prob

    def log_prob(self, value: PyTree) -> Array:
        """See `Distribution.log_prob`."""
        return _reduce_helper(self._distribution.log_prob(value))

    def entropy(self) -> Array:
        """See `Distribution.entropy`."""
        return _reduce_helper(self._distribution.entropy())

    def log_cdf(self, value: PyTree) -> Array:
        """See `Distribution.log_cdf`."""
        return _reduce_helper(self._distribution.log_cdf(value))

    def mean(self) -> Array:
        """Calculates the mean."""
        return self._distribution.mean()

    def median(self) -> Array:
        """Calculates the median."""
        return self._distribution.median()

    def variance(self) -> Array:
        """Calculates the variance."""
        return self._distribution.variance()

    def stddev(self) -> Array:
        """Calculates the standard deviation."""
        return self._distribution.stddev()

    def mode(self) -> Array:
        """Calculates the mode."""
        return self._distribution.mode()

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible disteqx distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The KL divergence `KL(self || other_dist)`.
        """
        return _kl_divergence_independent_independent(self, other_dist)


def _kl_divergence_independent_independent(
    dist1: Independent,
    dist2: Independent,
    *args,
    **kwargs,
) -> Array:
    """Batched KL divergence `KL(dist1 || dist2)` for Independent distributions.

    **Arguments:**
    - `dist1`: instance of an Independent distribution.
    - dist2`: instance of an Independent distribution.
    - `*args`: Additional args.
    - `**kwargs`: Additional kwargs.

    **Returns:**
    - `KL(dist1 || dist2)`
    """
    p = dist1.distribution
    q = dist2.distribution

    if dist1.event_shape == dist2.event_shape:
        if p.event_shape == q.event_shape:
            kl_divergence = _reduce_helper(p.kl_divergence(q))
        else:
            raise NotImplementedError(
                f"KL between Independents whose inner distributions have different "
                f"event shapes is not supported: obtained {p.event_shape} and "
                f"{q.event_shape}."
            )
    else:
        raise ValueError(
            f"Event shapes {dist1.event_shape} and {dist2.event_shape}"
            f" do not match."
        )

    return kl_divergence
