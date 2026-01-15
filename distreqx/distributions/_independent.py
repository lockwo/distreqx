"""Independent distribution."""

import operator

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Key, PyTree

from ._distribution import (
    AbstractCDFDistribution,
    AbstractDistribution,
    AbstractProbDistribution,
    AbstractSurvivalDistribution,
)


def _reduce_helper(pytree: PyTree) -> Array:
    sum_over_leaves = jtu.tree_map(jnp.sum, pytree)
    return jtu.tree_reduce(operator.add, sum_over_leaves)


class Independent(
    AbstractProbDistribution,
    AbstractCDFDistribution,
    AbstractSurvivalDistribution,
    strict=True,
):
    """Independent distribution obtained from child distributions.

    !!! tip

        `Independent` reinterprets batch dimensions as event dimensions. This is
        useful when you want to model a multivariate distribution as independent
        univariate distributions (e.g., diagonal Gaussian) but still want
        `log_prob` to return a single scalar per sample.
    """

    distribution: AbstractDistribution

    def __init__(
        self,
        distribution: AbstractDistribution,
    ):
        """Initializes an Independent distribution.

        **Arguments:**

        - `distribution`: Base distribution instance.
        """
        self.distribution = distribution

    @property
    def event_shape(self) -> tuple:
        """Shape of event of distribution samples."""
        return self.distribution.event_shape

    def sample(self, key: Key[Array, ""]) -> Array:
        """See `Distribution.sample`."""
        return self.distribution.sample(key)

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[Array, Array]:
        """See `Distribution.sample_and_log_prob`."""
        samples, log_prob = self.distribution.sample_and_log_prob(key)
        log_prob = _reduce_helper(log_prob)
        return samples, log_prob

    def log_prob(self, value: PyTree) -> Array:
        """See `Distribution.log_prob`."""
        return _reduce_helper(self.distribution.log_prob(value))

    def entropy(self) -> Array:
        """See `Distribution.entropy`."""
        return _reduce_helper(self.distribution.entropy())

    def log_cdf(self, value: PyTree) -> Array:
        """See `Distribution.log_cdf`."""
        return _reduce_helper(self.distribution.log_cdf(value))

    def mean(self) -> Array:
        """Calculates the mean."""
        return self.distribution.mean()

    def median(self) -> Array:
        """Calculates the median."""
        return self.distribution.median()

    def variance(self) -> Array:
        """Calculates the variance."""
        return self.distribution.variance()

    def stddev(self) -> Array:
        """Calculates the standard deviation."""
        return self.distribution.stddev()

    def mode(self) -> Array:
        """Calculates the mode."""
        return self.distribution.mode()

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible disteqx distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The KL divergence `KL(self || other_dist)`.
        """
        dist1 = self
        dist2 = other_dist
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
