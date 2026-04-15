from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from ._distribution import AbstractDistribution


def _is_dist(x: Any) -> bool:
    """Helper function to stop JAX tree traversal at the distribution level."""
    return isinstance(x, AbstractDistribution)


class Joint(AbstractDistribution, strict=True):
    """Joint distribution over a PyTree of statistically independent distributions.

    Samples from the Joint distribution take the form of a PyTree structure that
    matches the structure of the underlying distributions. Log-probabilities,
    entropies, and KL divergences are summed over the tree.
    """

    distributions: PyTree[AbstractDistribution]

    def __init__(self, distributions: PyTree[AbstractDistribution]):
        """Initializes a Joint distribution.

        **Arguments:**

        - `distributions`: A PyTree of `distreqx` distributions.
        """
        # Ensure there is at least one distribution in the tree
        leaves = jax.tree_util.tree_leaves(distributions, is_leaf=_is_dist)
        if not leaves:
            raise ValueError(
                "The distributions PyTree must contain at least one distribution."
            )

        self.distributions = distributions

    @property
    def event_shape(self) -> PyTree[tuple]:
        """Shape of the joint event."""
        return jax.tree_util.tree_map(
            lambda dist: dist.event_shape, self.distributions, is_leaf=_is_dist
        )

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[PyTree[Array], Array]:
        """Returns a joint sample and its total log prob."""
        leaves, treedef = jax.tree_util.tree_flatten(
            self.distributions, is_leaf=_is_dist
        )
        keys = jax.random.split(key, len(leaves))
        keys_tree = jax.tree_util.tree_unflatten(treedef, keys)

        # Call sample_and_log_prob on every distribution in the tree
        samples_and_log_probs = jax.tree_util.tree_map(
            lambda d, k: d.sample_and_log_prob(key=k),
            self.distributions,
            keys_tree,
            is_leaf=_is_dist,
        )

        # Extract samples into an identically structured PyTree
        samples = jax.tree_util.tree_map(
            lambda _, p: p[0],
            self.distributions,
            samples_and_log_probs,
            is_leaf=_is_dist,
        )

        # Extract log_probs
        log_probs = jax.tree_util.tree_map(
            lambda _, p: p[1],
            self.distributions,
            samples_and_log_probs,
            is_leaf=_is_dist,
        )

        # Safely reduce any element-wise/batched arrays into scalars
        # before summing the tree
        log_probs_summed = jax.tree_util.tree_map(jnp.sum, log_probs)
        total_log_prob = jnp.sum(
            jnp.asarray(jax.tree_util.tree_leaves(log_probs_summed))
        )

        return samples, total_log_prob

    def sample(self, key: Key[Array, ""]) -> PyTree[Array]:
        """Samples a joint event."""
        leaves, treedef = jax.tree_util.tree_flatten(
            self.distributions, is_leaf=_is_dist
        )
        keys = jax.random.split(key, len(leaves))
        keys_tree = jax.tree_util.tree_unflatten(treedef, keys)

        return jax.tree_util.tree_map(
            lambda d, k: d.sample(key=k),
            self.distributions,
            keys_tree,
            is_leaf=_is_dist,
        )

    def prob(self, value: PyTree[Array]) -> Array:
        """Calculates the total probability of a joint event."""
        return jnp.exp(self.log_prob(value))

    def log_prob(self, value: PyTree[Array]) -> Array:
        """Compute the total log probability of the distributions in the tree."""
        log_probs = jax.tree_util.tree_map(
            lambda dist, val: dist.log_prob(val),
            self.distributions,
            value,
            is_leaf=_is_dist,
        )
        log_probs_summed = jax.tree_util.tree_map(jnp.sum, log_probs)
        return jnp.sum(jnp.asarray(jax.tree_util.tree_leaves(log_probs_summed)))

    def entropy(self) -> Array:
        """Calculates the sum of Shannon entropies (in nats)."""
        entropies = jax.tree_util.tree_map(
            lambda dist: dist.entropy(), self.distributions, is_leaf=_is_dist
        )
        entropies_summed = jax.tree_util.tree_map(jnp.sum, entropies)
        return jnp.sum(jnp.asarray(jax.tree_util.tree_leaves(entropies_summed)))

    def cdf(self, value: PyTree[Array]) -> Array:
        """Evaluates the joint cumulative distribution function."""
        return jnp.exp(self.log_cdf(value))

    def log_cdf(self, value: PyTree[Array]) -> Array:
        """Evaluates the log of the joint CDF."""
        log_cdfs = jax.tree_util.tree_map(
            lambda dist, val: dist.log_cdf(val),
            self.distributions,
            value,
            is_leaf=_is_dist,
        )
        log_cdfs_summed = jax.tree_util.tree_map(jnp.sum, log_cdfs)
        return jnp.sum(jnp.asarray(jax.tree_util.tree_leaves(log_cdfs_summed)))

    def survival_function(self, value: PyTree[Array]) -> Array:
        """Evaluates the joint survival function."""
        return jnp.exp(self.log_survival_function(value))

    def log_survival_function(self, value: PyTree[Array]) -> Array:
        """Evaluates the log of the joint survival function."""
        log_survs = jax.tree_util.tree_map(
            lambda dist, val: dist.log_survival_function(val),
            self.distributions,
            value,
            is_leaf=_is_dist,
        )
        log_survs_summed = jax.tree_util.tree_map(jnp.sum, log_survs)
        return jnp.sum(jnp.asarray(jax.tree_util.tree_leaves(log_survs_summed)))

    def mean(self) -> PyTree[Array]:
        """Calculates the joint mean."""
        return jax.tree_util.tree_map(
            lambda dist: dist.mean(), self.distributions, is_leaf=_is_dist
        )

    def median(self) -> PyTree[Array]:
        """Calculates the joint median."""
        return jax.tree_util.tree_map(
            lambda dist: dist.median(), self.distributions, is_leaf=_is_dist
        )

    def mode(self) -> PyTree[Array]:
        """Calculates the joint mode."""
        return jax.tree_util.tree_map(
            lambda dist: dist.mode(), self.distributions, is_leaf=_is_dist
        )

    def variance(self) -> PyTree[Array]:
        """Calculates the joint variance."""
        return jax.tree_util.tree_map(
            lambda dist: dist.variance(), self.distributions, is_leaf=_is_dist
        )

    def stddev(self) -> PyTree[Array]:
        """Calculates the joint standard deviation."""
        return jax.tree_util.tree_map(
            lambda dist: dist.stddev(), self.distributions, is_leaf=_is_dist
        )

    def icdf(self, value: PyTree[Array]) -> PyTree[Array]:
        """Evaluates the joint inverse cumulative distribution function."""
        return jax.tree_util.tree_map(
            lambda dist, val: dist.icdf(val),
            self.distributions,
            value,
            is_leaf=_is_dist,
        )

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence between two Joint distributions."""
        if not isinstance(other_dist, Joint):
            raise TypeError(
                "KL divergence is only supported between two Joint distributions."
            )

        # The trees must match structurally to zip over them
        kl_divs = jax.tree_util.tree_map(
            lambda d1, d2: d1.kl_divergence(d2),
            self.distributions,
            other_dist.distributions,
            is_leaf=_is_dist,
        )
        kl_divs_summed = jax.tree_util.tree_map(jnp.sum, kl_divs)
        return jnp.sum(jnp.asarray(jax.tree_util.tree_leaves(kl_divs_summed)))
