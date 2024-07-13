"""Mixture distributions."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ._distribution import (
    AbstractCDFDistribution,
    AbstractDistribution,
    AbstractProbDistribution,
    AbstractSampleLogProbDistribution,
    AbstractSTDDistribution,
    AbstractSurivialDistribution,
)
from .categorical import Categorical


class MixtureSameFamily(
    AbstractSTDDistribution,
    AbstractSampleLogProbDistribution,
    AbstractSurivialDistribution,
    AbstractProbDistribution,
    AbstractCDFDistribution,
    strict=True,
):
    """Mixture with components provided from a single batched distribution."""

    _mixture_distribution: Categorical
    _components_distribution: AbstractDistribution

    def __init__(
        self,
        mixture_distribution: Categorical,
        components_distribution: AbstractDistribution,
    ) -> None:
        """Initializes a mixture distribution for components of a shared family.

        **Arguments*:*

        - `mixture_distribution`: Distribution over selecting components.
        - `components_distribution`: Component distribution.
        """
        self._mixture_distribution = mixture_distribution
        self._components_distribution = components_distribution

    @property
    def components_distribution(self) -> AbstractDistribution:
        """The components distribution."""
        return self._components_distribution

    @property
    def mixture_distribution(self):
        """The mixture distribution."""
        return self._mixture_distribution

    @property
    def event_shape(self):
        """Shape of event of distribution samples."""
        return self._components_distribution.event_shape

    def sample(self, key) -> Array:
        """See `AbstractDistribution._sample`."""
        key_mix, key_components = jax.random.split(key)
        mix_sample = self.mixture_distribution.sample(key_mix)

        num_components = self._mixture_distribution.num_categories

        # Sample from all components, then multiply with a one-hot mask and sum.
        # While this does computation that is not used eventually, it is faster on
        # GPU/TPUs, which excel at batched operations (as opposed to indexing). It
        # is in particular more efficient than using `gather` or `where` operations.
        mask = jax.nn.one_hot(mix_sample, num_components)
        samples_all = self.components_distribution.sample(key_components)


        # Need to sum over the component axis, which is the last one for scalar
        # components, the second-last one for 1-dim events, etc.
        samples = jnp.sum(samples_all * mask, axis=0)
        return samples

    def mean(self) -> Array:
        """Calculates the mean."""
        means = self.components_distribution.mean()
        weights = self._mixture_distribution.probs
        # Broadcast weights over event shape, and average over component axis.
        weights = weights.reshape(weights.shape + (1,) * len(self.event_shape))
        return jnp.sum(means * weights, axis=-1 - len(self.event_shape))

    def variance(self) -> Array:
        """Calculates the variance."""
        means = self.components_distribution.mean()
        variances = self.components_distribution.variance()
        weights = self._mixture_distribution.probs
        # Make weights broadcast over event shape.
        weights = weights.reshape(weights.shape + (1,) * len(self.event_shape))
        # Component axis to reduce over.
        component_axis = -1 - len(self.event_shape)

        # Using: Var(Y) = E[Var(Y|X)] + Var(E[Y|X]).
        mean = jnp.sum(means * weights, axis=component_axis)
        mean_cond_var = jnp.sum(weights * variances, axis=component_axis)
        # Need to add an axis to `mean` to make it broadcast over components.
        sq_diff = jnp.square(means - jnp.expand_dims(mean, axis=component_axis))
        var_cond_mean = jnp.sum(weights * sq_diff, axis=component_axis)
        return mean_cond_var + var_cond_mean

    def _per_mixture_component_log_prob(self, value: Array) -> Array:
        # Add component axis to make input broadcast with components distribution.

        # Compute `log_prob` in every component.
        lp = eqx.filter_vmap(
            lambda dist, x: dist.log_prob(x), in_axes=(eqx.if_array(0), None)
        )(self.components_distribution, value)
        # Last axis of mixture log probs are components.
        return lp + jax.nn.log_softmax(self._mixture_distribution.logits, axis=-1)

    def log_prob(self, value: Array) -> Array:
        # Reduce last axis of mixture log probs are components
        return jax.scipy.special.logsumexp(
            self._per_mixture_component_log_prob(value), axis=-1
        )

    def posterior_marginal(self, observation: Array) -> Categorical:
        return Categorical(logits=self._per_mixture_component_log_prob(observation))

    def posterior_mode(self, observation: Array) -> Array:
        return jnp.argmax(self._per_mixture_component_log_prob(observation), axis=-1)

    def median(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def log_cdf(self, value: PyTree[Array]) -> PyTree[Array]:
        raise NotImplementedError

    def kl_divergence(self, other_dist, **kwargs):
        raise NotImplementedError
