"""Independent distribution."""

import math
import operator

import equinox as eqx
import jax
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
    reinterpreted_batch_ndims: int

    def __init__(
        self,
        distribution: AbstractDistribution,
        reinterpreted_batch_ndims: int = 0,
    ):
        """Initializes an Independent distribution.

        **Arguments:**

        - `distribution`: Base distribution instance.
        - `reinterpreted_batch_ndims`: Number of batch dimensions to reinterpret
          as event dimensions. Defaults to 0, which preserves standard broadcasting
          behavior for natively batched distributions (e.g., `Normal`).
          **Note:** If you are passing a distribution that does not natively broadcast
          and was batched using `eqx.filter_vmap` (e.g., `MultivariateNormalTri`),
          you *must* explicitly set this to the number of mapped axes.
        """
        self.distribution = distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def _infer_shapes_and_dtype(self):
        """Infer the event shape by tracing `sample`."""
        if self.reinterpreted_batch_ndims == 0:
            return self.distribution.event_shape, self.distribution.dtype

        dummy_key = jax.random.key(0)

        # Close over the key so we only map over the distribution
        def _single_sample(d):
            return d.sample(dummy_key)

        fn = _single_sample
        for _ in range(self.reinterpreted_batch_ndims):
            fn = eqx.filter_vmap(fn)

        shape_dtype = eqx.filter_eval_shape(fn, self.distribution)
        return shape_dtype.shape, shape_dtype.dtype

    @property
    def dtype(self) -> jnp.dtype:
        """See `Distribution.dtype`."""
        return self._infer_shapes_and_dtype()[1]

    @property
    def event_shape(self) -> tuple[int, ...]:
        """See `Distribution.event_shape`."""
        event_shape = self._infer_shapes_and_dtype()[0]
        return event_shape

    def sample(self, key: Key[Array, ""]) -> Array:
        """See `Distribution.sample`."""
        if self.reinterpreted_batch_ndims == 0:
            return self.distribution.sample(key)

        bshape = self.event_shape[: self.reinterpreted_batch_ndims]
        total_batches = math.prod(bshape)
        keys = jax.random.split(key, total_batches).reshape(*bshape)

        def _single_sample(d, k):
            return d.sample(k)

        fn = _single_sample
        for _ in range(self.reinterpreted_batch_ndims):
            fn = eqx.filter_vmap(fn)

        return fn(self.distribution, keys)

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[Array, Array]:
        """See `Distribution.sample_and_log_prob`."""
        if self.reinterpreted_batch_ndims == 0:
            samples, log_prob = self.distribution.sample_and_log_prob(key)
            return samples, _reduce_helper(log_prob)

        bshape = self.event_shape[: self.reinterpreted_batch_ndims]
        total_batches = math.prod(bshape)
        keys = jax.random.split(key, total_batches).reshape(*bshape)

        def _single_sample_and_log_prob(d, k):
            return d.sample_and_log_prob(k)

        fn = _single_sample_and_log_prob
        for _ in range(self.reinterpreted_batch_ndims):
            fn = eqx.filter_vmap(fn)

        samples, log_probs = fn(self.distribution, keys)
        sum_axes = tuple(range(self.reinterpreted_batch_ndims))

        return samples, jnp.sum(log_probs, axis=sum_axes)

    def log_prob(self, value: PyTree) -> Array:
        """See `Distribution.log_prob`."""
        if self.reinterpreted_batch_ndims == 0:
            return _reduce_helper(self.distribution.log_prob(value))

        def _single_log_prob(d, x):
            return d.log_prob(x)

        fn = _single_log_prob
        for _ in range(self.reinterpreted_batch_ndims):
            fn = eqx.filter_vmap(fn)

        log_probs = fn(self.distribution, value)
        sum_axes = tuple(range(self.reinterpreted_batch_ndims))
        return jnp.sum(log_probs, axis=sum_axes)

    def entropy(self) -> Array:
        """See `Distribution.entropy`."""
        if self.reinterpreted_batch_ndims == 0:
            return _reduce_helper(self.distribution.entropy())

        def _single_entropy(d):
            return d.entropy()

        fn = _single_entropy
        for _ in range(self.reinterpreted_batch_ndims):
            fn = eqx.filter_vmap(fn)

        entropies = fn(self.distribution)
        sum_axes = tuple(range(self.reinterpreted_batch_ndims))
        return jnp.sum(entropies, axis=sum_axes)

    def log_cdf(self, value: PyTree) -> Array:
        """See `Distribution.log_cdf`."""
        if self.reinterpreted_batch_ndims == 0:
            return _reduce_helper(self.distribution.log_cdf(value))

        def _single_log_cdf(d, x):
            return d.log_cdf(x)

        fn = _single_log_cdf
        for _ in range(self.reinterpreted_batch_ndims):
            fn = eqx.filter_vmap(fn)

        log_cdfs = fn(self.distribution, value)
        sum_axes = tuple(range(self.reinterpreted_batch_ndims))
        return jnp.sum(log_cdfs, axis=sum_axes)

    def mean(self) -> Array:
        """Calculates the mean."""
        if self.reinterpreted_batch_ndims == 0:
            return self.distribution.mean()

        def _single_mean(d):
            return d.mean()

        fn = _single_mean
        for _ in range(self.reinterpreted_batch_ndims):
            fn = eqx.filter_vmap(fn)

        return fn(self.distribution)

    def icdf(self, value: Array) -> Array:
        """See `Distribution.icdf`."""
        raise NotImplementedError

    def median(self) -> Array:
        """Calculates the median."""
        if self.reinterpreted_batch_ndims == 0:
            return self.distribution.median()

        def _single_median(d):
            return d.median()

        fn = _single_median
        for _ in range(self.reinterpreted_batch_ndims):
            fn = eqx.filter_vmap(fn)

        return fn(self.distribution)

    def variance(self) -> Array:
        """Calculates the variance."""
        if self.reinterpreted_batch_ndims == 0:
            return self.distribution.variance()

        def _single_variance(d):
            return d.variance()

        fn = _single_variance
        for _ in range(self.reinterpreted_batch_ndims):
            fn = eqx.filter_vmap(fn)

        return fn(self.distribution)

    def stddev(self) -> Array:
        """Calculates the standard deviation."""
        if self.reinterpreted_batch_ndims == 0:
            return self.distribution.stddev()

        def _single_stddev(d):
            return d.stddev()

        fn = _single_stddev
        for _ in range(self.reinterpreted_batch_ndims):
            fn = eqx.filter_vmap(fn)

        return fn(self.distribution)

    def mode(self) -> Array:
        """Calculates the mode."""
        if self.reinterpreted_batch_ndims == 0:
            return self.distribution.mode()

        def _single_mode(d):
            return d.mode()

        fn = _single_mode
        for _ in range(self.reinterpreted_batch_ndims):
            fn = eqx.filter_vmap(fn)

        return fn(self.distribution)

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible distreqx distribution.
        - `kwargs`: Additional kwargs.

        **Returns:**

        The KL divergence `KL(self || other_dist)`.
        """
        dist1 = self
        dist2 = other_dist
        p = dist1.distribution
        q = dist2.distribution

        if dist1.event_shape == dist2.event_shape:
            # Safely extract the base event shapes without triggering unsafe traces
            # on the raw, un-wrapped vmapped inner distributions.
            d1_rndims = dist1.reinterpreted_batch_ndims
            d2_rndims = dist2.reinterpreted_batch_ndims
            p_base_shape = dist1.event_shape[d1_rndims:]  # fmt: skip
            q_base_shape = dist2.event_shape[d2_rndims:]  # fmt: skip

            if p_base_shape == q_base_shape:
                if self.reinterpreted_batch_ndims == 0:
                    kl_divergence = _reduce_helper(p.kl_divergence(q))
                else:

                    def _single_kl(d1, d2):
                        return d1.kl_divergence(d2)

                    fn = _single_kl
                    for _ in range(self.reinterpreted_batch_ndims):
                        fn = eqx.filter_vmap(fn)

                    kl_divs = fn(p, q)
                    sum_axes = tuple(range(self.reinterpreted_batch_ndims))
                    kl_divergence = jnp.sum(kl_divs, axis=sum_axes)
            else:
                raise NotImplementedError(
                    f"KL between Independents whose inner distributions have different "
                    f"event shapes is not supported: obtained {p_base_shape} and "
                    f"{q_base_shape}."
                )
        else:
            raise ValueError(
                f"Event shapes {dist1.event_shape} and {dist2.event_shape}"
                f" do not match."
            )

        return kl_divergence
