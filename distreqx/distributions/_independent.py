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

        !!! note

            If you are passing a distribution that does not natively broadcast
            and was batched using `eqx.filter_vmap` (e.g., `MultivariateNormalTri`),
            you *must* explicitly set `reinterpreted_batch_ndims` to the number of
            mapped axes.
        """
        self.distribution = distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def _vmap_method(self, method_name: str, obj, *args):
        """Calls `getattr(obj, method_name)(*args)`, vmapping over `obj` and
        `args` `reinterpreted_batch_ndims` times (a no-op when it's 0)."""

        def _single(o, *a):
            return getattr(o, method_name)(*a)

        fn = _single
        for _ in range(self.reinterpreted_batch_ndims):
            fn = eqx.filter_vmap(fn)
        return fn(obj, *args)

    def _vmap_and_sum(self, method_name: str, obj, *args):
        """As `_vmap_method`, but additionally sums the result over the
        reinterpreted batch axes (or fully reduces the pytree when there are
        none, matching `reinterpreted_batch_ndims == 0`)."""
        if self.reinterpreted_batch_ndims == 0:
            return _reduce_helper(getattr(obj, method_name)(*args))
        result = self._vmap_method(method_name, obj, *args)
        sum_axes = tuple(range(self.reinterpreted_batch_ndims))
        return jnp.sum(result, axis=sum_axes)

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

    @property
    def support(self) -> tuple[Array, Array]:
        """See `Distribution.support`."""
        return self.distribution.support

    def sample(self, key: Key[Array, ""]) -> Array:
        """See `Distribution.sample`."""
        if self.reinterpreted_batch_ndims == 0:
            return self.distribution.sample(key)

        bshape = self.event_shape[: self.reinterpreted_batch_ndims]
        total_batches = math.prod(bshape)
        keys = jax.random.split(key, total_batches).reshape(*bshape)
        return self._vmap_method("sample", self.distribution, keys)

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[Array, Array]:
        """See `Distribution.sample_and_log_prob`."""
        if self.reinterpreted_batch_ndims == 0:
            samples, log_prob = self.distribution.sample_and_log_prob(key)
            return samples, _reduce_helper(log_prob)

        bshape = self.event_shape[: self.reinterpreted_batch_ndims]
        total_batches = math.prod(bshape)
        keys = jax.random.split(key, total_batches).reshape(*bshape)

        samples, log_probs = self._vmap_method(
            "sample_and_log_prob", self.distribution, keys
        )
        sum_axes = tuple(range(self.reinterpreted_batch_ndims))
        return samples, jnp.sum(log_probs, axis=sum_axes)

    def log_prob(self, value: PyTree) -> Array:
        """See `Distribution.log_prob`."""
        return self._vmap_and_sum("log_prob", self.distribution, value)

    def entropy(self) -> Array:
        """See `Distribution.entropy`."""
        return self._vmap_and_sum("entropy", self.distribution)

    def log_cdf(self, value: PyTree) -> Array:
        """See `Distribution.log_cdf`."""
        return self._vmap_and_sum("log_cdf", self.distribution, value)

    def mean(self) -> Array:
        """Calculates the mean."""
        return self._vmap_method("mean", self.distribution)

    def icdf(self, value: Array) -> Array:
        """See `Distribution.icdf`."""
        raise NotImplementedError

    def median(self) -> Array:
        """Calculates the median."""
        return self._vmap_method("median", self.distribution)

    def variance(self) -> Array:
        """Calculates the variance."""
        return self._vmap_method("variance", self.distribution)

    def stddev(self) -> Array:
        """Calculates the standard deviation."""
        return self._vmap_method("stddev", self.distribution)

    def mode(self) -> Array:
        """Calculates the mode."""
        return self._vmap_method("mode", self.distribution)

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

        if dist1.event_shape != dist2.event_shape:
            raise ValueError(
                f"Event shapes {dist1.event_shape} and {dist2.event_shape}"
                f" do not match."
            )

        d1_rndims = dist1.reinterpreted_batch_ndims
        d2_rndims = dist2.reinterpreted_batch_ndims
        if d1_rndims != d2_rndims:
            # A mismatch here would silently sum over the wrong number of axes
            # on one side, so we enforce it rather than guess.
            raise ValueError(
                f"KL between Independents requires matching "
                f"`reinterpreted_batch_ndims`: got {d1_rndims} and {d2_rndims}."
            )

        # Safely extract the base event shapes without triggering unsafe traces
        # on the raw, un-wrapped vmapped inner distributions.
        p_base_shape = dist1.event_shape[d1_rndims:]
        q_base_shape = dist2.event_shape[d2_rndims:]

        if p_base_shape != q_base_shape:
            raise NotImplementedError(
                f"KL between Independents whose inner distributions have different "
                f"event shapes is not supported: obtained {p_base_shape} and "
                f"{q_base_shape}."
            )

        return self._vmap_and_sum("kl_divergence", p, q)
