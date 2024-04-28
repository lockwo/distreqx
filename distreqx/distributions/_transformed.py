"""Distribution representing a Bijector applied to a Distribution."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ..bijectors import AbstractBijector
from ._distribution import AbstractDistribution


class Transformed(AbstractDistribution):
    """Distribution of a random variable transformed by a bijective function.

    Let `X` be a continuous random variable and `Y = f(X)` be a random variable
    transformed by a differentiable bijection `f` (a "bijector"). Given the
    distribution of `X` (the "base distribution") and the bijector `f`, this class
    implements the distribution of `Y` (also known as the pushforward of the base
    distribution through `f`).

    The probability density of `Y` can be computed by:

    `log p(y) = log p(x) - log|det J(f)(x)|`

    where `p(x)` is the probability density of `X` (the "base density") and
    `J(f)(x)` is the Jacobian matrix of `f`, both evaluated at `x = f^{-1}(y)`.

    Sampling from a Transformed distribution involves two steps: sampling from the
    base distribution `x ~ p(x)` and then evaluating `y = f(x)`. For example:

    ```python
      dist = distrax.Normal(loc=0., scale=1.)
      bij = distrax.ScalarAffine(shift=jnp.asarray([3., 3., 3.]))
      transformed_dist = distrax.Transformed(distribution=dist, bijector=bij)
      samples = transformed_dist.sample(jax.random.PRNGKey(0))
      print(samples)  # [2.7941577, 2.7941577, 2.7941577]
    ```

    This assumes that the `forward` function of the bijector is traceable; that is,
    it is a pure function that does not contain run-time branching. Functions that
    do not strictly meet this requirement can still be used, but we cannot guarantee
    that the shapes, dtype, and KL computations involving the transformed distribution
    can be correctly obtained.
    """

    _distribution: AbstractDistribution
    _bijector: AbstractBijector

    def __init__(self, distribution: AbstractDistribution, bijector: AbstractBijector):
        """Initializes a Transformed distribution.

        **Arguments:**
        - `distribution`: the base distribution.
        - `bijector`: a differentiable bijective transformation. Can be a bijector or
            a callable to be wrapped by `Lambda` bijector.
        """
        self._distribution = distribution
        self._bijector = bijector

    @property
    def distribution(self):
        """The base distribution."""
        return self._distribution

    @property
    def bijector(self):
        """The bijector representing the transformation."""
        return self._bijector

    def _infer_shapes_and_dtype(self):
        """Infer the event shape by tracing `forward`."""
        dummy_shape = self.distribution.event_shape
        dummy = jnp.zeros(dummy_shape, dtype=self.distribution.dtype)
        shape_dtype = jax.eval_shape(self.bijector.forward, dummy)
        return shape_dtype, shape_dtype.dtype

    @property
    def dtype(self) -> jnp.dtype:
        """See `Distribution.dtype`."""
        return self._infer_shapes_and_dtype()[1]

    @property
    def event_shape(self) -> Tuple[int, ...]:
        """See `Distribution.event_shape`."""
        return self._infer_shapes_and_dtype()[0]

    def log_prob(self, value: Array) -> Array:
        """See `Distribution.log_prob`."""
        x, ildj_y = self.bijector.inverse_and_log_det(value)
        lp_x = self.distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y

    def sample(self, key: PRNGKeyArray) -> Array:
        """Return asamples."""
        x = self.distribution.sample(key)
        y = self.bijector.forward(x)
        return y

    def sample_and_log_prob(self, key: PRNGKeyArray) -> Tuple[Array, Array]:
        """Return a sample and log prob.

        This function is more efficient than calling `sample` and `log_prob`
        separately, because it uses only the forward methods of the bijector. It
        also works for bijectors that don't implement inverse methods.

        **Arguments:**

        - `key`: PRNG key.

        **Returns:**

        A tuple of a sample and its log probs.
        """
        x, lp_x = self.distribution.sample_and_log_prob(key)
        y, fldj = self.bijector.forward_and_log_det(x)
        lp_y = jnp.subtract(lp_x, fldj)
        return y, lp_y

    def mean(self) -> Array:
        """Calculates the mean."""
        if self.bijector.is_constant_jacobian:
            return self.bijector.forward(self.distribution.mean())
        else:
            raise NotImplementedError(
                "`mean` is not implemented for this transformed distribution, "
                "because its bijector's Jacobian is not known to be constant."
            )

    def mode(self) -> Array:
        """Calculates the mode."""
        if self.bijector.is_constant_log_det:
            return self.bijector.forward(self.distribution.mode())
        else:
            raise NotImplementedError(
                "`mode` is not implemented for this transformed distribution, "
                "because its bijector's Jacobian determinant is not known to be "
                "constant."
            )

    def entropy(self, input_hint: Optional[Array] = None) -> Array:
        """Calculates the Shannon entropy (in Nats).

        Only works for bijectors with constant Jacobian determinant.

        **Arguments:**

        - `input_hint`: an example sample from the base distribution, used to compute
            the constant forward log-determinant. If not specified, it is computed
            using a zero array of the shape and dtype of a sample from the base
            distribution.

        **Returns:**

        The entropy of the distribution.

        **Raises:**

        - `NotImplementedError`: if bijector's Jacobian determinant is not known to be
                               constant.
        """
        if self.bijector.is_constant_log_det:
            if input_hint is None:
                shape = self.distribution.event_shape
                input_hint = jnp.zeros(shape, dtype=self.distribution.dtype)
            entropy = self.distribution.entropy()
            fldj = self.bijector.forward_log_det_jacobian(input_hint)
            return entropy + fldj
        else:
            raise NotImplementedError(
                "`entropy` is not implemented for this transformed distribution, "
                "because its bijector's Jacobian determinant is not known to be "
                "constant."
            )

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence to another distribution.

        **Arguments:**

        - `other_dist`: A compatible disteqx distribution.
        - `kwargs`: Additional kwargs, can accept an `input_hint`.

        **Returns:**

        The KL divergence `KL(self || other_dist)`.
        """
        return _kl_divergence_transformed_transformed(self, other_dist, **kwargs)


def _kl_divergence_transformed_transformed(
    dist1: Transformed,
    dist2: Transformed,
    *unused_args,
    input_hint: Optional[Array] = None,
    **unused_kwargs,
) -> Array:
    """Obtains the KL divergence between two Transformed distributions.

    This computes the KL divergence between two Transformed distributions with the
    same bijector. If the two Transformed distributions do not have the same
    bijector, an error is raised. To determine if the bijectors are equal, this
    method proceeds as follows:
    - If both bijectors are the same instance of a distreqx bijector, then they are
      declared equal.
    - If not the same instance, we check if they are equal according to their
      `same_as` predicate.
    - Otherwise, the string representation of the Jaxpr of the `forward` method
      of each bijector is compared. If both string representations are equal, the
      bijectors are declared equal.
    - Otherwise, the bijectors cannot be guaranteed to be equal and an error is
      raised.

    **Arguments:**

    - `dist1`: A Transformed distribution.
    - `dist2`: A Transformed distribution.
    - `input_hint`: an example sample from the base distribution, used to trace the
        `forward` method. If not specified, it is computed using a zero array of
        the shape and dtype of a sample from the base distribution.

    **Returns:**

    `KL(dist1 || dist2)`.

    **Raises:**

    - `NotImplementedError`: If bijectors are not known to be equal.
    - `ValueError`: If the base distributions do not have the same `event_shape`.
    """
    if dist1.distribution.event_shape != dist2.distribution.event_shape:
        raise ValueError(
            f"The two base distributions do not have the same event shape: "
            f"{dist1.distribution.event_shape} and "
            f"{dist2.distribution.event_shape}."
        )

    bij1 = dist1.bijector
    bij2 = dist2.bijector

    # Check if the bijectors are different.
    if bij1 != bij2 and not bij1.same_as(bij2):
        if input_hint is None:
            input_hint = jnp.zeros(
                dist1.distribution.event_shape, dtype=dist1.distribution.dtype
            )
        jaxpr_bij1 = jax.make_jaxpr(bij1.forward)(input_hint).jaxpr
        jaxpr_bij2 = jax.make_jaxpr(bij2.forward)(input_hint).jaxpr
        if str(jaxpr_bij1) != str(jaxpr_bij2):
            raise NotImplementedError(
                f"The KL divergence cannot be obtained because it is not possible to "
                f"guarantee that the bijectors {dist1.bijector.name} and "
                f"{dist2.bijector.name} of the Transformed distributions are "
                f"equal. If possible, use the same instance of a distreqx bijector."
            )

    return dist1.distribution.kl_divergence(dist2.distribution)
