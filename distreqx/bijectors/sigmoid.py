"""Sigmoid bijector."""

from typing import Tuple
from ._bijector import AbstractBijector
import jax
from jaxtyping import Array
import jax.numpy as jnp


class Sigmoid(AbstractBijector):
    """A bijector that computes the logistic sigmoid.

    The log-determinant implementation in this bijector is more numerically stable
    than relying on the automatic differentiation approach used by Lambda, so this
    bijector should be preferred over Lambda(jax.nn.sigmoid) where possible.

    Note that the underlying implementation of `jax.nn.sigmoid` used by the
    `forward` function of this bijector does not support inputs of integer type.
    To invoke the forward function of this bijector on an argument of integer
    type, it should first be cast explicitly to a floating point type.

    When the absolute value of the input is large, `Sigmoid` becomes close to a
    constant, so that it is not possible to recover the input `x` from the output
    `y` within machine precision. In cases where it is needed to compute both the
    forward mapping and the backward mapping one after the other to recover the
    original input `x`, it is the user's responsibility to simplify the operation
    to avoid numerical issues. One example of such case is to use the bijector
    within a `Transformed` distribution and to obtain the log-probability of 
    samples obtained from the distribution's `sample` method. For values of the
    samples for which it is not possible to apply the inverse bijector accurately,
    `log_prob` returns NaN. This can be avoided by using `sample_and_log_prob`
    instead of `sample` followed by `log_prob`.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        return -_more_stable_softplus(-x) - _more_stable_softplus(x)

    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return _more_stable_sigmoid(x), self.forward_log_det_jacobian(x)

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        x = jnp.log(y) - jnp.log1p(-y)
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is Sigmoid


def _more_stable_sigmoid(x: Array) -> Array:
    """Where extremely negatively saturated, approximate sigmoid with exp(x)."""
    return jnp.where(x < -9, jnp.exp(x), jax.nn.sigmoid(x))


def _more_stable_softplus(x: Array) -> Array:
    """Where extremely saturated, approximate softplus with log1p(exp(x))."""
    return jnp.where(x < -9, jnp.log1p(jnp.exp(x)), jax.nn.softplus(x))
