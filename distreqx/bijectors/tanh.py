"""Tanh bijector."""
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import (
    AbstractBijector,
    AbstractFowardInverseBijector,
    AbstractInvLogDetJacBijector,
)


class Tanh(AbstractFowardInverseBijector, AbstractInvLogDetJacBijector, strict=True):
    """A bijector that computes the hyperbolic tangent.

    The log-determinant implementation in this bijector is more numerically stable
    than relying on the automatic differentiation approach used by Lambda, so this
    bijector should be preferred over Lambda(jnp.tanh) where possible.

    When the absolute value of the input is large, `Tanh` becomes close to a
    constant, so that it is not possible to recover the input `x` from the output
    `y` within machine precision. In cases where it is needed to compute both the
    forward mapping and the backward mapping one after the other to recover the
    original input `x`, it is the user's responsibility to simplify the operation
    to avoid numerical issues. One example of such case is to use the bijector
    within a `Transformed` distribution and to obtain the log-probability of samples
    obtained from the distribution's `sample` method. For values of the samples
    for which it is not possible to apply the inverse bijector accurately,
    `log_prob` returns NaN. This can be avoided by using `sample_and_log_prob`
    instead of `sample` followed by `log_prob`.
    """

    _is_constant_log_det: bool
    _is_constant_jacobian: bool

    def __init__(self) -> None:
        self._is_constant_jacobian = False
        self._is_constant_log_det = False

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        return 2 * (jnp.log(2) - x - jax.nn.softplus(-2 * x))

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return jnp.tanh(x), self.forward_log_det_jacobian(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        x = jnp.arctanh(y)
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is Tanh
