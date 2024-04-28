"""Scalar affine bijector."""

from typing import Optional, Tuple
from jaxtyping import Array
from ._bijector import AbstractBijector
from .._custom_types import RealScalarLike
import jax
import jax.numpy as jnp


class ScalarAffine(AbstractBijector):
    """An affine bijector that acts elementwise.

    The bijector is defined as follows:

    - Forward: `y = scale * x + shift`
    - Forward Jacobian determinant: `log|det J(x)| = log|scale|`
    - Inverse: `x = (y - shift) / scale`
    - Inverse Jacobian determinant: `log|det J(y)| = -log|scale|`

    where `scale` and `shift` are the bijector's parameters.
    """

    _shift: RealScalarLike
    _scale: RealScalarLike
    _inv_scale: RealScalarLike
    _log_scale: RealScalarLike

    def __init__(
        self,
        shift: RealScalarLike,
        scale: Optional[RealScalarLike] = None,
        log_scale: Optional[RealScalarLike] = None,
    ):
        """Initializes a ScalarAffine bijector.

        **Arguments:**

        - `shift`: the bijector's shift parameter.
        - `scale`: the bijector's scale parameter. NOTE: `scale` must be non-zero,
            otherwise the bijector is not invertible. It is the user's
            responsibility to make sure `scale` is non-zero; the class will
            make no attempt to verify this.
        - `log_scale`: the log of the scale parameter. If specified, the
            bijector's scale is set equal to `exp(log_scale)`. Unlike
            `scale`, `log_scale` is an unconstrained parameter. NOTE: either `scale`
            or `log_scale` can be specified, but not both. If neither is specified,
            the bijector's scale will default to 1.

        **Raises:**

        - `ValueError`: if both `scale` and `log_scale` are not None.
        """
        super().__init__(event_ndims_in=0, is_constant_jacobian=True)
        self._shift = shift
        if scale is None and log_scale is None:
            self._scale = 1.0
            self._inv_scale = 1.0
            self._log_scale = 0.0
        elif log_scale is None:
            self._scale = scale
            self._inv_scale = 1.0 / scale
            self._log_scale = jnp.log(jnp.abs(scale))
        elif scale is None:
            self._scale = jnp.exp(log_scale)
            self._inv_scale = jnp.exp(jnp.negative(log_scale))
            self._log_scale = log_scale
        else:
            raise ValueError(
                "Only one of `scale` and `log_scale` can be specified, not both."
            )

    @property
    def shift(self) -> RealScalarLike:
        """The bijector's shift."""
        return self._shift

    @property
    def log_scale(self) -> RealScalarLike:
        """The log of the bijector's scale."""
        return self._log_scale

    @property
    def scale(self) -> RealScalarLike:
        """The bijector's scale."""
        assert self._scale is not None  # By construction.
        return self._scale

    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        return self._scale * x + self._shift

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        return self._log_scale

    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        return self._inv_scale * (y - self._shift)

    def inverse_log_det_jacobian(self, y: Array) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        batch_shape = jax.lax.broadcast_shapes(self._batch_shape, y.shape)
        return jnp.broadcast_to(jnp.negative(self._log_scale), batch_shape)

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is ScalarAffine:
            return all(
                (
                    self.shift is other.shift,
                    self.scale is other.scale,
                    self.log_scale is other.log_scale,
                )
            )
        else:
            return False
