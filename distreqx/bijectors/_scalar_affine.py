"""Scalar affine bijector."""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ._bijector import AbstractBijector


class AbstractScalarAffine(AbstractBijector, strict=True):
    r"""An affine bijector that acts elementwise.

    The bijector is defined as follows:

    - Forward: $y = \text{scale} \cdot x + \text{shift}$
    - Forward Jacobian determinant: $\log|\det J(x)| = \log|\text{scale}|$
    - Inverse: $x = (y - \text{shift}) / \text{scale}$
    - Inverse Jacobian determinant: $\log|\det J(y)| = -\log|\text{scale}|$

    where `scale` and `shift` are the bijector's parameters.
    """

    shift: eqx.AbstractVar[Array]
    scale: eqx.AbstractVar[Array]
    inv_scale: eqx.AbstractVar[Array]
    log_scale: eqx.AbstractVar[Array]

    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        return self.scale * x + self.shift

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        return self.log_scale

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        return self.inv_scale * (y - self.shift)

    def inverse_log_det_jacobian(self, y: Array) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        return jnp.negative(self.log_scale)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)


class ScalarAffine(AbstractScalarAffine, strict=True):
    r"""An affine bijector that acts elementwise.

    The bijector is defined as follows:

    - Forward: $y = \text{scale} \cdot x + \text{shift}$
    - Forward Jacobian determinant: $\log|\det J(x)| = \log|\text{scale}|$
    - Inverse: $x = (y - \text{shift}) / \text{scale}$
    - Inverse Jacobian determinant: $\log|\det J(y)| = -\log|\text{scale}|$

    where `scale` and `shift` are the bijector's parameters.
    """

    shift: Array
    scale: Array
    inv_scale: Array
    log_scale: Array
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(
        self,
        shift: Array,
        scale: Optional[Array] = None,
        log_scale: Optional[Array] = None,
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
        self._is_constant_jacobian = True
        self._is_constant_log_det = True
        self.shift = shift
        if scale is None and log_scale is None:
            self.scale = jnp.ones_like(shift)
            self.inv_scale = jnp.ones_like(shift)
            self.log_scale = jnp.zeros_like(shift)
        elif log_scale is None and scale is not None:
            self.scale = scale
            self.inv_scale = 1.0 / scale
            self.log_scale = jnp.log(jnp.abs(scale))
        elif scale is None and log_scale is not None:
            self.scale = jnp.exp(log_scale)
            self.inv_scale = jnp.exp(jnp.negative(log_scale))
            self.log_scale = log_scale
        else:
            raise ValueError(
                "Only one of `scale` and `log_scale` can be specified, not both."
            )

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
