from collections.abc import Callable
from typing import Optional

import equinox as eqx
from jaxtyping import PyTree

# Assuming distreqx has ported these utilities from distrax
from ..utils import transformations
from ._bijector import (
    AbstractBijector,
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class Lambda(
    AbstractFowardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """Wrapper to automatically turn JAX functions into fully fledged bijectors.

    This class takes in JAX functions that implement bijector methods and
    constructs a bijector out of them. Any functions not explicitly specified
    by the user will be automatically derived from the existing functions where
    possible, by tracing their JAXPR representation during initialization.
    """

    _fn_forward: Callable[[PyTree], PyTree]
    _fn_inverse: Callable[[PyTree], PyTree]
    _fn_forward_log_det: Callable[[PyTree], PyTree]
    _fn_inverse_log_det: Callable[[PyTree], PyTree]

    _is_constant_jacobian: bool = eqx.field(init=False)
    _is_constant_log_det: bool = eqx.field(init=False)

    def __init__(
        self,
        forward: Optional[Callable[[PyTree], PyTree]] = None,
        inverse: Optional[Callable[[PyTree], PyTree]] = None,
        forward_log_det_jacobian: Optional[Callable[[PyTree], PyTree]] = None,
        inverse_log_det_jacobian: Optional[Callable[[PyTree], PyTree]] = None,
        is_constant_jacobian: Optional[bool] = None,
    ):
        """Initializes a Lambda bijector and eagerly derives missing methods."""
        if forward is None and inverse is None:
            raise ValueError(
                "The Lambda bijector requires at least one of `forward` "
                "or `inverse` to be specified, but neither is."
            )

        if forward is None and inverse is None:
            raise ValueError(
                "The Lambda bijector requires at least one of `forward` "
                "or `inverse` to be specified, but neither is."
            )

        if forward is None:
            assert (
                inverse is not None
            )  # Tells Pyright `inverse` is definitely a Callable here
            forward = transformations.inv(inverse)
        elif inverse is None:
            assert (
                forward is not None
            )  # Tells Pyright `forward` is definitely a Callable here
            inverse = transformations.inv(forward)

        jac_functions_specified = (
            forward_log_det_jacobian is not None or inverse_log_det_jacobian is not None
        )

        if not jac_functions_specified:
            # Derive both if neither is provided
            forward_log_det_jacobian = transformations.log_det_scalar(forward)
            inverse_log_det_jacobian = transformations.log_det_scalar(inverse)
        else:
            # If the user provided one but not the other, we derive the missing one
            if forward_log_det_jacobian is None:
                forward_log_det_jacobian = transformations.log_det_scalar(forward)
            if inverse_log_det_jacobian is None:
                inverse_log_det_jacobian = transformations.log_det_scalar(inverse)

        if is_constant_jacobian is None:
            fn = inverse if forward is None else forward
            is_constant_jacobian = transformations.is_constant_jacobian(fn)

        self._fn_forward = forward
        self._fn_inverse = inverse
        self._fn_forward_log_det = forward_log_det_jacobian
        self._fn_inverse_log_det = inverse_log_det_jacobian

        self._is_constant_jacobian = is_constant_jacobian
        self._is_constant_log_det = is_constant_jacobian

    def forward_and_log_det(self, x: PyTree) -> tuple[PyTree, PyTree]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self._fn_forward(x), self._fn_forward_log_det(x)

    def inverse_and_log_det(self, y: PyTree) -> tuple[PyTree, PyTree]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self._fn_inverse(y), self._fn_inverse_log_det(y)

    def same_as(self, other: AbstractBijector) -> bool:
        """
        Returns True if the other is a Lambda bijector with the exact
        same callables.
        """
        return (
            type(other) is Lambda
            and self._fn_forward is other._fn_forward
            and self._fn_inverse is other._fn_inverse
            and self._fn_forward_log_det is other._fn_forward_log_det
            and self._fn_inverse_log_det is other._fn_inverse_log_det
        )
