import equinox as eqx
from jaxtyping import PyTree

from ._bijector import (
    AbstractBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class Inverse(AbstractFwdLogDetJacBijector, AbstractInvLogDetJacBijector, strict=True):
    """Inverted version of a given bijector."""

    bijector: AbstractBijector

    _is_constant_jacobian: bool = eqx.field(init=False)
    _is_constant_log_det: bool = eqx.field(init=False)

    def __post_init__(self):
        is_constant_jacobian = self.bijector.is_constant_jacobian
        is_constant_log_det = self.bijector.is_constant_log_det

        if is_constant_jacobian and not is_constant_log_det:
            raise ValueError(
                "The Jacobian is said to be constant, but its "
                "determinant is said not to be, which is impossible."
            )

        object.__setattr__(self, "_is_constant_jacobian", is_constant_jacobian)
        object.__setattr__(self, "_is_constant_log_det", is_constant_log_det)

    def forward(self, x: PyTree) -> PyTree:
        """Computes y = f(x)."""
        return self.bijector.inverse(x)

    def inverse(self, y: PyTree) -> PyTree:
        """Computes x = f^{-1}(y)."""
        return self.bijector.forward(y)

    def forward_and_log_det(self, x: PyTree) -> tuple[PyTree, PyTree]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.bijector.inverse_and_log_det(x)

    def inverse_and_log_det(self, y: PyTree) -> tuple[PyTree, PyTree]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.bijector.forward_and_log_det(y)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Inverse:
            return self.bijector.same_as(other.bijector)
        return False
