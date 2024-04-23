from typing import Tuple
from jaxtyping import Array
from abc import abstractmethod
import equinox as eqx


class AbstractBijector(eqx.Module):
    """Differentiable bijection that knows to compute its Jacobian determinant.

    A bijector implements a differentiable and bijective transformation `f`, whose
    inverse is also differentiable (`f` is called a "diffeomorphism"). A bijector
    can be used to transform a continuous random variable `X` to a continuous
    random variable `Y = f(X)` in the context of `TransformedDistribution`.

    Typically, a bijector subclass will implement the following methods:

    - `forward_and_log_det(x)` (required)
    - `inverse_and_log_det(y)` (optional)

    The remaining methods are defined in terms of the above by default.

    Subclass requirements:

    - Subclasses must ensure that `f` is differentiable and bijective, and that
      their methods correctly implement `f^{-1}`, `J(f)` and `J(f^{-1})`. Distreqx
      will assume these properties hold, and will make no attempt to verify them.
    """

    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(
        self,
        is_constant_jacobian: bool = False,
        is_constant_log_det: bool = False,
    ):
        """Initializes a Bijector.

        **Arguments:**

        - `is_constant_jacobian`: Whether the Jacobian is promised to be constant
            (which is the case if and only if the bijector is affine). A value of
            False will be interpreted as "we don't know whether the Jacobian is
            constant", rather than "the Jacobian is definitely not constant". Only
            set to True if you're absolutely sure the Jacobian is constant; if
            you're not sure, set to False.
        - `is_constant_log_det`: Whether the Jacobian determinant is promised to be
            constant (which is the case for, e.g., volume-preserving bijectors). If
            None, it defaults to `is_constant_jacobian`. Note that the Jacobian
            determinant can be constant without the Jacobian itself being constant.
            Only set to True if you're absoltely sure the Jacobian determinant is
            constant; if you're not sure, set to False.
        """
        if is_constant_log_det is None:
            is_constant_log_det = is_constant_jacobian
        if is_constant_jacobian and not is_constant_log_det:
            raise ValueError(
                "The Jacobian is said to be constant, but its "
                "determinant is said not to be, which is impossible."
            )
        self._is_constant_jacobian = is_constant_jacobian
        self._is_constant_log_det = is_constant_log_det

    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        y, _ = self.forward_and_log_det(x)
        return y

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        x, _ = self.inverse_and_log_det(y)
        return x

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        _, logdet = self.forward_and_log_det(x)
        return logdet

    def inverse_log_det_jacobian(self, y: Array) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        _, logdet = self.inverse_and_log_det(y)
        return logdet

    @abstractmethod
    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        raise NotImplementedError(
            f"Bijector {self.name} does not implement `forward_and_log_det`."
        )

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        raise NotImplementedError(
            f"Bijector {self.name} does not implement `inverse_and_log_det`."
        )

    @property
    def is_constant_jacobian(self) -> bool:
        """Whether the Jacobian is promised to be constant."""
        return self._is_constant_jacobian

    @property
    def is_constant_log_det(self) -> bool:
        """Whether the Jacobian determinant is promised to be constant."""
        return self._is_constant_log_det

    @property
    def name(self) -> str:
        """Name of the bijector."""
        return self.__class__.__name__

    def same_as(self, other) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        del other
        return False
