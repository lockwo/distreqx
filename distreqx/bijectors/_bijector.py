from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, PyTree


class AbstractBijector(eqx.Module, strict=True):
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

    _is_constant_jacobian: eqx.AbstractVar[bool]
    _is_constant_log_det: eqx.AbstractVar[bool]

    @abstractmethod
    def forward(self, x: PyTree) -> PyTree:
        R"""Computes $y = f(x)$."""
        raise NotImplementedError

    @abstractmethod
    def inverse(self, y: PyTree) -> PyTree:
        r"""Computes $x = f^{-1}(y)$."""
        raise NotImplementedError

    @abstractmethod
    def forward_log_det_jacobian(self, x: PyTree) -> PyTree:
        r"""Computes $\log|\det J(f)(x)|$."""
        raise NotImplementedError

    @abstractmethod
    def inverse_log_det_jacobian(self, y: PyTree) -> PyTree:
        r"""Computes $\log|\det J(f^{-1})(y)|$."""
        raise NotImplementedError

    @abstractmethod
    def forward_and_log_det(self, x: PyTree) -> tuple[PyTree, PyTree]:
        r"""Computes $y = f(x)$ and $\log|\det J(f)(x)|$."""
        raise NotImplementedError(
            f"Bijector {self.name} does not implement `forward_and_log_det`."
        )

    @abstractmethod
    def inverse_and_log_det(self, y: Array) -> tuple[PyTree, PyTree]:
        r"""Computes $x = f^{-1}(y)$ and $\log|\det J(f^{-1})(y)|$."""
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

    @abstractmethod
    def same_as(self, other) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        raise NotImplementedError


class AbstractInvLogDetJacBijector(AbstractBijector, strict=True):
    """AbstractBijector + concrete `inverse_log_det_jacobian`."""

    def inverse_log_det_jacobian(self, y: PyTree) -> PyTree:
        r"""Computes $\log|\det J(f^{-1})(y)|$."""
        _, logdet = self.inverse_and_log_det(y)
        return logdet


class AbstractFwdLogDetJacBijector(AbstractBijector, strict=True):
    """AbstractBijector + concrete `forward_log_det_jacobian`."""

    def forward_log_det_jacobian(self, x: PyTree) -> PyTree:
        r"""Computes $\log|\det J(f)(x)|$."""
        _, logdet = self.forward_and_log_det(x)
        return logdet


class AbstractFowardInverseBijector(AbstractBijector, strict=True):
    """AbstractBijector + concrete `forward` and `reverse`."""

    def forward(self, x: PyTree) -> PyTree:
        R"""Computes $y = f(x)$."""
        y, _ = self.forward_and_log_det(x)
        return y

    def inverse(self, y: PyTree) -> PyTree:
        r"""Computes $x = f^{-1}(y)$."""
        x, _ = self.inverse_and_log_det(y)
        return x
