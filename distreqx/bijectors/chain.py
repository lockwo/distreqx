"""Chain Bijector for composing a sequence of Bijector transformations."""

from typing import List, Sequence

from jaxtyping import Array

from ._bijector import (
    AbstractBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class Chain(AbstractFwdLogDetJacBijector, AbstractInvLogDetJacBijector, strict=True):
    """Composition of a sequence of bijectors into a single bijector.

    Bijectors are composable: if `f` and `g` are bijectors, then `g o f` is also
    a bijector. Given a sequence of bijectors `[f1, ..., fN]`, this class
    implements the bijector defined by `fN o ... o f1`.

    NOTE: the bijectors are applied in reverse order from the order they appear in
    the sequence. For example, consider the following code where `f` and `g` are
    two bijectors:

    ```python
    layers = []
    layers.append(f)
    layers.append(g)
    bijector = distrax.Chain(layers)
    y = bijector.forward(x)
    ```

    The above code will transform `x` by first applying `g`, then `f`, so that
    `y = f(g(x))`.
    """

    _bijectors: List[AbstractBijector]
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, bijectors: Sequence[AbstractBijector]):
        """Initializes a Chain bijector.

        **Arguments:**

        - `bijectors`: a sequence of bijectors to be composed into one. Each bijector
            can be a distreqx bijector or a callable to be wrapped
            by `Lambda`. The sequence must contain at least one bijector.
        """
        if not bijectors:
            raise ValueError("The sequence of bijectors cannot be empty.")
        self._bijectors = list(bijectors)

        is_constant_jacobian = all(b.is_constant_jacobian for b in self._bijectors)
        is_constant_log_det = all(b.is_constant_log_det for b in self._bijectors)
        if is_constant_log_det is None:
            is_constant_log_det = is_constant_jacobian
        if is_constant_jacobian and not is_constant_log_det:
            raise ValueError(
                "The Jacobian is said to be constant, but its "
                "determinant is said not to be, which is impossible."
            )
        self._is_constant_jacobian = is_constant_jacobian
        self._is_constant_log_det = is_constant_log_det

    @property
    def bijectors(self) -> List[AbstractBijector]:
        """The list of bijectors in the chain."""
        return self._bijectors

    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        for bijector in reversed(self._bijectors):
            x = bijector.forward(x)
        return x

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        for bijector in self._bijectors:
            y = bijector.inverse(y)
        return y

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        x, log_det = self._bijectors[-1].forward_and_log_det(x)
        for bijector in reversed(self._bijectors[:-1]):
            x, ld = bijector.forward_and_log_det(x)
            log_det += ld
        return x, log_det

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        y, log_det = self._bijectors[0].inverse_and_log_det(y)
        for bijector in self._bijectors[1:]:
            y, ld = bijector.inverse_and_log_det(y)
            log_det += ld
        return y, log_det

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Chain:
            if len(self.bijectors) != len(other.bijectors):
                return False
            for bij1, bij2 in zip(self.bijectors, other.bijectors):
                if not bij1.same_as(bij2):
                    return False
            return True
        elif len(self.bijectors) == 1:
            return self.bijectors[0].same_as(other)

        return False
