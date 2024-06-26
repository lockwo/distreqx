"""Wrapper to turn independent Bijectors into block Bijectors."""

from jaxtyping import Array

from ..utils import sum_last
from ._bijector import AbstractBijector


class Block(AbstractBijector, strict=True):
    """A wrapper that promotes a bijector to a block bijector.

    A block bijector applies a bijector to a k-dimensional array of events, but
    considers that array of events to be a single event. In practical terms, this
    means that the log det Jacobian will be summed over its last k dimensions.

    For example, consider a scalar bijector (such as `Tanh`) that operates on
    scalar events. We may want to apply this bijector identically to a 4D array of
    shape [N, H, W, C] representing a sequence of N images. Doing so naively with
    a `vmap` will produce a log det Jacobian of shape [N, H, W, C], because the
    scalar bijector will assume scalar events and so all 4 dimensions will be
    considered as batch. To promote the scalar bijector to a "block scalar" that
    operates on the 3D arrays can be done by `Block(bijector, ndims=3)`. Then,
    applying the block bijector will produce a log det Jacobian of shape [N]
    as desired.

    In general, suppose `bijector` operates on n-dimensional events. Then,
    `Block(bijector, k)` will promote `bijector` to a block bijector that
    operates on (k + n)-dimensional events, summing the log det Jacobian over its
    last k dimensions.
    """

    _ndims: int
    _bijector: AbstractBijector
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, bijector: AbstractBijector, ndims: int):
        """Initializes a Block.

        **Arguments:**

        - `bijector`: the bijector to be promoted to a block bijector. It can be a
            distreqx bijector or a callable to be wrapped by `Lambda`.
        - `ndims`: number of dimensions to promote to event dimensions.
        """
        if ndims < 0:
            raise ValueError(f"`ndims` must be non-negative; got {ndims}.")
        self._bijector = bijector
        self._ndims = ndims
        self._is_constant_jacobian = self._bijector.is_constant_jacobian
        self._is_constant_log_det = self._bijector.is_constant_log_det

    @property
    def bijector(self) -> AbstractBijector:
        """The base bijector, without promoting to a block bijector."""
        return self._bijector

    @property
    def ndims(self) -> int:
        """The number of batch dimensions promoted to event dimensions."""
        return self._ndims

    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        return self._bijector.forward(x)

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        return self._bijector.inverse(y)

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        log_det = self._bijector.forward_log_det_jacobian(x)
        return sum_last(log_det, self._ndims)

    def inverse_log_det_jacobian(self, y: Array) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        log_det = self._bijector.inverse_log_det_jacobian(y)
        return sum_last(log_det, self._ndims)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        y, log_det = self._bijector.forward_and_log_det(x)
        return y, sum_last(log_det, self._ndims)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        x, log_det = self._bijector.inverse_and_log_det(y)
        return x, sum_last(log_det, self._ndims)

    @property
    def name(self) -> str:
        """Name of the bijector."""
        return self.__class__.__name__ + self._bijector.name

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Block:
            return self.bijector.same_as(other.bijector) and self.ndims == other.ndims

        return False
