from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTreeDef

from ._bijector import (
    AbstractBijector,
    AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class Restructure(
    AbstractForwardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
    strict=True,
):
    """A bijector that restructures a PyTree of arrays.

    This is equivalent to `tfp.bijectors.Restructure`. It maps values between
    different nested structures (e.g., lists to dicts) without modifying the
    underlying arrays themselves.
    """

    in_treedef: PyTreeDef = eqx.field(static=True)  # type: ignore
    out_treedef: PyTreeDef = eqx.field(static=True)  # type: ignore
    forward_permutation: tuple[int, ...] = eqx.field(static=True)
    inverse_permutation: tuple[int, ...] = eqx.field(static=True)

    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def __init__(self, in_structure: Any, out_structure: Any):
        """Initializes a Restructure bijector.

        **Arguments:**

        - `in_structure`: A PyTree defining the input structure. Its leaves must
          be unique identifier tokens (e.g., integers or strings).
        - `out_structure`: A PyTree defining the desired output structure. It must
          contain the exact same set of tokens as `in_structure`.
        """
        self.in_treedef = jax.tree_util.tree_structure(in_structure)
        self.out_treedef = jax.tree_util.tree_structure(out_structure)

        flat_in = jax.tree_util.tree_leaves(in_structure)
        flat_out = jax.tree_util.tree_leaves(out_structure)

        if len(flat_in) != len(set(flat_in)):
            raise ValueError(
                f"in_structure cannot have duplicate tokens. Got: {flat_in}"
            )
        if len(flat_out) != len(set(flat_out)):
            raise ValueError(
                f"out_structure cannot have duplicate tokens. Got: {flat_out}"
            )
        if set(flat_in) != set(flat_out):
            raise ValueError(
                f"Structures are incompatible: in_structure tokens {set(flat_in)} "
                f"do not match out_structure tokens {set(flat_out)}."
            )

        # Pre-compute the routing permutations for fast forward/inverse passes
        self.forward_permutation = tuple(flat_in.index(token) for token in flat_out)
        self.inverse_permutation = tuple(flat_out.index(token) for token in flat_in)

    def forward_and_log_det(self, x: Any) -> tuple[Any, Array]:
        """Computes y = restructure(x) and log|det J(f)(x)| = 0."""
        if jax.tree_util.tree_structure(x) != self.in_treedef:
            raise ValueError("Input `x` does not match the expected `in_structure`.")

        flat_x = jax.tree_util.tree_leaves(x)
        flat_y = [flat_x[i] for i in self.forward_permutation]
        y = jax.tree_util.tree_unflatten(self.out_treedef, flat_y)

        # Pull dtype from the first leaf to match JAX array constraints
        dtype = flat_x[0].dtype if flat_x else jnp.float32
        return y, jnp.zeros((), dtype=dtype)

    def inverse_and_log_det(self, y: Any) -> tuple[Any, Array]:
        """Computes x = restructure^{-1}(y) and log|det J(f^{-1})(y)| = 0."""
        if jax.tree_util.tree_structure(y) != self.out_treedef:
            raise ValueError("Input `y` does not match the expected `out_structure`.")

        flat_y = jax.tree_util.tree_leaves(y)
        flat_x = [flat_y[i] for i in self.inverse_permutation]
        x = jax.tree_util.tree_unflatten(self.in_treedef, flat_x)

        dtype = flat_y[0].dtype if flat_y else jnp.float32
        return x, jnp.zeros((), dtype=dtype)

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return (
            type(other) is Restructure
            and self.in_treedef == other.in_treedef
            and self.out_treedef == other.out_treedef
            and self.forward_permutation == other.forward_permutation
        )
