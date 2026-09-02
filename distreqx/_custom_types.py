import jax
from jaxtyping import PyTree

EventT = tuple[int] | PyTree[jax.ShapeDtypeStruct]
