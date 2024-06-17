from typing import Union

import jax
from jaxtyping import (
    PyTree,
)


EventT = Union[tuple[int], PyTree[jax.ShapeDtypeStruct]]
