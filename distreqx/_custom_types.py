from typing import Tuple, Union

import jax
from jaxtyping import (
    PyTree,
)


EventT = Union[Tuple[int], PyTree[jax.ShapeDtypeStruct]]
