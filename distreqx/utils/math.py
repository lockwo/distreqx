"""Utility math functions."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array


@jax.custom_jvp
def multiply_no_nan(x: Array, y: Array) -> Array:
    """Computes the element-wise product of `x` and `y`, returning 0
    where `y` is zero, even if `x` is NaN or infinite.

    **Arguments:**

    - `x`: First input.
    - `y`: Second input.

    **Returns:**

    - The product of `x` and `y`.

    **Raises:**

    - ValueError if the shapes of `x` and `y` do not match.
    """
    dtype = jnp.result_type(x, y)
    return jnp.where(y == 0, jnp.zeros((), dtype=dtype), x * y)


@multiply_no_nan.defjvp
def multiply_no_nan_jvp(
    primals: Tuple[Array, Array], tangents: Tuple[Array, Array]
) -> Tuple[Array, Array]:
    """Custom gradient computation for `multiply_no_nan`.

    **Arguments:**

    - `primals`: A tuple containing the primal values of `x` and `y`.
    - `tangents`: A tuple containing the tangent values of `x` and `y`.

    **Returns:**

    - A tuple containing the output of the primal and tangent operations.
    """
    x, y = primals
    x_dot, y_dot = tangents
    primal_out = multiply_no_nan(x, y)
    tangent_out = y * x_dot + x * y_dot
    return primal_out, tangent_out


@jax.custom_jvp
def power_no_nan(x: Array, y: Array) -> Array:
    """Computes `x ** y`, ensuring the result is 1.0 when `y` is zero,
    following the convention `0 ** 0 = 1`.

    **Arguments:**

    - `x`: First input.
    - `y`: Second input.

    **Returns:**

    - The power `x ** y`.
    """
    dtype = jnp.result_type(x, y)
    return jnp.where(y == 0, jnp.ones((), dtype=dtype), jnp.power(x, y))


@power_no_nan.defjvp
def power_no_nan_jvp(
    primals: Tuple[Array, Array], tangents: Tuple[Array, Array]
) -> Tuple[Array, Array]:
    """Custom gradient computation for `power_no_nan`.

    **Arguments:**

    - `primals`: A tuple containing the primal values of `x` and `y`.
    - `tangents`: A tuple containing the tangent values of `x` and `y`.

    **Returns:**

    - A tuple containing the output of the primal and tangent operations.
    """
    x, y = primals
    x_dot, y_dot = tangents
    primal_out = power_no_nan(x, y)
    tangent_out = y * power_no_nan(x, y - 1) * x_dot + primal_out * jnp.log(x) * y_dot
    return primal_out, tangent_out


def mul_exp(x: Array, logp: Array) -> Array:
    """Returns `x * exp(logp)` with zero output if `exp(logp) == 0`.

    **Arguments:**

    - `x`: An array.
    - `logp`: An array representing logarithms.

    **Returns:**

    - The result of `x * exp(logp)`.
    """
    p = jnp.exp(logp)
    x = jnp.where(p == 0, 0.0, x)
    return x * p


def normalize(
    *, probs: Optional[Array] = None, logits: Optional[Array] = None
) -> Array:
    """Normalizes logits via log_softmax or probabilities to ensure they sum to one.

    **Arguments:**

    - `probs`: Probability values.
    - `logits`: Logit values.

    **Returns:**

    - Normalized probabilities or logits.
    """
    if logits is None:
        if probs is None:
            raise ValueError("both logits and probs are None!")
        probs = jnp.asarray(probs)
        return probs / probs.sum(axis=-1, keepdims=True)
    else:
        if logits is None:
            raise ValueError("both logits and probs are None!")
        logits = jnp.asarray(logits)
        return jax.nn.log_softmax(logits, axis=-1)


def sum_last(x: Array, ndims: int) -> Array:
    """Sums the last `ndims` axes of array `x`.

    **Arguments:**

    - `x`: An array.
    - `ndims`: The number of last dimensions to sum.

    **Returns:**

    - The sum of the last `ndims` dimensions of `x`.
    """
    axes_to_sum = tuple(range(-ndims, 0))
    return jnp.sum(x, axis=axes_to_sum)


def log_expbig_minus_expsmall(big: Array, small: Array) -> Array:
    """Stable implementation of `log(exp(big) - exp(small))`.

    **Arguments:**

    - `big`: First input.
    - `small`: Second input. It must be `small <= big`.

    **Returns:**

    - The resulting `log(exp(big) - exp(small))`.
    """
    return big + jnp.log1p(-jnp.exp(small - big))


def log_beta(a: Array, b: Array) -> Array:
    """Obtains the log of the beta function `log B(a, b)`.

    **Arguments:**

    - `a`: First input. It must be positive.
    - `b`: Second input. It must be positive.

    **Returns:**

    - The value `log B(a, b) = log Gamma(a) + log Gamma(b) - log Gamma(a + b)`,
      where `Gamma` is the Gamma function, obtained through stable
      computation of `log Gamma`.
    """
    return jax.lax.lgamma(a) + jax.lax.lgamma(b) - jax.lax.lgamma(a + b)


def log_beta_multivariate(a: Array) -> Array:
    """Obtains the log of the multivariate beta function `log B(a)`.

    **Arguments:**

    - `a`: An array of length `K` containing positive values.

    **Returns:**

    - The value
      `log B(a) = sum_{k=1}^{K} log Gamma(a_k) - log Gamma(sum_{k=1}^{K} a_k)`,
      where `Gamma` is the Gamma function, obtained through stable
      computation of `log Gamma`.
    """
    return jnp.sum(jax.lax.lgamma(a), axis=-1) - jax.lax.lgamma(jnp.sum(a, axis=-1))
