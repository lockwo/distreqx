"""Truncated Normal distribution."""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from ._distribution import AbstractProbDistribution

_half_log2pi = 0.5 * math.log(2 * math.pi)
_inv_sqrt_2pi = 1.0 / math.sqrt(2 * math.pi)


class TruncatedNormal(AbstractProbDistribution):
    """
    Truncated Normal distribution with `loc`, `scale`,
    `low`, and `high` parameters.
    """

    loc: Array
    scale: Array
    low: Array
    high: Array

    def __init__(self, loc: Array, scale: Array, low: Array, high: Array):
        """Initializes a Truncated Normal distribution.

        **Arguments:**

        - `loc`: Mean of the untruncated distribution.
        - `scale`: Standard deviation of the untruncated distribution.
        - `low`: Lower bound of the truncation.
        - `high`: Upper bound of the truncation.
        """
        self.loc = jnp.array(loc)
        self.scale = jnp.array(scale)
        self.low = jnp.array(low)
        self.high = jnp.array(high)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of event of distribution samples."""
        return jnp.broadcast_shapes(
            self.loc.shape, self.scale.shape, self.low.shape, self.high.shape
        )

    @property
    def support(self) -> tuple[Array, Array]:
        """See `Distribution.support`."""
        return (self.low, self.high)

    def _standardize(self, value: Array) -> Array:
        return (value - self.loc) / self.scale

    @property
    def _std_low(self) -> Array:
        return self._standardize(self.low)

    @property
    def _std_high(self) -> Array:
        return self._standardize(self.high)

    def _log_normalizer(self) -> Array:
        """Calculates the log of the normalization constant Z."""
        cdf_high = jax.scipy.special.ndtr(self._std_high)
        cdf_low = jax.scipy.special.ndtr(self._std_low)
        return jnp.log(cdf_high - cdf_low)

    def _sample_from_std_trunc_normal(self, key: Key[Array, ""]) -> Array:
        dtype = jnp.result_type(self.loc, self.scale, self.low, self.high)
        return jax.random.truncated_normal(
            key,
            lower=self._std_low,
            upper=self._std_high,
            shape=self.event_shape,
            dtype=dtype,
        )

    def sample(self, key: Key[Array, ""]) -> Array:
        """See `Distribution.sample`."""
        rnd = self._sample_from_std_trunc_normal(key)
        return self.scale * rnd + self.loc

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[Array, Array]:
        """See `Distribution.sample_and_log_prob`."""
        rnd = self._sample_from_std_trunc_normal(key)
        samples = self.scale * rnd + self.loc

        log_unnormalized = -0.5 * jnp.square(rnd) - _half_log2pi - jnp.log(self.scale)
        log_prob = log_unnormalized - self._log_normalizer()
        return samples, log_prob

    def log_prob(self, value: Array) -> Array:
        """See `Distribution.log_prob`."""
        z = self._standardize(value)
        log_unnormalized = -0.5 * jnp.square(z) - _half_log2pi - jnp.log(self.scale)
        log_prob_val = log_unnormalized - self._log_normalizer()

        # Prob is 0 (log_prob is -inf) outside the bounds
        is_valid = (value >= self.low) & (value <= self.high)
        return jnp.where(is_valid, log_prob_val, -jnp.inf)

    def cdf(self, value: Array) -> Array:
        """See `Distribution.cdf`."""
        z = self._standardize(value)
        cdf_unnormalized = jax.scipy.special.ndtr(z) - jax.scipy.special.ndtr(
            self._std_low
        )
        Z = jax.scipy.special.ndtr(self._std_high) - jax.scipy.special.ndtr(
            self._std_low
        )
        cdf_val = cdf_unnormalized / Z

        # Clamp out-of-bounds CDF values
        cdf_val = jnp.where(value < self.low, jnp.zeros_like(cdf_val), cdf_val)
        cdf_val = jnp.where(value > self.high, jnp.ones_like(cdf_val), cdf_val)
        return cdf_val

    def log_cdf(self, value: Array) -> Array:
        """See `Distribution.log_cdf`."""
        return jnp.log(self.cdf(value))

    def icdf(self, value: Array) -> Array:
        """See `Distribution.icdf`."""
        cdf_low = jax.scipy.special.ndtr(self._std_low)
        cdf_high = jax.scipy.special.ndtr(self._std_high)
        Z = cdf_high - cdf_low
        p_unnormalized = value * Z + cdf_low
        return self.loc + self.scale * jax.scipy.special.ndtri(p_unnormalized)

    def survival_function(self, value: Array) -> Array:
        """See `Distribution.survival_function`."""
        z = self._standardize(value)
        Z = jax.scipy.special.ndtr(self._std_high) - jax.scipy.special.ndtr(
            self._std_low
        )
        surv_unnormalized = jax.scipy.special.ndtr(
            self._std_high
        ) - jax.scipy.special.ndtr(z)
        surv_val = surv_unnormalized / Z

        # Clamp out-of-bounds survival values
        surv_val = jnp.where(value < self.low, jnp.ones_like(surv_val), surv_val)
        surv_val = jnp.where(value > self.high, jnp.zeros_like(surv_val), surv_val)
        return surv_val

    def log_survival_function(self, value: Array) -> Array:
        """See `Distribution.log_survival_function`."""
        return jnp.log(self.survival_function(value))

    def _pdf(self, value: Array) -> Array:
        """Standard normal PDF helper for mean/variance/entropy calculations."""
        return _inv_sqrt_2pi * jnp.exp(-0.5 * jnp.square(value))

    def entropy(self) -> Array:
        """Calculates the Shannon entropy (in nats)."""
        alpha = self._std_low
        beta = self._std_high
        Z = jax.scipy.special.ndtr(beta) - jax.scipy.special.ndtr(alpha)

        pdf_alpha = self._pdf(alpha)
        pdf_beta = self._pdf(beta)

        term1 = jnp.log(self.scale * Z) + 0.5 * math.log(2 * math.pi * math.e)
        term2 = (alpha * pdf_alpha - beta * pdf_beta) / (2.0 * Z)
        return term1 + term2

    def mean(self) -> Array:
        """Calculates the mean."""
        alpha = self._std_low
        beta = self._std_high
        Z = jax.scipy.special.ndtr(beta) - jax.scipy.special.ndtr(alpha)

        pdf_alpha = self._pdf(alpha)
        pdf_beta = self._pdf(beta)

        return self.loc + self.scale * ((pdf_alpha - pdf_beta) / Z)

    def variance(self) -> Array:
        """Calculates the variance."""
        alpha = self._std_low
        beta = self._std_high
        Z = jax.scipy.special.ndtr(beta) - jax.scipy.special.ndtr(alpha)

        pdf_alpha = self._pdf(alpha)
        pdf_beta = self._pdf(beta)

        term1 = 1.0
        term2 = (alpha * pdf_alpha - beta * pdf_beta) / Z
        term3 = jnp.square((pdf_alpha - pdf_beta) / Z)

        return jnp.square(self.scale) * (term1 + term2 - term3)

    def stddev(self) -> Array:
        """Calculates the standard deviation."""
        return jnp.sqrt(self.variance())

    def mode(self) -> Array:
        """Calculates the mode."""
        # For a truncated normal, the mode is the mean if the mean is within bounds,
        # otherwise it is the bound closest to the mean.
        return jnp.clip(self.loc, self.low, self.high)

    def median(self) -> Array:
        """Calculates the median."""
        return self.icdf(jnp.array(0.5, dtype=self.loc.dtype))

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates the KL divergence to another distribution.

        Raises:
            NotImplementedError: The KL divergence between arbitrary truncated normal
            distributions lacks a stable general analytical solution.
        """
        raise NotImplementedError(
            "KL divergence for TruncatedNormal is not analytically tractable."
        )
