from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import _distribution


class AbstractAll(
    _distribution.AbstractSurvivalDistribution,
    _distribution.AbstractSTDDistribution,
    _distribution.AbstractSampleLogProbDistribution,
    _distribution.AbstractCDFDistribution,
    _distribution.AbstractProbDistribution,
    strict=True,
):
    pass


class DummyUnivariateDist(AbstractAll):
    """Dummy univariate distribution for testing."""

    def sample(self, key):
        return jax.random.uniform(key)

    def log_prob(self, value):
        """Log probability density/mass function."""

    @property
    def event_shape(self):
        return (1,)

    def entropy(self):
        raise NotImplementedError

    def kl_divergence(self, other_dist, **kwargs):
        raise NotImplementedError

    def log_cdf(self, value):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def median(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError


class DummyMultivariateDist(AbstractAll):
    """Dummy multivariate distribution for testing."""

    _dimension: tuple

    def sample(self, key):
        return jax.random.uniform(key, shape=self._dimension)

    def log_prob(self, value):
        """Log probability density/mass function."""

    @property
    def event_shape(self):
        return self._dimension

    def entropy(self):
        raise NotImplementedError

    def kl_divergence(self, other_dist, **kwargs):
        raise NotImplementedError

    def log_cdf(self, value):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def median(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError


class DistributionTest(TestCase):
    def setUp(self):
        self.uni_dist = DummyUnivariateDist()

    def test_sample_univariate_shape(self):
        sample_fn = self.uni_dist.sample
        samples = sample_fn(jax.random.key(0))
        np.testing.assert_equal(samples.shape, ())

    @parameterized.expand(
        [
            ("single dimension", (5,), (5,)),
            ("single dimension repeat", (5,), (5,)),
            ("two dimensions", (4, 5), (4, 5)),
        ]
    )
    def test_sample_multivariate_shape(self, name, var_dim, expected_shape):
        mult_dist = DummyMultivariateDist(var_dim)
        sample_fn = mult_dist.sample
        samples = sample_fn(jax.random.key(0))
        np.testing.assert_equal(samples.shape, expected_shape)

    def test_jittable(self):
        dist = DummyMultivariateDist((5,))
        sampler = jax.jit(dist.sample)
        seed = jax.random.key(0)
        np.testing.assert_array_equal(sampler(seed), dist.sample(seed))

    def test_multivariate_survival_function_raises(self):
        mult_dist = DummyMultivariateDist((42,))
        with self.assertRaises(NotImplementedError):
            mult_dist.survival_function(jnp.zeros(42))
        with self.assertRaises(NotImplementedError):
            mult_dist.log_survival_function(jnp.zeros(42))
