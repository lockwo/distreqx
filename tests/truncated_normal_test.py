"""Tests for `truncated_normal.py`."""

from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore
from scipy import stats

from distreqx.distributions import TruncatedNormal


class TruncatedNormalTest(TestCase):
    @parameterized.expand(
        [
            (
                "1d std trunc normal",
                (jnp.array(0.0), jnp.array(1.0), jnp.array(-1.0), jnp.array(1.0)),
            ),
            (
                "2d std trunc normal",
                (jnp.zeros(2), jnp.ones(2), -jnp.ones(2), jnp.ones(2)),
            ),
            (
                "rank 2 std trunc normal",
                (
                    jnp.zeros((3, 2)),
                    jnp.ones((3, 2)),
                    -jnp.ones((3, 2)),
                    jnp.ones((3, 2)),
                ),
            ),
        ]
    )
    def test_event_shape(self, name, distr_params):
        loc, scale, low, high = distr_params
        self.assertEqual(loc.shape, TruncatedNormal(loc, scale, low, high).event_shape)

    @parameterized.expand(
        [
            (
                "1d std trunc normal",
                (jnp.array(0.0), jnp.array(1.0), jnp.array(-1.0), jnp.array(1.0)),
            ),
            (
                "2d std trunc normal",
                (jnp.zeros(2), jnp.ones(2), -jnp.ones(2), jnp.ones(2)),
            ),
            (
                "rank 2 std trunc normal",
                (
                    jnp.zeros((3, 2)),
                    jnp.ones((3, 2)),
                    -jnp.ones((3, 2)),
                    jnp.ones((3, 2)),
                ),
            ),
        ]
    )
    def test_sample_shape(self, name, distr_params):
        distr_params = tuple(jnp.asarray(p, dtype=jnp.float32) for p in distr_params)
        key = jax.random.key(0)
        self.assertEqual(
            distr_params[0].shape,
            TruncatedNormal(*distr_params).sample(key).shape,
        )

    @parameterized.expand(
        [
            (
                "1d std trunc normal",
                (jnp.array(0.0), jnp.array(1.0), jnp.array(-1.0), jnp.array(1.0)),
            ),
            (
                "2d std trunc normal",
                (jnp.zeros(2), jnp.ones(2), -jnp.ones(2), jnp.ones(2)),
            ),
            (
                "rank 2 std trunc normal",
                (
                    jnp.zeros((3, 2)),
                    jnp.ones((3, 2)),
                    -jnp.ones((3, 2)),
                    jnp.ones((3, 2)),
                ),
            ),
        ]
    )
    @jax.numpy_rank_promotion("raise")
    def test_sample_and_log_prob(self, name, distr_params):
        distr_params = tuple(jnp.asarray(p, dtype=jnp.float32) for p in distr_params)
        key = jax.random.key(0)
        dist = TruncatedNormal(*distr_params)
        result = dist.sample_and_log_prob(key)
        self.assertEqual(distr_params[0].shape, result[0].shape)
        self.assertEqual(distr_params[0].shape, result[1].shape)

    @parameterized.expand(
        [
            (
                "1d std trunc normal",
                (jnp.array(0.0), jnp.array(1.0), jnp.array(-1.0), jnp.array(1.0)),
            ),
            (
                "2d std trunc normal",
                (jnp.zeros(2), jnp.ones(2), -jnp.ones(2), jnp.ones(2)),
            ),
            (
                "rank 2 std trunc normal",
                (
                    jnp.zeros((3, 2)),
                    jnp.ones((3, 2)),
                    -jnp.ones((3, 2)),
                    jnp.ones((3, 2)),
                ),
            ),
        ]
    )
    def test_method_with_input(self, name, distr_params):
        distr_params = tuple(jnp.asarray(p, dtype=jnp.float32) for p in distr_params)
        value = jnp.asarray(distr_params[0], dtype=jnp.float32)
        dist = TruncatedNormal(*distr_params)
        for method in [
            "log_prob",
            "cdf",
            "log_cdf",
            "survival_function",
            "log_survival_function",
        ]:
            with self.subTest(method):
                result = getattr(dist, method)(value)
                self.assertEqual(value.shape, result.shape)

    @parameterized.expand(
        [
            ("entropy", (0, 1, -1, 1), "entropy"),
            ("mean", (0, 1, -1, 1), "mean"),
            ("mean from 1d params", ([-1, 1], [1, 2], [-2, 0], [0, 2]), "mean"),
            ("variance", (0, 1, -1, 1), "variance"),
            (
                "variance from np params",
                (np.ones(2), np.ones(2), np.zeros(2), 2 * np.ones(2)),
                "variance",
            ),
            ("stddev", (0, 1, -1, 1), "stddev"),
            ("mode", (0, 1, -1, 1), "mode"),
            ("median", (0, 1, -1, 1), "median"),
        ]
    )
    def test_method(self, name, distr_params, function_string):
        distr_params = tuple(jnp.asarray(p, dtype=jnp.float32) for p in distr_params)
        dist = TruncatedNormal(*distr_params)
        result = getattr(dist, function_string)()
        self.assertEqual(distr_params[0].shape, result.shape)

    @parameterized.expand(
        [
            ("loc within bounds", 0.5, 1.5, -1.0, 2.0),
            ("loc outside bounds", 3.0, 1.0, -1.0, 1.0),
            ("negative loc", -2.0, 0.5, -3.0, -1.5),
        ]
    )
    def test_stats_match_scipy(self, name, loc, scale, low, high):
        # Cross-check the closed-form moment/entropy formulas against scipy's
        # reference implementation, not just their output shapes.
        a, b = (low - loc) / scale, (high - loc) / scale
        ref = stats.truncnorm(a, b, loc=loc, scale=scale)
        dist = TruncatedNormal(
            jnp.array(loc), jnp.array(scale), jnp.array(low), jnp.array(high)
        )

        np.testing.assert_allclose(dist.mean(), ref.mean(), rtol=1e-5)
        np.testing.assert_allclose(dist.variance(), ref.var(), rtol=1e-5)
        np.testing.assert_allclose(dist.stddev(), ref.std(), rtol=1e-5)
        np.testing.assert_allclose(dist.entropy(), ref.entropy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(dist.median(), ref.median(), rtol=1e-5)

        xs = np.linspace(low + 1e-3, high - 1e-3, 5)
        jxs = jnp.asarray(xs, dtype=jnp.float32)
        np.testing.assert_allclose(
            dist.log_prob(jxs),
            ref.logpdf(xs),  # pyright: ignore[reportAttributeAccessIssue]
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(dist.cdf(jxs), ref.cdf(xs), rtol=1e-4, atol=1e-6)

        expected_mode = np.clip(loc, low, high)
        np.testing.assert_allclose(dist.mode(), expected_mode, rtol=1e-5)

    @parameterized.expand(
        [
            ("standard", ([0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [1.0, 2.0])),
        ]
    )
    def test_median(self, name, distr_params):
        distr_params = tuple(jnp.asarray(p, dtype=jnp.float32) for p in distr_params)
        dist = TruncatedNormal(*distr_params)
        np.testing.assert_allclose(
            dist.median(), dist.icdf(jnp.array(0.5, dtype=jnp.float32)), rtol=1e-5
        )

    def test_kl_divergence_raises(self):
        dist1 = TruncatedNormal(
            jnp.array(0.0), jnp.array(1.0), jnp.array(-1.0), jnp.array(1.0)
        )
        dist2 = TruncatedNormal(
            jnp.array(0.0), jnp.array(1.0), jnp.array(-2.0), jnp.array(2.0)
        )
        with self.assertRaises(NotImplementedError):
            dist1.kl_divergence(dist2)

    @parameterized.expand(
        [
            (
                "1d std trunc normal",
                (jnp.array(0.0), jnp.array(1.0), jnp.array(-1.0), jnp.array(1.0)),
            ),
            (
                "2d std trunc normal",
                (jnp.zeros(2), jnp.ones(2), -jnp.ones(2), jnp.ones(2)),
            ),
            (
                "rank 2 std trunc normal",
                (
                    jnp.zeros((3, 2)),
                    jnp.ones((3, 2)),
                    -jnp.ones((3, 2)),
                    jnp.ones((3, 2)),
                ),
            ),
        ]
    )
    def test_icdf_shape(self, name, distr_params):
        distr_params = tuple(jnp.asarray(p, dtype=jnp.float32) for p in distr_params)
        value = 0.5 * jnp.ones_like(distr_params[0])
        dist = TruncatedNormal(*distr_params)
        result = dist.icdf(value)
        self.assertEqual(value.shape, result.shape)

    def test_icdf_values(self):
        loc = jnp.array([0.0, 1.0, -2.0])
        scale = jnp.array([1.0, 2.0, 0.5])
        low = jnp.array([-2.0, -1.0, -2.5])
        high = jnp.array([2.0, 3.0, -1.5])
        dist = TruncatedNormal(loc, scale, low, high)

        x = jnp.array([0.5, 1.0, -2.0])
        np.testing.assert_allclose(dist.icdf(dist.cdf(x)), x, rtol=1e-5)

        u = jnp.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(dist.cdf(dist.icdf(u)), u, rtol=1e-5)

    def test_out_of_bounds(self):
        dist = TruncatedNormal(
            loc=jnp.array(0.0),
            scale=jnp.array(1.0),
            low=jnp.array(-1.0),
            high=jnp.array(1.0),
        )

        expected_inf = jnp.array(-jnp.inf, dtype=jnp.float32)
        np.testing.assert_array_equal(dist.log_prob(jnp.array(1.5)), expected_inf)
        np.testing.assert_array_equal(dist.log_prob(jnp.array(-1.5)), expected_inf)

        np.testing.assert_allclose(dist.cdf(jnp.array(-1.5)), 0.0, atol=1e-6)
        np.testing.assert_allclose(dist.cdf(jnp.array(1.5)), 1.0, atol=1e-6)

        np.testing.assert_allclose(
            dist.survival_function(jnp.array(-1.5)), 1.0, atol=1e-6
        )
        np.testing.assert_allclose(
            dist.survival_function(jnp.array(1.5)), 0.0, atol=1e-6
        )

    def test_vmap_inputs(self):
        def log_prob_sum(dist, x):
            return dist.log_prob(x).sum()

        loc = jnp.arange(3 * 4 * 5).reshape((3, 4, 5))
        scale = jnp.ones((3, 4, 5))
        low = loc - 1.0
        high = loc + 1.0
        dist = TruncatedNormal(loc, scale, low, high)
        x = loc

        with self.subTest("no vmap"):
            actual = log_prob_sum(dist, x)
            expected = dist.log_prob(x).sum()
            np.testing.assert_allclose(actual, expected)

        with self.subTest("axis=0"):
            actual = jax.vmap(log_prob_sum, in_axes=0)(dist, x)
            expected = dist.log_prob(x).sum(axis=(1, 2))
            np.testing.assert_allclose(actual, expected)

        with self.subTest("axis=1"):
            actual = jax.vmap(log_prob_sum, in_axes=1)(dist, x)
            expected = dist.log_prob(x).sum(axis=(0, 2))
            np.testing.assert_allclose(actual, expected)

    def test_vmap_outputs(self):
        def summed_dist(loc, scale, low, high):
            return TruncatedNormal(
                loc.sum(keepdims=True),
                scale.sum(keepdims=True),
                low.sum(keepdims=True),
                high.sum(keepdims=True),
            )

        loc = jnp.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(jnp.float32)
        scale = jnp.ones((3, 4, 5))
        low = loc - 1.0
        high = loc + 1.0

        actual = jax.vmap(summed_dist)(loc, scale, low, high)
        expected = TruncatedNormal(
            loc.sum(axis=(1, 2), keepdims=True),
            scale.sum(axis=(1, 2), keepdims=True),
            low.sum(axis=(1, 2), keepdims=True),
            high.sum(axis=(1, 2), keepdims=True),
        )

        np.testing.assert_equal(actual.event_shape, expected.event_shape)

        x = jnp.array([[[1.0]], [[2.0]], [[3.0]]])
        np.testing.assert_equal(actual.log_prob(x).shape, expected.log_prob(x).shape)
