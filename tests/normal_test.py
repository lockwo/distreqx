"""Tests for `normal.py`."""

from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import Normal


class NormalTest(TestCase):
    @parameterized.expand(
        [
            ("1d std normal", (jnp.array(0), jnp.array(1))),
            ("2d std normal", (jnp.zeros(2), jnp.ones(2))),
            ("rank 2 std normal", (jnp.zeros((3, 2)), jnp.ones((3, 2)))),
        ]
    )
    def test_event_shape(self, name, distr_params):
        loc, scale = distr_params
        self.assertEqual(loc.shape, Normal(loc, scale).event_shape)

    @parameterized.expand(
        [
            ("1d std normal", (jnp.array(0), jnp.array(1))),
            ("2d std normal", (jnp.zeros(2), jnp.ones(2))),
            ("rank 2 std normal", (jnp.zeros((3, 2)), jnp.ones((3, 2)))),
        ]
    )
    def test_sample_shape(self, name, distr_params):
        distr_params = (
            jnp.asarray(distr_params[0], dtype=jnp.float32),
            jnp.asarray(distr_params[1], dtype=jnp.float32),
        )
        key = jax.random.PRNGKey(0)
        self.assertEqual(
            distr_params[0].shape,
            Normal(distr_params[0], distr_params[1]).sample(key).shape,
        )

    @parameterized.expand(
        [
            ("1d std normal", (jnp.array(0), jnp.array(1))),
            ("2d std normal", (jnp.zeros(2), jnp.ones(2))),
            ("rank 2 std normal", (jnp.zeros((3, 2)), jnp.ones((3, 2)))),
        ]
    )
    @jax.numpy_rank_promotion("raise")
    def test_sample_and_log_prob(self, name, distr_params):
        distr_params = (
            jnp.asarray(distr_params[0], dtype=jnp.float32),
            jnp.asarray(distr_params[1], dtype=jnp.float32),
        )
        key = jax.random.PRNGKey(0)
        dist = Normal(distr_params[0], distr_params[1])
        result = dist.sample_and_log_prob(key)
        self.assertEqual(
            distr_params[0].shape,
            result[0].shape,
        )
        self.assertEqual(
            distr_params[0].shape,
            result[1].shape,
        )

    @parameterized.expand(
        [
            ("1d std normal", (jnp.array(0), jnp.array(1))),
            ("2d std normal", (jnp.zeros(2), jnp.ones(2))),
            ("rank 2 std normal", (jnp.zeros((3, 2)), jnp.ones((3, 2)))),
        ]
    )
    def test_method_with_input(self, name, distr_params):
        distr_params = (
            jnp.asarray(distr_params[0], dtype=jnp.float32),
            jnp.asarray(distr_params[1], dtype=jnp.float32),
        )
        value = jnp.asarray(distr_params[0], dtype=jnp.float32)
        dist = Normal(distr_params[0], distr_params[1])
        for method in [
            "log_prob",
            "prob",
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
            ("entropy", (0.0, 1.0), "entropy"),
            ("mean", (0, 1), "mean"),
            ("mean from 1d params", ([-1, 1], [1, 2]), "mean"),
            ("variance", (0, 1), "variance"),
            ("variance from np params", (np.ones(2), np.ones(2)), "variance"),
            ("stddev", (0, 1), "stddev"),
            ("stddev from rank 2 params", (np.ones((2, 3)), np.ones((2, 3))), "stddev"),
            ("mode", (0, 1), "mode"),
        ]
    )
    def test_method(self, name, distr_params, function_string):
        distr_params = (
            jnp.asarray(distr_params[0], dtype=jnp.float32),
            jnp.asarray(distr_params[1], dtype=jnp.float32),
        )
        dist = Normal(distr_params[0], distr_params[1])
        result = getattr(dist, function_string)()
        self.assertEqual(distr_params[0].shape, result.shape)

    @parameterized.expand(
        [
            ("no broadcast", ([0.0, 1.0, -0.5], [0.5, 1.0, 1.5])),
        ]
    )
    def test_median(self, name, distr_params):
        distr_params = (
            jnp.asarray(distr_params[0], dtype=jnp.float32),
            jnp.asarray(distr_params[1], dtype=jnp.float32),
        )
        dist = Normal(distr_params[0], distr_params[1])
        np.testing.assert_allclose(dist.median(), dist.mean(), rtol=1e-3)

    @parameterized.expand(
        [
            ("kl", "kl_divergence"),
            ("cross-ent", "cross_entropy"),
        ]
    )
    def test_with_two_distributions(self, name, function_string):
        dist1_kwargs = {
            "loc": jnp.array(np.random.randn(3, 2)),
            "scale": jnp.asarray([[0.8, 0.2], [0.1, 1.2], [1.4, 3.1]]),
        }
        dist2_kwargs = {
            "loc": jnp.array(np.random.randn(3, 2)),
            "scale": jnp.array(0.1 + np.random.rand(3, 2)),
        }
        dist1 = Normal(**dist1_kwargs)
        dist2 = Normal(**dist2_kwargs)

        result = getattr(dist1, function_string)(dist2)
        self.assertEqual(dist1_kwargs["loc"].shape, result.shape)
        result = getattr(dist1, function_string)(dist1)
        self.assertEqual(dist1_kwargs["loc"].shape, result.shape)
        if name == "kl":
            np.testing.assert_allclose(jnp.zeros_like(dist1_kwargs["loc"]), result)
        elif name == "cross-ent":
            np.testing.assert_allclose(dist1.entropy(), result)

    def test_vmap_inputs(self):
        def log_prob_sum(dist, x):
            return dist.log_prob(x).sum()

        dist = Normal(jnp.arange(3 * 4 * 5).reshape((3, 4, 5)), jnp.ones((3, 4, 5)))
        x = jnp.zeros((3, 4, 5))

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
        def summed_dist(loc, scale):
            return Normal(loc.sum(keepdims=True), scale.sum(keepdims=True))

        loc = jnp.arange((3 * 4 * 5)).reshape((3, 4, 5))
        scale = jnp.ones((3, 4, 5))

        actual = jax.vmap(summed_dist)(loc, scale)
        expected = Normal(
            loc.sum(axis=(1, 2), keepdims=True), scale.sum(axis=(1, 2), keepdims=True)
        )

        np.testing.assert_equal(actual.event_shape, expected.event_shape)

        x = jnp.array([[[1]], [[2]], [[3]]])
        np.testing.assert_allclose(actual.log_prob(x), expected.log_prob(x), rtol=1e-6)
