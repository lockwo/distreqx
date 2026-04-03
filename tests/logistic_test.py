"""Tests for `logistic.py`."""

from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import Logistic


class LogisticTest(TestCase):
    @parameterized.expand(
        [
            ("1d std logistic", (jnp.array(0.0), jnp.array(1.0))),
            ("2d std logistic", (jnp.zeros(2), jnp.ones(2))),
            ("rank 2 std logistic", (jnp.zeros((3, 2)), jnp.ones((3, 2)))),
        ]
    )
    def test_event_shape(self, name, distr_params):
        loc, scale = distr_params
        self.assertEqual(loc.shape, Logistic(loc, scale).event_shape)

    @parameterized.expand(
        [
            ("1d std logistic", (jnp.array(0.0), jnp.array(1.0))),
            ("2d std logistic", (jnp.zeros(2), jnp.ones(2))),
            ("rank 2 std logistic", (jnp.zeros((3, 2)), jnp.ones((3, 2)))),
        ]
    )
    def test_sample_shape(self, name, distr_params):
        distr_params = (
            jnp.asarray(distr_params[0], dtype=jnp.float32),
            jnp.asarray(distr_params[1], dtype=jnp.float32),
        )
        key = jax.random.key(0)
        self.assertEqual(
            distr_params[0].shape,
            Logistic(distr_params[0], distr_params[1]).sample(key).shape,
        )

    @parameterized.expand(
        [
            ("1d std logistic", (jnp.array(0.0), jnp.array(1.0))),
            ("2d std logistic", (jnp.zeros(2), jnp.ones(2))),
            ("rank 2 std logistic", (jnp.zeros((3, 2)), jnp.ones((3, 2)))),
        ]
    )
    @jax.numpy_rank_promotion("raise")
    def test_sample_and_log_prob(self, name, distr_params):
        distr_params = (
            jnp.asarray(distr_params[0], dtype=jnp.float32),
            jnp.asarray(distr_params[1], dtype=jnp.float32),
        )
        key = jax.random.key(0)
        dist = Logistic(distr_params[0], distr_params[1])
        samples, log_prob = dist.sample_and_log_prob(key)
        self.assertEqual(distr_params[0].shape, samples.shape)
        self.assertEqual(distr_params[0].shape, log_prob.shape)
        expected_log_prob = dist.log_prob(samples)
        np.testing.assert_allclose(log_prob, expected_log_prob, rtol=1e-5)

    @parameterized.expand(
        [
            ("1d std logistic", (jnp.array(0.0), jnp.array(1.0))),
            ("2d std logistic", (jnp.zeros(2), jnp.ones(2))),
            ("rank 2 std logistic", (jnp.zeros((3, 2)), jnp.ones((3, 2)))),
        ]
    )
    def test_method_with_input(self, name, distr_params):
        distr_params = (
            jnp.asarray(distr_params[0], dtype=jnp.float32),
            jnp.asarray(distr_params[1], dtype=jnp.float32),
        )
        value = jnp.asarray(distr_params[0], dtype=jnp.float32)
        dist = Logistic(distr_params[0], distr_params[1])
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
            ("mean", (0.0, 1.0), "mean"),
            ("mean from 1d params", ([-1.0, 1.0], [1.0, 2.0]), "mean"),
            ("variance", (0.0, 1.0), "variance"),
            ("variance from np params", (np.ones(2), np.ones(2)), "variance"),
            ("stddev", (0.0, 1.0), "stddev"),
            (
                "stddev from rank 2 params",
                (np.ones((2, 3)), np.ones((2, 3))),
                "stddev",
            ),
            ("mode", (0.0, 1.0), "mode"),
        ]
    )
    def test_method(self, name, distr_params, function_string):
        distr_params = (
            jnp.asarray(distr_params[0], dtype=jnp.float32),
            jnp.asarray(distr_params[1], dtype=jnp.float32),
        )
        dist = Logistic(distr_params[0], distr_params[1])
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
        dist = Logistic(distr_params[0], distr_params[1])
        np.testing.assert_allclose(dist.median(), dist.mean(), rtol=1e-3)

    @parameterized.expand(
        [
            ("1d std logistic", (jnp.array(0.0), jnp.array(1.0))),
            ("2d std logistic", (jnp.zeros(2), jnp.ones(2))),
            ("rank 2 std logistic", (jnp.zeros((3, 2)), jnp.ones((3, 2)))),
        ]
    )
    def test_icdf_shape(self, name, distr_params):
        distr_params = (
            jnp.asarray(distr_params[0], dtype=jnp.float32),
            jnp.asarray(distr_params[1], dtype=jnp.float32),
        )
        value = 0.5 * jnp.ones_like(distr_params[0])
        dist = Logistic(distr_params[0], distr_params[1])
        result = dist.icdf(value)
        self.assertEqual(value.shape, result.shape)

    def test_icdf_values(self):
        loc = jnp.array([0.0, 1.0, -2.0])
        scale = jnp.array([1.0, 2.0, 0.5])
        dist = Logistic(loc, scale)

        # icdf(cdf(x)) should be x
        x = jnp.array([0.5, -1.0, -1.5])
        np.testing.assert_allclose(dist.icdf(dist.cdf(x)), x, rtol=1e-5)

        # cdf(icdf(u)) should be u
        u = jnp.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(dist.cdf(dist.icdf(u)), u, rtol=1e-5)

    def test_cdf_values(self):
        dist = Logistic(jnp.array(0.0), jnp.array(1.0))
        np.testing.assert_allclose(dist.cdf(jnp.array(0.0)), 0.5, rtol=1e-5)

    def test_entropy_values(self):
        dist = Logistic(jnp.array(0.0), jnp.array(1.0))
        np.testing.assert_allclose(dist.entropy(), 2.0, rtol=1e-5)


    def test_log_cdf_consistency(self):
        """log_cdf should be log of cdf."""
        loc = jnp.array([0.0, 1.0])
        scale = jnp.array([1.0, 2.0])
        dist = Logistic(loc, scale)
        x = jnp.array([0.5, -1.0])
        np.testing.assert_allclose(
            dist.log_cdf(x), jnp.log(dist.cdf(x)), rtol=1e-5
        )

    def test_vmap_inputs(self):
        def log_prob_sum(dist, x):
            return dist.log_prob(x).sum()

        dist = Logistic(
            jnp.arange(3 * 4 * 5, dtype=jnp.float32).reshape((3, 4, 5)),
            jnp.ones((3, 4, 5)),
        )
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
            return Logistic(loc.sum(keepdims=True), scale.sum(keepdims=True))

        loc = jnp.arange(3 * 4 * 5, dtype=jnp.float32).reshape((3, 4, 5))
        scale = jnp.ones((3, 4, 5))

        actual = jax.vmap(summed_dist)(loc, scale)
        expected = Logistic(
            loc.sum(axis=(1, 2), keepdims=True),
            scale.sum(axis=(1, 2), keepdims=True),
        )

        np.testing.assert_equal(actual.event_shape, expected.event_shape)

        x = jnp.array([[[1.0]], [[2.0]], [[3.0]]])
        np.testing.assert_allclose(
            actual.log_prob(x), expected.log_prob(x), rtol=1e-6
        )

    def test_jit(self):
        dist = Logistic(jnp.array(0.0), jnp.array(1.0))
        x = jnp.array(0.5)
        jitted_log_prob = eqx.filter_jit(dist.log_prob)
        np.testing.assert_allclose(
            jitted_log_prob(x), dist.log_prob(x), rtol=1e-5
        )
