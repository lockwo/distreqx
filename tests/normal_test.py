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
        key = jax.random.key(0)
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
        key = jax.random.key(0)
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

    @parameterized.expand(
        [
            ("1d std normal", (jnp.array(0.0), jnp.array(1.0))),
            ("2d std normal", (jnp.zeros(2), jnp.ones(2))),
            ("rank 2 std normal", (jnp.zeros((3, 2)), jnp.ones((3, 2)))),
        ]
    )
    def test_icdf_shape(self, name, distr_params):
        distr_params = (
            jnp.asarray(distr_params[0], dtype=jnp.float32),
            jnp.asarray(distr_params[1], dtype=jnp.float32),
        )
        value = 0.5 * jnp.ones_like(distr_params[0])
        dist = Normal(distr_params[0], distr_params[1])
        result = dist.icdf(value)
        self.assertEqual(value.shape, result.shape)

    def test_icdf_values(self):
        loc = jnp.array([0.0, 1.0, -2.0])
        scale = jnp.array([1.0, 2.0, 0.5])
        dist = Normal(loc, scale)

        # icdf(cdf(x)) should be x
        x = jnp.array([0.5, -1.0, -1.5])
        np.testing.assert_allclose(dist.icdf(dist.cdf(x)), x, rtol=1e-5)

        # cdf(icdf(u)) should be u
        u = jnp.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(dist.cdf(dist.icdf(u)), u, rtol=1e-5)

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

    # ===== Compatibility: jit =====

    def test_jit_sample(self):
        """Verify that sample works under jit compilation."""
        dist = Normal(jnp.array(0.0), jnp.array(1.0))
        jitted_sample = jax.jit(dist.sample)
        result = jitted_sample(jax.random.key(0))
        self.assertEqual(result.shape, ())

    def test_jit_log_prob(self):
        """Verify that log_prob works under jit compilation."""
        dist = Normal(jnp.zeros(3), jnp.ones(3))
        jitted_log_prob = jax.jit(dist.log_prob)
        result = jitted_log_prob(jnp.array([0.0, 1.0, -1.0]))
        self.assertEqual(result.shape, (3,))

    def test_jit_cdf(self):
        """Verify that cdf works under jit compilation."""
        dist = Normal(jnp.array(0.0), jnp.array(1.0))
        jitted_cdf = jax.jit(dist.cdf)
        result = jitted_cdf(jnp.array(0.0))
        np.testing.assert_allclose(result, 0.5, rtol=1e-5)

    def test_jit_entropy(self):
        """Verify that entropy works under jit compilation."""
        dist = Normal(jnp.array(0.0), jnp.array(1.0))
        jitted_entropy = jax.jit(dist.entropy)
        result = jitted_entropy()
        # Entropy of standard normal = 0.5 * log(2 * pi * e)
        expected = 0.5 * jnp.log(2 * jnp.pi * jnp.e)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    # ===== Compatibility: grad =====

    def test_grad_log_prob_wrt_value(self):
        """Verify gradient of log_prob with respect to value."""
        dist = Normal(jnp.array(0.0), jnp.array(1.0))

        def scalar_log_prob(x):
            return dist.log_prob(x)

        grad_fn = jax.grad(scalar_log_prob)
        # For standard normal, d/dx log_prob(x) = -x
        grad_at_1 = grad_fn(jnp.array(1.0))
        np.testing.assert_allclose(grad_at_1, -1.0, rtol=1e-5)

        grad_at_0 = grad_fn(jnp.array(0.0))
        np.testing.assert_allclose(grad_at_0, 0.0, atol=1e-5)

    def test_grad_log_prob_wrt_loc(self):
        """Verify gradient of log_prob with respect to loc parameter."""

        def log_prob_from_loc(loc):
            dist = Normal(loc, jnp.array(1.0))
            return dist.log_prob(jnp.array(0.5))

        grad_fn = jax.grad(log_prob_from_loc)
        # d/d_mu log N(x; mu, 1) = (x - mu) / sigma^2 = (0.5 - mu)
        grad_at_0 = grad_fn(jnp.array(0.0))
        np.testing.assert_allclose(grad_at_0, 0.5, rtol=1e-5)

    def test_grad_log_prob_wrt_scale(self):
        """Verify gradient of log_prob with respect to scale parameter."""

        def log_prob_from_scale(scale):
            dist = Normal(jnp.array(0.0), scale)
            return dist.log_prob(jnp.array(1.0))

        grad_fn = jax.grad(log_prob_from_scale)
        # d/d_sigma log N(x; 0, sigma) = (x^2 / sigma^3) - 1/sigma
        # At sigma=1, x=1: 1/1 - 1/1 = 0
        grad_at_1 = grad_fn(jnp.array(1.0))
        np.testing.assert_allclose(grad_at_1, 0.0, atol=1e-5)

    def test_grad_entropy_wrt_scale(self):
        """Verify gradient of entropy with respect to scale."""

        def entropy_from_scale(scale):
            dist = Normal(jnp.array(0.0), scale)
            return dist.entropy()

        grad_fn = jax.grad(entropy_from_scale)
        # d/d_sigma H = 1/sigma
        grad_at_2 = grad_fn(jnp.array(2.0))
        np.testing.assert_allclose(grad_at_2, 0.5, rtol=1e-5)

    # ===== Correctness: analytic solutions =====

    def test_log_prob_standard_normal_at_zero(self):
        """log_prob of standard normal at x=0 equals -0.5*log(2*pi)."""
        dist = Normal(jnp.array(0.0), jnp.array(1.0))
        expected = -0.5 * jnp.log(2 * jnp.pi)
        np.testing.assert_allclose(dist.log_prob(jnp.array(0.0)), expected, rtol=1e-5)

    def test_entropy_standard_normal(self):
        """Entropy of standard normal equals 0.5*log(2*pi*e)."""
        dist = Normal(jnp.array(0.0), jnp.array(1.0))
        expected = 0.5 * jnp.log(2 * jnp.pi * jnp.e)
        np.testing.assert_allclose(dist.entropy(), expected, rtol=1e-5)

    def test_cdf_at_mean_equals_half(self):
        """CDF at the mean should equal 0.5 for any Normal distribution."""
        loc = jnp.array([1.0, -2.0, 5.0])
        scale = jnp.array([0.5, 3.0, 0.1])
        dist = Normal(loc, scale)
        np.testing.assert_allclose(dist.cdf(loc), 0.5 * jnp.ones(3), rtol=1e-5)

    def test_variance_equals_scale_squared(self):
        """Variance should equal scale^2."""
        scale = jnp.array([0.5, 1.0, 2.0, 10.0])
        dist = Normal(jnp.zeros(4), scale)
        np.testing.assert_allclose(dist.variance(), scale**2, rtol=1e-6)

    def test_kl_divergence_to_self_is_zero(self):
        """KL divergence of a distribution to itself should be zero."""
        dist = Normal(jnp.array([1.0, 2.0]), jnp.array([0.5, 1.5]))
        kl = dist.kl_divergence(dist)
        np.testing.assert_allclose(kl, jnp.zeros(2), atol=1e-6)

    # ===== Edge cases =====

    def test_large_values(self):
        """Distribution handles large input values without NaN."""
        dist = Normal(jnp.array(0.0), jnp.array(1.0))
        large_x = jnp.array(100.0)
        log_p = dist.log_prob(large_x)
        self.assertFalse(jnp.isnan(log_p))
        # log_prob of very extreme values should be very negative
        self.assertTrue(log_p < -1000.0)

    def test_small_scale(self):
        """Distribution handles very small scale without NaN in log_prob."""
        dist = Normal(jnp.array(0.0), jnp.array(1e-6))
        log_p = dist.log_prob(jnp.array(0.0))
        self.assertFalse(jnp.isnan(log_p))
        # With very small scale, log_prob at the mean should be very large (peaked)
        self.assertTrue(log_p > 10.0)

    def test_large_scale(self):
        """Distribution handles very large scale."""
        dist = Normal(jnp.array(0.0), jnp.array(1e6))
        log_p = dist.log_prob(jnp.array(0.0))
        self.assertFalse(jnp.isnan(log_p))
        # With very large scale, log_prob should be very negative (flat)
        self.assertTrue(log_p < -10.0)

    def test_float64_dtype(self):
        """Distribution works with float64 inputs."""
        prev = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        try:
            loc = jnp.array(0.0, dtype=jnp.float64)
            scale = jnp.array(1.0, dtype=jnp.float64)
            dist = Normal(loc, scale)
            sample = dist.sample(jax.random.key(0))
            self.assertEqual(sample.dtype, jnp.float64)
        finally:
            jax.config.update("jax_enable_x64", prev)

    def test_negative_values_in_sample(self):
        """Samples can be negative (support is all reals)."""
        dist = Normal(jnp.array(-10.0), jnp.array(0.1))
        samples = jax.vmap(dist.sample)(jax.random.split(jax.random.key(0), 100))
        # With loc=-10 and small scale, all samples should be negative
        self.assertTrue(jnp.all(samples < 0))
