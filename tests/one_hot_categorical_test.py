"""Tests for `one_hot_categorical.py`."""

from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import one_hot_categorical
from distreqx.utils import math


class OneHotCategoricalTest(TestCase):
    def setUp(self):
        self.dist = one_hot_categorical.OneHotCategorical
        self.key = jax.random.PRNGKey(0)
        self.p = jnp.asarray([0.1, 0.4, 0.2, 0.3])
        self.logits = jnp.log(self.p) - 1.0  # intended unnormalization
        self.assertion_fn = lambda x: lambda a, b: self.assertTrue(
            np.allclose(a, b, rtol=x)
        )

    def test_parameters_from_probs(self):
        dist = self.dist(probs=self.p)
        self.assertTrue(
            np.allclose(dist.logits, math.normalize(logits=jnp.log(self.p)), rtol=2e-3)
        )
        self.assertTrue(
            np.allclose(dist.probs, math.normalize(probs=self.p), rtol=2e-3)
        )

    def test_parameters_from_logits(self):
        dist = self.dist(logits=self.logits)
        self.assertTrue(
            np.allclose(dist.logits, math.normalize(logits=self.logits), rtol=2e-3)
        )
        self.assertTrue(
            np.allclose(dist.probs, math.normalize(probs=self.p), rtol=2e-3)
        )

    @parameterized.expand(
        [
            ("1d probs", (4,), True),
            ("1d logits", (4,), False),
            ("2d probs", (3, 4), True),
            ("2d logits", (3, 4), False),
        ]
    )
    def test_properties(self, name, shape, from_probs):
        rng = np.random.default_rng(42)
        probs = rng.uniform(size=shape)  # Intentional unnormalization of `probs`.
        logits = np.log(probs)
        dist_kwargs = (
            {"probs": jnp.array(probs)} if from_probs else {"logits": jnp.array(logits)}
        )
        dist = self.dist(**dist_kwargs)
        self.assertEqual(dist.event_shape, (shape[-1],))
        self.assertEqual(dist.num_categories, shape[-1])
        np.testing.assert_allclose(
            dist.logits, math.normalize(logits=jnp.array(logits)), rtol=1e-3
        )
        np.testing.assert_allclose(
            dist.probs, math.normalize(probs=jnp.array(probs)), rtol=1e-3
        )

    @parameterized.expand(
        [
            (
                "probs and logits",
                {"logits": jnp.array([0.1, -0.2]), "probs": jnp.array([0.6, 0.4])},
            ),
            ("both probs and logits are None", {"logits": None, "probs": None}),
        ]
    )
    def test_raises_on_invalid_inputs(self, name, dist_params):
        with self.assertRaises(ValueError):
            self.dist(**dist_params)

    @parameterized.expand(
        [
            ("1d logits, no shape", {"logits": jnp.array([0.0, 1.0, -0.5])}, (3,)),
            ("1d probs, no shape", {"probs": jnp.array([0.2, 0.5, 0.3])}, (3,)),
            (
                "2d logits, 1-tuple shape",
                {"logits": jnp.array([[0.0, 1.0, -0.5], [-0.1, 0.3, 0.0]])},
                (2, 3),
            ),
            (
                "2d probs, 1-tuple shape",
                {"probs": jnp.array([[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]])},
                (2, 3),
            ),
        ]
    )
    def test_sample_shape(self, name, distr_params, sample_shape):
        distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
        dist = self.dist(**distr_params)
        samples = dist.sample(self.key)
        self.assertEqual(samples.shape, sample_shape)

    @parameterized.expand(
        [
            ("1d logits, no shape", {"logits": jnp.array([0.0, 1.0, -0.5])}, (3,)),
            ("1d probs, no shape", {"probs": jnp.array([0.2, 0.5, 0.3])}, (3,)),
        ]
    )
    def test_sample_and_log_prob(self, name, distr_params, sample_shape):
        distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
        dist = self.dist(**distr_params)
        samples, log_prob = dist.sample_and_log_prob(self.key)
        self.assertTupleEqual(samples.shape, sample_shape)
        self.assertTupleEqual(log_prob.shape, ())

    @parameterized.expand(
        [
            ("sample, float16", "sample", jnp.float16),
            ("sample, float32", "sample", jnp.float32),
            ("sample_and_log_prob, float16", "sample_and_log_prob", jnp.float16),
            ("sample_and_log_prob, float32", "sample_and_log_prob", jnp.float32),
        ]
    )
    def test_sample_dtype(self, name, method, dtype):
        dist_params = {"logits": jnp.array([0.1, -0.1, 0.5, -0.8, 1.5]).astype(dtype)}
        dist = self.dist(**dist_params)
        samples = getattr(dist, method)(self.key)
        samples = samples[0] if method == "sample_and_log_prob" else samples
        self.assertEqual(samples.dtype, jnp.int8)

    @parameterized.expand(
        [
            ("sample, from probs", "sample", True),
            ("sample, from logits", "sample", False),
            ("sample_and_log_prob, from probs", "sample_and_log_prob", True),
            ("sample_and_log_prob, from logits", "sample_and_log_prob", False),
        ]
    )
    def test_sample_values(self, name, method, from_probs):
        probs = np.array([[0.5, 0.25, 0.25], [0.0, 0.0, 1.0]])  # Includes edge case.
        num_categories = probs.shape[-1]
        logits = np.log(probs)
        n_samples = 100000
        dist_kwargs = (
            {"probs": jnp.array(probs)} if from_probs else {"logits": jnp.array(logits)}
        )
        dist = self.dist(**dist_kwargs)
        sample_fn = lambda key: jax.jit(jax.vmap(getattr(dist, method)))(
            jax.random.split(key, n_samples)
        )
        samples = sample_fn(self.key)
        samples = samples[0] if method == "sample_and_log_prob" else samples
        self.assertEqual(samples.shape, (n_samples,) + probs.shape)
        samples_arg = jnp.argmax(samples, axis=-1)
        self.assertTrue(
            np.all(np.logical_and(samples_arg >= 0, samples_arg < num_categories))
        )
        np.testing.assert_array_equal(jnp.floor(samples_arg), samples_arg)
        np.testing.assert_allclose(np.mean(samples, axis=0), probs, rtol=0.1)

    @parameterized.expand(
        [
            ("sample", "sample"),
            ("sample_and_log_prob", "sample_and_log_prob"),
        ]
    )
    def test_sample_values_invalid_probs(self, name, method):
        # Check that samples=-1 if probs are negative or NaN after normalization.
        n_samples = 1000
        probs = np.array(
            [
                [0.1, -0.4, 0.2, 0.3],  # Negative probabilities.
                [-0.1, 0.1, 0.0, 0.0],  # NaN probabilities after normalization.
                [0.1, 0.25, 0.2, 0.8],  # Valid (unnormalized) probabilities.
            ]
        )
        dist = self.dist(probs=jnp.array(probs))
        sample_fn = lambda key: jax.jit(jax.vmap(getattr(dist, method)))(
            jax.random.split(key, n_samples)
        )
        samples = sample_fn(self.key)
        samples = samples[0] if method == "sample_and_log_prob" else samples
        np.testing.assert_allclose(samples[:, :-1, :], -1, rtol=1e-4)
        np.testing.assert_array_compare(lambda x, y: x >= y, samples[:, -1, :], 0)

    @parameterized.expand(
        [
            ("1d logits, 1d value", {"logits": [0.0, 0.5, -0.5]}, [1, 0, 2, 0]),
            ("1d probs, 1d value", {"probs": [0.3, 0.2, 0.5]}, [1, 0, 2, 0]),
            ("extreme probs", {"probs": [0.0, 1.0, 0.0]}, [0, 1, 1, 2]),
        ]
    )
    def test_method_with_input(self, name, distr_params, value):
        distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
        dist = self.dist(**distr_params)
        value = jax.nn.one_hot(jnp.asarray(value), dist.num_categories, axis=-1)
        for method in [
            "prob",
            "log_prob",
            "cdf",
            "log_cdf",
            "survival_function",
            "log_survival_function",
        ]:
            with self.subTest(method=method):
                fn = getattr(dist, method)
                x = jax.vmap(fn)(value)
                if len(x.shape) > 0:
                    self.assertEqual(x.shape, (value.shape[0],))
                else:
                    self.assertEqual(x.shape, ())

    def test_method_with_input_outside_domain(self):
        probs = jnp.asarray([0.2, 0.3, 0.5])
        dist = self.dist(probs=probs)
        value = jnp.asarray([0, 0, 0])
        np.testing.assert_allclose(dist.prob(value), np.asarray(0.0), atol=1e-5)
        self.assertTrue(np.all(dist.log_prob(value) == 0.0))

    @parameterized.expand(
        [
            ("2d probs", True),
            ("2d logits", False),
        ]
    )
    def test_method(self, name, from_probs):
        rng = np.random.default_rng(42)
        probs = rng.uniform(size=(4, 3))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        logits = np.log(probs)
        distr_params = (
            {"probs": jnp.array(probs)} if from_probs else {"logits": jnp.array(logits)}
        )
        dist = self.dist(**distr_params)
        for method in ["entropy", "mode"]:
            fn = getattr(dist, method)
            x = fn()
            x_shape = (
                (dist.logits.shape[0],) if method == "entropy" else dist.logits.shape
            )
            self.assertEqual(x_shape, x.shape)

    @parameterized.expand(
        [
            ("kl distreqx_to_distreqx", "kl_divergence", "distreqx_to_distreqx"),
            ("cross-ent distreqx_to_distreqx", "cross_entropy", "distreqx_to_distreqx"),
        ]
    )
    def test_with_two_distributions(self, name, function_string, mode_string):
        dist1_kwargs = {"probs": jnp.asarray([[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]])}
        dist2_kwargs = {"logits": jnp.asarray([0.0, 0.1, 0.1])}
        dist1 = self.dist(**dist1_kwargs)
        dist2 = self.dist(**dist2_kwargs)

        with self.subTest(method=function_string):
            fn = getattr(dist1, function_string)
            x = fn(dist2)
            self.assertEqual(
                dist1.logits.shape[:-1], x.shape
            )  # TODO: test for known solutions

    @parameterized.expand(
        [
            ("kl distreqx_to_distreqx", "kl_divergence", "distreqx_to_distreqx"),
            ("cross-ent distreqx_to_distreqx", "cross_entropy", "distreqx_to_distreqx"),
        ]
    )
    def test_with_two_distributions_extreme_cases(
        self, name, function_string, mode_string
    ):
        dist1_kwargs = {
            "probs": jnp.asarray([[0.1, 0.5, 0.4], [0.4, 0.0, 0.6], [0.4, 0.6, 0.0]])
        }
        dist2_kwargs = {"logits": jnp.asarray([0.0, 0.1, -jnp.inf])}
        dist1 = self.dist(**dist1_kwargs)
        dist2 = self.dist(**dist2_kwargs)
        with self.subTest(method=function_string):
            fn = getattr(dist1, function_string)
            x = fn(dist2)
            self.assertEqual(
                dist1.logits.shape[:-1], x.shape
            )  # TODO: test for known solutions

    @parameterized.expand(
        [
            ("kl distreqx_to_distreqx", "kl_divergence"),
            ("cross-ent distreqx_to_distreqx", "cross_entropy"),
        ]
    )
    def test_with_two_distributions_raises_on_invalid_num_categories(
        self, name, function_string
    ):
        probs1 = jnp.asarray([0.1, 0.5, 0.4])
        distreqx_dist1 = self.dist(probs=probs1)
        logits2 = jnp.asarray([-0.1, 0.3])
        distreqx_dist2 = self.dist(logits=logits2)
        first_fn = getattr(distreqx_dist1, function_string)
        with self.assertRaises(ValueError):
            _ = first_fn(distreqx_dist2)
        second_fn = getattr(distreqx_dist2, function_string)
        with self.assertRaises(ValueError):
            _ = second_fn(distreqx_dist1)

    def test_jittable(self):
        @eqx.filter_jit
        def f(dist):
            return dist.sample(key=jax.random.PRNGKey(0))

        dist_params = {"logits": jnp.array([0.0, 4.0, -1.0, 4.0])}
        dist = self.dist(**dist_params)
        y = f(dist)
        self.assertIsInstance(y, jax.Array)

    # TODO: test_slice, test_slice_ellipsis

    def test_vmap_inputs(self):
        def log_prob_sum(dist, x):
            return dist.log_prob(x).sum()

        dist = self.dist(jnp.arange(3 * 4 * 5).reshape((3, 4, 5)))
        x = jnp.zeros((3, 4), jnp.int_)
        x = jax.nn.one_hot(x, dist.num_categories, axis=-1)

        with self.subTest("no vmap"):
            actual = log_prob_sum(dist, x)
            expected = dist.log_prob(x).sum()
            np.testing.assert_allclose(actual, expected)

        with self.subTest("axis=0"):
            actual = jax.vmap(log_prob_sum, in_axes=0)(dist, x)
            expected = dist.log_prob(x).sum(axis=1)
            np.testing.assert_allclose(actual, expected)

        with self.subTest("axis=1"):
            actual = jax.vmap(log_prob_sum, in_axes=1)(dist, x)
            expected = dist.log_prob(x).sum(axis=0)
            np.testing.assert_allclose(actual, expected)

    def test_vmap_outputs(self):
        def summed_dist(logits):
            logits1 = logits.sum(keepdims=True)
            logits2 = -logits1
            logits = jnp.concatenate([logits1, logits2], axis=-1)
            return self.dist(logits)

        logits = jnp.arange((3 * 4 * 5)).reshape((3, 4, 5))
        actual = jax.vmap(summed_dist)(logits)

        logits1 = logits.sum(axis=(1, 2), keepdims=True)
        logits2 = -logits1
        logits = jnp.concatenate([logits1, logits2], axis=-1)
        expected = self.dist(logits)

        np.testing.assert_equal(actual.event_shape, expected.event_shape)

        x = jnp.array([[[0]], [[1]], [[0]]], jnp.int_)
        np.testing.assert_allclose(actual.log_prob(x), expected.log_prob(x), rtol=1e-6)

    @parameterized.expand(
        [
            ("-inf logits", np.array([-jnp.inf, 2, -3, -jnp.inf, 5.0])),
            ("uniform large negative logits", np.array([-1e9] * 11)),
            ("uniform large positive logits", np.array([1e9] * 11)),
            ("uniform", np.array([0.0] * 11)),
            ("typical", np.array([1, 7, -3, 2, 4.0])),
        ]
    )
    def test_entropy_grad(self, name, logits):
        clipped_logits = jnp.maximum(-10000, logits)

        def entropy_fn(logits):
            return self.dist(logits).entropy()

        entropy, grads = jax.value_and_grad(entropy_fn)(logits)
        expected_entropy, expected_grads = jax.value_and_grad(entropy_fn)(
            clipped_logits
        )
        np.testing.assert_allclose(expected_entropy, entropy, rtol=1e-6)
        np.testing.assert_allclose(expected_grads, grads, rtol=1e-6)
        self.assertTrue(np.isfinite(entropy).all())
        self.assertTrue(np.isfinite(grads).all())

    @parameterized.expand(
        [
            (
                "-inf logits1",
                np.array([-jnp.inf, 2, -3, -jnp.inf, 5.0]),
                np.array([1, 7, -3, 2, 4.0]),
            ),
            (
                "-inf logits both",
                np.array([-jnp.inf, 2, -1000, -jnp.inf, 5.0]),
                np.array([-jnp.inf, 7, -jnp.inf, 2, 4.0]),
            ),
            ("typical", np.array([5, -2, 0, 1, 4.0]), np.array([1, 7, -3, 2, 4.0])),
        ]
    )
    def test_kl_grad(self, name, logits1, logits2):
        clipped_logits1 = jnp.maximum(-10000, logits1)
        clipped_logits2 = jnp.maximum(-10000, logits2)

        def kl_fn(logits1, logits2):
            return self.dist(logits1).kl_divergence(self.dist(logits2))

        kl, grads = jax.value_and_grad(kl_fn, argnums=(0, 1))(logits1, logits2)
        expected_kl, expected_grads = jax.value_and_grad(kl_fn, argnums=(0, 1))(
            clipped_logits1, clipped_logits2
        )
        np.testing.assert_allclose(expected_kl, kl, rtol=1e-6)
        np.testing.assert_allclose(expected_grads, grads, rtol=1e-6)
        self.assertTrue(np.isfinite(kl).all())
        self.assertTrue(np.isfinite(grads).all())
