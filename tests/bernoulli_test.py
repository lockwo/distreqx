from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore
from scipy import special as sp_special

from distreqx.distributions import bernoulli


class BernoulliTest(TestCase):
    def setUp(self):
        self.dist = bernoulli.Bernoulli
        self.p = np.asarray([0.2, 0.4, 0.6, 0.8])
        self.logits = sp_special.logit(self.p)
        self.key = jax.random.PRNGKey(0)

    @parameterized.expand(
        [
            ("0d probs", (), True),
            ("0d logits", (), False),
            ("1d probs", (4,), True),
            ("1d logits", (4,), False),
            ("2d probs", (3, 4), True),
            ("2d logits", (3, 4), False),
        ]
    )
    def test_properties(self, name, shape, from_probs):
        rng = np.random.default_rng(42)
        probs = rng.uniform(size=shape)
        logits = sp_special.logit(probs)
        dist_kwargs = (
            {"probs": jnp.array(probs)} if from_probs else {"logits": jnp.array(logits)}
        )
        dist = self.dist(**dist_kwargs)
        np.testing.assert_allclose(dist.logits, logits, rtol=1e-3)
        np.testing.assert_allclose(dist.probs, probs, rtol=1e-3)

    @parameterized.expand(
        [
            (
                "probs and logits",
                {"logits": jnp.array([0.1, -0.2]), "probs": jnp.array([0.5, 0.4])},
            ),
            ("both probs and logits are None", {"logits": None, "probs": None}),
        ]
    )
    def test_raises_on_invalid_inputs(self, name, dist_params):
        with self.assertRaises(ValueError):
            self.dist(**dist_params)

    @parameterized.expand(
        [
            ("1d probs, 1-tuple shape", {"probs": [0.1, 0.5, 0.3]}, (3,)),
            (
                "2d probs, 2-tuple shape",
                {"probs": [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
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
            ("sample, from probs", "sample", True),
            ("sample, from logits", "sample", False),
            ("sample_and_log_prob, from probs", "sample_and_log_prob", True),
            ("sample_and_log_prob, from logits", "sample_and_log_prob", False),
        ]
    )
    def test_sample_values(self, name, method, from_probs):
        probs = np.array([0.0, 0.2, 0.5, 0.8, 1.0])  # Includes edge cases (0 and 1).
        logits = sp_special.logit(probs)
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
        self.assertTrue(np.all(np.logical_or(samples == 0, samples == 1)))
        np.testing.assert_allclose(np.mean(samples, axis=0), probs, rtol=0.1)
        np.testing.assert_allclose(np.std(samples, axis=0), dist.stddev(), atol=2e-3)

    @parameterized.expand(
        [
            ("1d probs, 1-tuple shape", {"probs": [0.1, 0.5, 0.3]}, (3,)),
            (
                "2d probs, 2-tuple shape",
                {"probs": [[0.1, 0.4, 0.5], [0.5, 0.25, 0.25]]},
                (2, 3),
            ),
        ]
    )
    def test_sample_and_log_prob(self, name, distr_params, sample_shape):
        distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
        dist = self.dist(**distr_params)
        samples, log_prob = dist.sample_and_log_prob(self.key)
        self.assertTupleEqual(samples.shape, sample_shape)
        self.assertTupleEqual(log_prob.shape, sample_shape)

    @parameterized.expand(
        [
            ("sample, float16", "sample", jnp.float16),
            ("sample, float32", "sample", jnp.float32),
            ("sample_and_log_prob, float16", "sample_and_log_prob", jnp.float16),
            ("sample_and_log_prob, float32", "sample_and_log_prob", jnp.float32),
        ]
    )
    def test_sample_dtype(self, name, method, dtype):
        dist_params = {"logits": jnp.array(self.logits).astype(dtype)}
        dist = self.dist(**dist_params)
        samples = getattr(dist, method)(self.key)
        samples = samples[0] if method == "sample_and_log_prob" else samples
        self.assertEqual(samples.dtype, jnp.int8)

    @parameterized.expand(
        [
            ("1d logits, int value", {"logits": [0.0, 0.5, -0.5]}, 1),
            ("1d probs, int value", {"probs": [0.3, 0.2, 0.5]}, 1),
            ("1d logits, 1d value", {"logits": [0.0, 0.5, -0.5]}, [1, 0, 1]),
            ("1d probs, 1d value", {"probs": [0.3, 0.2, 0.5]}, [1, 0, 1]),
            (
                "2d logits, 1d value",
                {"logits": [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
                [1, 0, 1],
            ),
            (
                "2d probs, 1d value",
                {"probs": [[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]},
                [1, 0, 1],
            ),
            (
                "2d logits, 2d value",
                {"logits": [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]},
                [[1, 0, 0], [1, 1, 0]],
            ),
            (
                "2d probs, 2d value",
                {"probs": [[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]},
                [[1, 0, 0], [1, 1, 0]],
            ),
            (
                "edge cases with logits",
                {"logits": [-np.inf, -np.inf, np.inf, np.inf]},
                [0, 1, 0, 1],
            ),
            ("edge cases with probs", {"probs": [0.0, 0.0, 1.0, 1.0]}, [0, 1, 0, 1]),
        ]
    )
    def test_method_with_value(self, name, distr_params, value):
        distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
        value = jnp.asarray(value)
        for method in [
            "prob",
            "log_prob",
            "cdf",
            "log_cdf",
            "survival_function",
            "log_survival_function",
        ]:
            with self.subTest(method=method):
                dist1 = self.dist(**distr_params)
                fn = getattr(dist1, method)
                x = fn(value)
                self.assertEqual(dist1.logits.shape, x.shape)

    @parameterized.expand(
        [
            ("from logits", {"logits": [[0.0, 0.5, -0.5], [-0.2, 0.3, 0.5]]}),
            ("from probs", {"probs": [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]]}),
        ]
    )
    def test_method(self, name, distr_params):
        distr_params = {k: jnp.asarray(v) for k, v in distr_params.items()}
        for method in ["entropy", "mode", "mean", "variance", "stddev"]:
            with self.subTest(method=method):
                dist1 = self.dist(**distr_params)
                fn = getattr(dist1, method)
                x = fn()
                self.assertEqual(dist1.logits.shape, x.shape)

    @parameterized.expand(
        [
            ("kl", "kl_divergence", "distrax_to_distrax"),
            ("cross-ent", "cross_entropy", "distrax_to_distrax"),
        ]
    )
    def test_with_two_distributions(self, name, function_string, mode_string):
        dist1_kwargs = {"probs": jnp.asarray([[0.1, 0.5, 0.4], [0.2, 0.4, 0.8]])}
        dist2_kwargs = {
            "logits": jnp.asarray([0.0, -0.1, 0.1]),
        }
        with self.subTest(method=function_string):
            dist1 = self.dist(**dist1_kwargs)
            dist2 = self.dist(**dist2_kwargs)
            fn = getattr(dist1, function_string)
            x = fn(dist2)
            self.assertEqual(dist1.logits.shape, x.shape)
