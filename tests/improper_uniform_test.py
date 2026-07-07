from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp

from distreqx.distributions import ImproperUniform


class ImproperUniformTest(TestCase):
    def setUp(self):
        self.key = jax.random.key(0)

    def test_defaults_to_scalar_event_shape(self):
        dist = ImproperUniform()
        self.assertEqual(dist.event_shape, ())

    def test_event_shape_and_support(self):
        dist = ImproperUniform(shape=(2, 3))
        self.assertEqual(dist.event_shape, (2, 3))

        lower, upper = dist.support
        self.assertEqual(lower.shape, (2, 3))
        self.assertTrue(jnp.all(lower == -jnp.inf))
        self.assertTrue(jnp.all(upper == jnp.inf))

    def test_log_prob_and_prob_are_unnormalized_constants(self):
        dist = ImproperUniform()
        value = jnp.array([-1e6, 0.0, 1e6])

        self.assertTrue(jnp.all(dist.log_prob(value) == 0.0))
        self.assertTrue(jnp.all(dist.prob(value) == 1.0))

    def test_entropy_is_infinite(self):
        dist = ImproperUniform(shape=(3,))
        self.assertTrue(jnp.all(dist.entropy() == jnp.inf))

    def test_undefined_methods_raise(self):
        dist = ImproperUniform()
        value = jnp.array(0.0)

        with self.assertRaisesRegex(NotImplementedError, "Cannot sample"):
            dist.sample(self.key)
        with self.assertRaisesRegex(NotImplementedError, "Cannot sample"):
            dist.sample_and_log_prob(self.key)
        with self.assertRaisesRegex(NotImplementedError, "undefined"):
            dist.icdf(value)
        with self.assertRaisesRegex(NotImplementedError, "undefined"):
            dist.log_cdf(value)
        with self.assertRaisesRegex(NotImplementedError, "undefined"):
            dist.cdf(value)
        with self.assertRaisesRegex(NotImplementedError, "undefined"):
            dist.survival_function(value)
        with self.assertRaisesRegex(NotImplementedError, "undefined"):
            dist.log_survival_function(value)
        with self.assertRaisesRegex(NotImplementedError, "undefined"):
            dist.mean()
        with self.assertRaisesRegex(NotImplementedError, "undefined"):
            dist.median()
        with self.assertRaisesRegex(NotImplementedError, "undefined"):
            dist.variance()
        with self.assertRaisesRegex(NotImplementedError, "undefined"):
            dist.stddev()
        with self.assertRaisesRegex(NotImplementedError, "undefined"):
            dist.mode()
        with self.assertRaisesRegex(NotImplementedError, "undefined"):
            dist.kl_divergence(dist)

    def test_jittable(self):
        @eqx.filter_jit
        def f(dist, value):
            return dist.log_prob(value), dist.prob(value)

        dist = ImproperUniform()
        log_prob, prob = f(dist, jnp.array(1.0))
        self.assertIsInstance(log_prob, jax.Array)
        self.assertIsInstance(prob, jax.Array)
