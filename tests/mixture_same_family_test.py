import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
from parameterized import parameterized  # type: ignore

from distreqx.distributions import (
    Categorical,
    MixtureSameFamily,
    MultivariateNormalDiag,
)


class MixtureSameFamilyTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(0)
        self.num_components = 3
        self.logits_shape = (self.num_components,)
        self.logits = jax.random.normal(key=self.key, shape=self.logits_shape)
        self.probs = jax.nn.softmax(self.logits, axis=-1)

        key_loc, key_scale = jax.random.split(self.key)
        self.components_shape = (5,)
        self.loc = jax.random.normal(
            key=key_loc, shape=self.logits_shape + self.components_shape
        )
        self.scale_diag = (
            jax.random.uniform(
                key=key_scale, shape=self.logits_shape + self.components_shape
            )
            + 0.5
        )

    def test_event_shape(self):
        mixture_dist = Categorical(logits=self.logits)
        components_dist = eqx.filter_vmap(MultivariateNormalDiag)(
            self.loc, self.scale_diag
        )
        dist = MixtureSameFamily(
            mixture_distribution=mixture_dist, components_distribution=components_dist
        )
        self.assertEqual(dist.event_shape, self.logits_shape + self.components_shape)

    def test_sample_shape(self):
        mixture_dist = Categorical(logits=self.logits)
        components_dist = eqx.filter_vmap(MultivariateNormalDiag)(
            self.loc, self.scale_diag
        )
        dist = MixtureSameFamily(
            mixture_distribution=mixture_dist, components_distribution=components_dist
        )
        samples = dist.sample(self.key)
        self.assertEqual(samples.shape, self.components_shape)

    @parameterized.expand(
        [
            ("mean", "mean"),
            ("variance", "variance"),
            ("stddev", "stddev"),
        ]
    )
    def test_method(self, name, method_name):
        mixture_dist = Categorical(logits=self.logits)
        components_dist = eqx.filter_vmap(MultivariateNormalDiag)(
            self.loc, self.scale_diag
        )
        dist = MixtureSameFamily(
            mixture_distribution=mixture_dist, components_distribution=components_dist
        )
        method = getattr(dist, method_name)
        result = method()
        self.assertIsInstance(result, jnp.ndarray)

    def test_jittable(self):
        mixture_dist = Categorical(logits=self.logits)
        components_dist = eqx.filter_vmap(MultivariateNormalDiag)(
            self.loc, self.scale_diag
        )
        dist = MixtureSameFamily(
            mixture_distribution=mixture_dist, components_distribution=components_dist
        )
        sample = eqx.filter_jit(dist.sample)(self.key)
        self.assertIsInstance(sample, jnp.ndarray)
