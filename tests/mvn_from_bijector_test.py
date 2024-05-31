"""Tests for `mvn_from_bijector.py`."""

from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import AbstractLinearBijector, DiagLinear
from distreqx.distributions import MultivariateNormalFromBijector


class MockLinear(AbstractLinearBijector):
    """A mock linear bijector."""

    def __init__(self, event_dims: int):
        super().__init__(event_dims)

    def forward_and_log_det(self, x):
        """Computes y = f(x) and log|det J(f)(x)|."""
        return x, jnp.zeros_like(x)[:-1]


class MultivariateNormalFromBijectorTest(TestCase):
    @parameterized.expand(
        [
            ("loc is 0d", 4, jnp.zeros(shape=())),
            ("loc and scale dims not compatible", 3, jnp.zeros((4,))),
        ]
    )
    def test_raises_on_wrong_inputs(self, name, event_dims, loc):
        bij = MockLinear(event_dims)
        with self.assertRaises(ValueError):
            MultivariateNormalFromBijector(loc, bij)

    @parameterized.expand([("no broadcast", jnp.ones((4,)), jnp.zeros((4,)), (4,))])
    def test_loc_scale_and_shapes(self, name, diag, loc, expected_shape):
        scale = DiagLinear(diag)
        dist = MultivariateNormalFromBijector(loc, scale)
        np.testing.assert_allclose(dist.loc, np.zeros(expected_shape))
        self.assertTrue(scale.same_as(dist.scale))
        self.assertEqual(dist.event_shape, (4,))

    def test_sample(self):
        prng = jax.random.PRNGKey(42)
        keys = jax.random.split(prng, 2)
        diag = 0.5 + jax.random.uniform(keys[0], (4,))
        loc = jax.random.normal(keys[1], (4,))
        scale = DiagLinear(diag)
        dist = MultivariateNormalFromBijector(loc, scale)
        num_samples = 100_000
        sample_fn = lambda seed: dist.sample(key=seed)
        samples = eqx.filter_vmap(sample_fn)(jax.random.split(prng, num_samples))
        self.assertEqual(samples.shape, (num_samples, 4))
        np.testing.assert_allclose(jnp.mean(samples, axis=0), loc, rtol=0.1)
        np.testing.assert_allclose(jnp.std(samples, axis=0), diag, rtol=0.1)

    @parameterized.expand(
        [
            ("no broadcast", (4,), (4,)),
        ]
    )
    def test_mean_median_mode(self, name, diag_shape, loc_shape):
        prng = jax.random.PRNGKey(42)
        diag = jax.random.normal(prng, diag_shape)
        loc = jax.random.normal(prng, loc_shape)
        scale = DiagLinear(diag)
        batch_shape = jnp.broadcast_shapes(diag_shape, loc_shape)[:-1]
        dist = MultivariateNormalFromBijector(loc, scale)
        for method in ["mean", "median", "mode"]:
            with self.subTest(method=method):
                fn = getattr(dist, method)
                np.testing.assert_allclose(
                    fn(), jnp.broadcast_to(loc, batch_shape + loc.shape[-1:])
                )

    @parameterized.expand(
        [
            ("kl distreqx_to_distreqx", "kl_divergence"),
            ("cross-ent distreqx_to_distreqx", "cross_entropy"),
        ]
    )
    def test_with_two_distributions(self, name, function_string):
        rng = np.random.default_rng(42)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        dist1_kwargs = {
            "loc": jnp.array(rng.normal(size=(5,)).astype(np.float32)),
            "scale": DiagLinear(
                0.1 + jnp.array(rng1.uniform(size=(5,)).astype(np.float32))
            ),
        }
        dist2_kwargs = {
            "loc": jnp.asarray([-2.4, -1.0, 0.0, 1.2, 6.5]).astype(np.float32),
            "scale": DiagLinear(
                0.1 + jnp.array(rng2.uniform(size=(5,)).astype(np.float32))
            ),
        }

        dist1 = MultivariateNormalFromBijector(**dist1_kwargs)
        dist2 = MultivariateNormalFromBijector(**dist2_kwargs)

        if function_string == "kl_divergence":
            result1 = dist1.kl_divergence(dist2)
            result2 = dist2.kl_divergence(dist1)
        elif function_string == "cross_entropy":
            result1 = dist1.cross_entropy(dist2)
            result2 = dist2.cross_entropy(dist1)
        else:
            raise ValueError(f"Unsupported function string: {function_string}")
        np.testing.assert_allclose(result1, result2, rtol=1e-3)

    def test_kl_divergence_raises_on_incompatible_distributions(self):
        dim = 4
        dist1 = MultivariateNormalFromBijector(
            loc=jnp.zeros((dim,)),
            scale=DiagLinear(diag=jnp.ones((dim,))),
        )
        dim = 5
        dist2 = MultivariateNormalFromBijector(
            loc=jnp.zeros((dim,)),
            scale=DiagLinear(diag=jnp.ones((dim,))),
        )
        with self.assertRaises(TypeError):
            dist1.kl_divergence(dist2)
