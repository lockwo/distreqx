from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.distributions import Deterministic, Joint, Normal


class JointTest(TestCase):
    def setUp(self):
        self.key = jax.random.key(0)

        # A heterogeneous tree of distributions
        self.tree_dists = {
            "a": Deterministic(loc=1.0),
            "b": {"c": Deterministic(loc=2.0), "d": Deterministic(loc=-1.5)},
        }
        self.joint_dist = Joint(distributions=self.tree_dists)

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "contain at least one distribution"):
            Joint(distributions={})

    @parameterized.expand([("float32", jnp.float32), ("float64", jnp.float64)])
    def test_sample_and_stats(self, name, dtype):
        # We cast the inputs to target dtype to test dtype consistency
        tree_dists = {
            "a": Deterministic(loc=jnp.array(1.0, dtype=dtype)),
            "b": Deterministic(loc=jnp.array(2.0, dtype=dtype)),
        }
        joint_dist = Joint(distributions=tree_dists)

        # 1. Sample & Log Prob
        sample, log_prob = joint_dist.sample_and_log_prob(self.key)

        # Structure matching
        self.assertEqual(set(sample.keys()), {"a", "b"})
        self.assertion_fn()(sample["a"], 1.0)
        self.assertion_fn()(sample["b"], 2.0)
        self.assertEqual(sample["a"].dtype, dtype)

        # Log Prob: log(1) + log(1) = 0.0
        self.assertEqual(log_prob, 0.0)

        # 2. PyTree Statistics
        mean = joint_dist.mean()
        self.assertion_fn()(mean["a"], 1.0)
        self.assertion_fn()(mean["b"], 2.0)

        variance = joint_dist.variance()
        self.assertion_fn()(variance["a"], 0.0)

        # Entropy is summed over leaves
        self.assertion_fn()(joint_dist.entropy(), 0.0)

    def test_log_prob_and_cdf(self):
        # Valid sample that perfectly matches
        valid_val = {
            "a": jnp.array(1.0),
            "b": {"c": jnp.array(2.0), "d": jnp.array(-1.5)},
        }
        self.assertion_fn()(self.joint_dist.log_prob(valid_val), 0.0)
        self.assertion_fn()(self.joint_dist.cdf(valid_val), 1.0)

        # Invalid sample for log_prob AND cdf.
        # By setting "c" to 0.0 (which is less than loc=2.0), the CDF for
        # "c" becomes 0.0. Thus, the joint CDF (product of marginals)
        # becomes 0.0.
        invalid_val = {
            "a": jnp.array(1.0),
            "b": {"c": jnp.array(0.0), "d": jnp.array(-1.5)},
        }
        self.assertEqual(self.joint_dist.log_prob(invalid_val), -jnp.inf)
        self.assertEqual(self.joint_dist.cdf(invalid_val), 0.0)

    def test_kl_divergence(self):
        # Identical Joint
        dist2 = Joint(
            distributions={
                "a": Deterministic(loc=1.0),
                "b": {"c": Deterministic(loc=2.0), "d": Deterministic(loc=-1.5)},
            }
        )
        self.assertEqual(self.joint_dist.kl_divergence(dist2), 0.0)

        # Shifted Joint (Infinite KL due to disjoint supports in Deterministic)
        dist3 = Joint(
            distributions={
                "a": Deterministic(loc=5.0),
                "b": {"c": Deterministic(loc=2.0), "d": Deterministic(loc=-1.5)},
            }
        )
        self.assertEqual(self.joint_dist.kl_divergence(dist3), jnp.inf)

    def test_heterogeneous_shapes(self):
        """Verifies that the Joint distribution gracefully handles mixed shapes."""
        # Mix a scalar leaf with a vector leaf
        tree_dists = {
            "scalar": Deterministic(loc=1.0),
            "vector": Normal(loc=jnp.zeros(2), scale=jnp.ones(2)),
        }
        hetero_joint = Joint(distributions=tree_dists)

        # 1. Event Shape checks
        event_shape = hetero_joint.event_shape
        self.assertEqual(event_shape["scalar"], ())
        self.assertEqual(event_shape["vector"], (2,))

        # 2. Sample checks
        sample = hetero_joint.sample(self.key)
        self.assertEqual(sample["scalar"].shape, ())
        self.assertEqual(sample["vector"].shape, (2,))

        # 3. Math checks (Ensure ragged arrays are safely reduced to scalars)
        lp = hetero_joint.log_prob(sample)
        self.assertEqual(lp.shape, ())
        self.assertFalse(jnp.isnan(lp))

        ent = hetero_joint.entropy()
        self.assertEqual(ent.shape, ())
        self.assertFalse(jnp.isnan(ent))

    def test_jittable(self):
        @eqx.filter_jit
        def f(dist):
            return dist.sample_and_log_prob(key=self.key)

        samples, log_prob = f(self.joint_dist)
        self.assertIsInstance(samples["a"], jax.Array)
        self.assertIsInstance(log_prob, jax.Array)
