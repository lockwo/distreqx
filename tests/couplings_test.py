from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import Lambda, MaskedCoupling, SplitCoupling


def make_inner_bij(params):
    """An elementwise scaling bijector: f(x) = params * x + 3.0"""
    return Lambda(
        forward=lambda x: x * params + 3.0,
        inverse=lambda y: (y - 3.0) / params,
        forward_log_det_jacobian=lambda x: jnp.full_like(x, jnp.log(params)),
        inverse_log_det_jacobian=lambda y: jnp.full_like(y, -jnp.log(params)),
    )


class SplitCouplingTest(TestCase):
    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    @parameterized.expand([("no_swap", False), ("swap", True)])
    def test_forward_and_inverse(self, name, swap):
        # Conditioner simply squares the conditioning half
        conditioner = lambda x: x**2

        bij = SplitCoupling(
            split_index=2, conditioner=conditioner, bijector=make_inner_bij, swap=swap
        )

        # Let x = [1.0, 2.0, 3.0, 4.0]
        x = jnp.array([1.0, 2.0, 3.0, 4.0])

        y, log_det_fwd = bij.forward_and_log_det(x)
        x_rec, log_det_inv = bij.inverse_and_log_det(y)

        # 1. Ensure inverse recovers original input perfectly
        self.assertion_fn()(x_rec, x)
        self.assertion_fn()(log_det_fwd, -log_det_inv)

        # 2. Check mathematical correctness
        if not swap:
            # x1 = [1, 2], x2 = [3, 4]
            # params = x1^2 = [1, 4]
            # y2 = x2 * params + 3 = [3*1 + 3, 4*4 + 3] = [6, 19]
            self.assertion_fn()(y, jnp.array([1.0, 2.0, 6.0, 19.0]))

            # logdet is 0 for x1, and log(params) for x2
            self.assertion_fn()(
                log_det_fwd, jnp.array([0.0, 0.0, jnp.log(1.0), jnp.log(4.0)])
            )
        else:
            # x1 = [1, 2], x2 = [3, 4]. Swapped: condition on x2.
            # params = x2^2 = [9, 16]
            # y1 = x1 * params + 3 = [1*9 + 3, 2*16 + 3] = [12, 35]
            self.assertion_fn()(y, jnp.array([12.0, 35.0, 3.0, 4.0]))
            self.assertion_fn()(
                log_det_fwd, jnp.array([jnp.log(9.0), jnp.log(16.0), 0.0, 0.0])
            )

    def test_jittable(self):
        @eqx.filter_jit
        def f(bij, x):
            return bij.forward_and_log_det(x)

        bij = SplitCoupling(
            split_index=1, conditioner=lambda x: x, bijector=make_inner_bij
        )
        x = jnp.array([1.0, 2.0, 3.0])
        y, logdet = f(bij, x)
        self.assertIsInstance(y, jax.Array)
        self.assertIsInstance(logdet, jax.Array)


class MaskedCouplingTest(TestCase):
    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_forward_and_inverse(self):
        # A checkerboard mask
        mask = jnp.array([True, False, True, False])
        conditioner = lambda x: x + 1.0  # simple shift

        bij = MaskedCoupling(
            mask=mask, conditioner=conditioner, bijector=make_inner_bij
        )

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y, log_det_fwd = bij.forward_and_log_det(x)
        x_rec, log_det_inv = bij.inverse_and_log_det(y)

        # 1. Recover identity
        self.assertion_fn()(x_rec, x)
        self.assertion_fn()(log_det_fwd, -log_det_inv)

        # 2. Check math
        # Masked x (conditioner input) = [1.0, 0.0, 3.0, 0.0]
        # params = masked_x + 1.0 = [2.0, 1.0, 4.0, 1.0]
        # Unmasked indices (1 and 3) transformed:
        # y[1] = x[1] * params[1] + 3.0 = 2.0 * 1.0 + 3.0 = 5.0
        # y[3] = x[3] * params[3] + 3.0 = 4.0 * 1.0 + 3.0 = 7.0
        self.assertion_fn()(y, jnp.array([1.0, 5.0, 3.0, 7.0]))

        # LogDet is 0.0 for True mask, log(params) for False mask
        expected_logdet = jnp.array([0.0, jnp.log(1.0), 0.0, jnp.log(1.0)])
        self.assertion_fn()(log_det_fwd, expected_logdet)

    def test_jittable(self):
        @eqx.filter_jit
        def f(bij, x):
            return bij.forward_and_log_det(x)

        bij = MaskedCoupling(
            mask=jnp.array([True, False]),
            conditioner=lambda x: x,
            bijector=make_inner_bij,
        )
        x = jnp.array([1.0, 2.0])
        y, logdet = f(bij, x)
        self.assertIsInstance(y, jax.Array)
        self.assertIsInstance(logdet, jax.Array)
