"""Tests for `linear.py`."""

from unittest import TestCase

from parameterized import parameterized  # type: ignore

from distreqx.bijectors import (
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
    AbstractLinearBijector,
)


class MockLinear(
    AbstractLinearBijector,
    AbstractFowardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
    strict=True,
):
    _is_constant_jacobian: bool
    _is_constant_log_det: bool
    _event_dims: int

    def __init__(self, dims):
        self._event_dims = dims
        self._is_constant_jacobian = True
        self._is_constant_log_det = True

    def forward_and_log_det(self, x):
        raise Exception

    def same_as(self, other):
        raise NotImplementedError

    def inverse_and_log_det(self, y):
        raise NotImplementedError(
            f"Bijector {self.name} does not implement `inverse_and_log_det`."
        )


class LinearTest(TestCase):
    @parameterized.expand(
        [
            ("1", 1),
            ("10", 10),
        ]
    )
    def test_properties(self, name, event_dims):
        bij = MockLinear(event_dims)
        self.assertTrue(bij.is_constant_jacobian)
        self.assertTrue(bij.is_constant_log_det)
        self.assertEqual(bij.event_dims, event_dims)
        with self.assertRaises(NotImplementedError):
            bij.matrix
