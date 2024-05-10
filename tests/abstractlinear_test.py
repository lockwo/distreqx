"""Tests for `linear.py`."""

from unittest import TestCase
from distreqx.bijectors import AbstractLinearBijector
from parameterized import parameterized  # type: ignore


class MockLinear(AbstractLinearBijector):

    def forward_and_log_det(self, x):
        raise Exception


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
