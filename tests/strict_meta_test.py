"""Tests that strict abstract/final enforcement is active."""

from abc import abstractmethod
from unittest import TestCase

from ihoop.eqx import AbstractStrictModule


class StrictMetaTest(TestCase):

    def test_concrete_class_is_final(self):
        """Concrete strict classes cannot be subclassed."""

        class AbstractFoo(AbstractStrictModule):
            @abstractmethod
            def bar(self):
                raise NotImplementedError

        class Foo(AbstractFoo):
            def bar(self):
                return 42

        with self.assertRaises(TypeError, msg="Concrete classes must be final"):

            class SubFoo(Foo):
                pass

    def test_abstract_class_cannot_be_instantiated(self):
        """Abstract strict classes cannot be instantiated."""

        class AbstractFoo(AbstractStrictModule):
            @abstractmethod
            def bar(self):
                raise NotImplementedError

        with self.assertRaises(TypeError):
            AbstractFoo()  # type: ignore[reportAbstractUsage]
