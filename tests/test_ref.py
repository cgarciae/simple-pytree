import dataclasses

import jax
from attr import mutable

from simple_pytree import PytreeRef, ref_field
from simple_pytree.pytree import Pytree


class TestPytreeRef:
    def test_ref(self):
        p1 = PytreeRef(1)
        assert p1.ref.value == 1

        p2 = jax.tree_map(lambda x: x + 1, p1)

        assert p1.ref.value == 2
        assert p2.ref.value == 2
        assert p1.ref is p2.ref
        assert p1 is not p2


class TestRefField:
    def test_ref_field(self):
        @dataclasses.dataclass
        class Foo(Pytree, mutable=True):
            a: int = ref_field()

        foo1 = Foo()
        foo1.a = 2
        assert foo1.a == 2

        foo2 = jax.tree_map(lambda x: x + 1, foo1)

        assert foo1.a == 3
        assert foo2.a == 3
