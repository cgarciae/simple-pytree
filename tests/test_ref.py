import dataclasses

import jax
import pytest

from simple_pytree import (
    Pytree,
    PytreeRef,
    clone_references,
    cross_barrier,
    incremented_ref,
    ref_field,
)


class TestPytreeRef:
    def test_ref(self):
        p1 = PytreeRef(1)
        assert p1.value == 1

        p2 = jax.tree_map(lambda x: x + 1, p1)

        assert p1.value == 2
        assert p2.value == 2
        assert p1 is not p2
        assert p1.ref is p2.ref

        p1.value = 3

        assert p1.value == 3
        assert p2.value == 3

    def test_ref_context(self):
        p1 = PytreeRef(1)
        p2 = jax.tree_map(lambda x: x, p1)  # copy
        assert p1.value == 1
        assert p2.value == 1
        p1.value = 2  # OK
        assert p2.value == 2

        with incremented_ref():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different context"
            ):
                p1.value = 3

            p1, p2 = clone_references((p1, p2))
            assert p1.value == 2
            p2.value = 3  # OK
            assert p1.value == 3

        with pytest.raises(
            ValueError, match="Cannot mutate ref from different context"
        ):
            p1.value = 4

        p1, p2 = clone_references((p1, p2))
        assert p1.value == 3
        p1.value = 4  # OK
        assert p2.value == 4

    def test_ref_trace_level(self):
        p1: PytreeRef[int] = PytreeRef(1)

        @jax.jit
        def f():
            with pytest.raises(
                ValueError, match="Cannot mutate ref from different trace level"
            ):
                p1.value = 2
            return 1

        f()

        @cross_barrier(jax.jit)
        def g(p2: PytreeRef[int]):
            p2.value = 2
            assert p1.ref is not p2.ref
            return p2

        p2 = g(p1)
        p2_ref = p2.ref

        assert p1.value == 1
        assert p2.value == 2

        p2.value = 3
        assert p1.value == 1
        assert p2.value == 3

        p3 = g(p1)
        p3_ref = p3.ref

        assert p3_ref is not p2_ref
        assert p3.value == 2

    def test_barrier(self):
        p1: PytreeRef[int] = PytreeRef(1)

        @cross_barrier(jax.jit)
        def g(p2: PytreeRef[int]):
            p2.value = 2
            assert p1.ref is not p2.ref
            return p2

        p2 = g(p1)
        assert p1.ref is not p2.ref
        assert p1.value == 1
        assert p2.value == 2

        # test passing a reference to a jitted function without cross_barrier
        @jax.jit
        def f(p1):
            return None

        with pytest.raises(
            ValueError, match="Cannot mutate ref from different trace level"
        ):
            f(p1)

        assert isinstance(p1.value, int)
        assert p1.value == 1


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
