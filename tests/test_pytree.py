import dataclasses

import jax
import pytest

from simple_pytree import MutablePytree, Pytree, node_field, static_field


class TestPytree:
    def test_pytree(self):
        class Foo(MutablePytree):
            x: int = static_field()

            def __init__(self, y) -> None:
                self.x = 2
                self.y = y

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

    def test_pytree_dataclass(self):
        @dataclasses.dataclass
        class Foo(MutablePytree):
            y: int = node_field()
            x: int = static_field(2)

        pytree: Foo = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6


class TestImmutablePytree:
    def test_immutable_pytree(self):
        class Foo(Pytree):
            x: int = static_field()

            def __init__(self, y) -> None:
                self.x = 2
                self.y = y

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        with pytest.raises(
            AttributeError, match="is immutable, trying to update field"
        ):
            pytree.x = 4

    def test_immutable_pytree_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class Foo(MutablePytree):
            y: int = node_field()
            x: int = static_field(2)

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        with pytest.raises(AttributeError, match="cannot assign to field"):
            pytree.x = 4
