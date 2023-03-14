import dataclasses
import typing as tp

import jax

from simple_pytree.pytree import field

A = tp.TypeVar("A")


@dataclasses.dataclass
class Ref(tp.Generic[A]):
    value: A


class PytreeRef(tp.Generic[A]):
    ref: Ref[A]

    def __init__(self, ref: tp.Union[Ref[A], A]):
        if isinstance(ref, Ref):
            self.ref = ref
        else:
            self.ref = Ref(ref)


def pytree_flatten(pytree: PytreeRef[A]) -> tp.Tuple[tp.Tuple[A], Ref[A]]:
    return (pytree.ref.value,), pytree.ref


def pytree_unflatten(ref: Ref[A], children: tp.Tuple[A]) -> PytreeRef[A]:
    ref.value = children[0]
    return PytreeRef(ref=ref)


jax.tree_util.register_pytree_node(PytreeRef, pytree_flatten, pytree_unflatten)


@dataclasses.dataclass
class RefField(tp.Generic[A]):
    default: tp.Any = dataclasses.MISSING
    name: str = ""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, f"_ref_{self.name}"):
            if self.default is not dataclasses.MISSING:
                obj.__dict__[f"_ref_{self.name}"] = PytreeRef(self.default)
            else:
                raise AttributeError(f"Attribute {self.name} is not set")
        return getattr(obj, f"_ref_{self.name}").ref.value

    def __set__(self, obj, value: tp.Union[A, Ref[A], PytreeRef[A], "RefField[A]"]):
        if isinstance(value, RefField):
            return

        if hasattr(obj, f"_ref_{self.name}"):
            if isinstance(value, (Ref, PytreeRef, RefField)):
                raise AttributeError(f"Cannot change reference of {self.name}")
            getattr(obj, f"_ref_{self.name}").ref.value = value
        elif isinstance(value, PytreeRef):
            setattr(obj, f"_ref_{self.name}", value)
        else:
            setattr(obj, f"_ref_{self.name}", PytreeRef(value))


def ref_field(
    default: tp.Any = dataclasses.MISSING,
    *,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
):
    return field(
        default=RefField(default=default),
        pytree_node=True,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )
