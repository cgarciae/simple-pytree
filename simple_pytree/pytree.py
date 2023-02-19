import dataclasses
import inspect
import typing as tp
from abc import ABCMeta
from contextlib import contextmanager
from copy import copy

import jax

P = tp.TypeVar("P", bound="PytreeObject")


def field(
    default: tp.Any = dataclasses.MISSING,
    *,
    pytree_node: bool = True,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Any:
    if metadata is None:
        metadata = {}
    else:
        metadata = dict(metadata)

    if "pytree_node" in metadata:
        raise ValueError("node is already in metadata")

    metadata["pytree_node"] = pytree_node

    return dataclasses.field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


def static_field(
    default: tp.Any = dataclasses.MISSING,
    *,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Any:
    return field(
        default=default,
        pytree_node=False,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


def node_field(
    default: tp.Any = dataclasses.MISSING,
    *,
    default_factory: tp.Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
    metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Any:
    return field(
        default=default,
        pytree_node=True,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


class PytreeMeta(ABCMeta):
    def __call__(self, *args: tp.Any, **kwds: tp.Any):
        obj = super().__call__(*args, **kwds)
        object.__setattr__(obj, "_pytree__initialized", True)
        return obj


class Pytree(metaclass=PytreeMeta):
    _pytree__initialized: bool
    _pytree__static_fields: tp.Set[str]

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node(
            cls,
            flatten_func=tree_flatten,
            unflatten_func=lambda *_args: tree_unflatten(cls, *_args),
        )

        # init class variables
        cls._pytree__initialized = False  # initialize mutable
        cls._pytree__static_fields = set()

        # get class info
        class_vars = _get_all_class_vars(cls)

        for field, value in class_vars.items():
            if "_pytree__" in field or (
                isinstance(value, dataclasses.Field)
                and value.metadata is not None
                and not value.metadata.get(
                    "pytree_node", True
                )  # not pytree_node = static
            ):
                cls._pytree__static_fields.add(field)

    def replace(self: P, **kwargs: tp.Any) -> P:
        """
        Replace the values of the fields of the object with the values of the
        keyword arguments. If the object is a dataclass, `dataclasses.replace`
        will be used. Otherwise, a new object will be created with the same
        type as the original object.
        """
        if dataclasses.is_dataclass(self):
            return dataclasses.replace(self, **kwargs)

        fields = vars(self)
        for key in kwargs:
            if key not in fields:
                raise ValueError(f"'{key}' is not a field of {type(self).__name__}")

        pytree = copy(self)
        pytree.__dict__.update(kwargs)
        return pytree


def tree_flatten(pytree: Pytree):
    node_fields = {}
    static_fields = {}
    fields = vars(pytree)

    for field, value in fields.items():
        if field in pytree._pytree__static_fields:
            static_fields[field] = value
        else:
            node_fields[field] = value

    children = (node_fields,)

    return children, static_fields


def tree_unflatten(cls: tp.Type[P], static_fields, children):
    (node_fields,) = children
    attrs = dict(node_fields, **static_fields)

    pytree = cls.__new__(cls)
    pytree.__dict__.update(attrs)

    return pytree


def _get_all_class_vars(cls: type) -> tp.Dict[str, tp.Any]:
    d = {}
    for c in reversed(cls.mro()):
        if hasattr(c, "__dict__"):
            d.update(vars(c))
    return d


class ImmutablePytree(Pytree):
    if not tp.TYPE_CHECKING:

        def __setattr__(self: P, field: str, value: tp.Any):
            if self._pytree__initialized:
                raise AttributeError(
                    f"{type(self).__name__} is immutable, trying to update field {field}"
                )

            object.__setattr__(self, field, value)
