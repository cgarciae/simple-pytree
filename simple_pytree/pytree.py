import dataclasses
import importlib.util
import typing as tp
from abc import ABCMeta
from copy import copy

import jax

P = tp.TypeVar("P", bound="Pytree")


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
):
    if metadata is None:
        metadata = {}
    else:
        metadata = dict(metadata)

    if "pytree_node" in metadata:
        raise ValueError("'pytree_node' found in metadata")

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
):
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


class PytreeMeta(ABCMeta):
    def __call__(self: tp.Type[P], *args: tp.Any, **kwds: tp.Any) -> P:
        obj: Pytree = super().__call__(*args, **kwds)
        object.__setattr__(obj, "_pytree__initialized", True)
        return obj


class Pytree(metaclass=PytreeMeta):
    _pytree__initialized: bool
    _pytree__class_is_mutable: bool

    def __init_subclass__(cls, mutable: bool = False):
        # init class variables
        cls._pytree__initialized = False  # initialize mutable
        cls._pytree__class_is_mutable = mutable

        # get class info
        class_vars = _get_all_class_vars(cls)
        static_fields: tp.List[str] = []

        for field, value in class_vars.items():
            if "_pytree__" in field or (
                isinstance(value, dataclasses.Field)
                and not value.metadata.get("pytree_node", True)
            ):
                static_fields.append(field)

        jax.tree_util.register_pytree_node(
            cls,
            flatten_func=lambda pytree: tree_flatten(
                pytree, static_fields, with_key_paths=False
            ),
            unflatten_func=lambda *_args: tree_unflatten(cls, *_args),
        )

        # flax serialization support
        if importlib.util.find_spec("flax") is not None:
            from flax import serialization

            serialization.register_serialization_state(
                cls,
                lambda pytree: to_state_dict(pytree, static_fields),
                lambda pytree, state: from_state_dict(pytree, state, static_fields),
            )

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

    if not tp.TYPE_CHECKING:

        def __setattr__(self: P, field: str, value: tp.Any):
            if self._pytree__initialized and not self._pytree__class_is_mutable:
                raise AttributeError(
                    f"{type(self)} is immutable, trying to update field {field}"
                )

            object.__setattr__(self, field, value)


def tree_flatten(
    pytree: Pytree, static_field_names: tp.List[str], with_key_paths: bool
) -> tp.Tuple[tp.List[tp.Any], tp.Tuple[tp.Tuple[str, ...], tp.Dict[str, tp.Any]]]:
    static_fields = {}

    node_names = []
    node_values = []
    for field, value in vars(pytree).items():
        if field in static_field_names:
            static_fields[field] = value
        else:
            if with_key_paths:
                value = (jax.tree_util.GetAttrKey(field), value)
            node_names.append(field)
            node_values.append(value)

    node_names = tuple(node_names)
    return node_values, (node_names, static_fields)


def tree_unflatten(
    cls: tp.Type[P],
    metadata: tp.Tuple[tp.Tuple[str, ...], tp.Dict[str, tp.Any]],
    node_values: tp.List[tp.Any],
) -> P:
    node_names, static_fields = metadata
    node_fields = dict(zip(node_names, node_values))
    pytree = cls.__new__(cls)
    pytree.__dict__.update(node_fields, **static_fields)
    return pytree


def _get_all_class_vars(cls: type) -> tp.Dict[str, tp.Any]:
    d = {}
    for c in reversed(cls.mro()):
        if hasattr(c, "__dict__"):
            d.update(vars(c))
    return d


def to_state_dict(pytree: Pytree, static_fields: tp.List[str]) -> tp.Dict[str, tp.Any]:
    from flax import serialization

    state_dict = {
        name: serialization.to_state_dict(getattr(pytree, name))
        for name in pytree.__dict__
        if name not in static_fields
    }
    return state_dict


def from_state_dict(
    pytree: P, state: tp.Dict[str, tp.Any], static_fields: tp.List[str]
) -> P:
    """Restore the state of a data class."""
    from flax import serialization

    state = state.copy()  # copy the state so we can pop the restored fields.
    updates = {}
    for name in pytree.__dict__:
        if name in static_fields:
            continue
        if name not in state:
            raise ValueError(
                f"Missing field {name} in state dict while restoring"
                f" an instance of {type(pytree).__name__},"
                f" at path {serialization.current_path()}"
            )
        value = getattr(pytree, name)
        value_state = state.pop(name)
        updates[name] = serialization.from_state_dict(value, value_state, name=name)
    if state:
        names = ",".join(state.keys())
        raise ValueError(
            f'Unknown field(s) "{names}" in state dict while'
            f" restoring an instance of {type(pytree).__name__}"
            f" at path {serialization.current_path()}"
        )
    return pytree.replace(**updates)
