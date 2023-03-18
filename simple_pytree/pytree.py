import dataclasses
import importlib.util
import itertools
import typing as tp
from abc import ABCMeta
from copy import copy
from functools import partial

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
        obj: P = self.__new__(self)
        obj.__dict__["_pytree__initializing"] = True
        try:
            obj.__init__(*args, **kwds)
        finally:
            del obj.__dict__["_pytree__initializing"]
        return obj


class Pytree(metaclass=PytreeMeta):
    _pytree__initializing: bool
    _pytree__class_is_mutable: bool
    _pytree__static_fields: tp.FrozenSet[str]
    _pytree__setter_descriptors: tp.FrozenSet[str]

    def __init_subclass__(cls, mutable: bool = False):
        super().__init_subclass__()

        # gather class info
        class_vars = vars(cls)
        setter_descriptors = set()
        static_fields = _inherited_static_fields(cls)

        for field, value in class_vars.items():
            if isinstance(value, dataclasses.Field) and not value.metadata.get(
                "pytree_node", True
            ):
                static_fields.add(field)

            # add setter descriptors
            if hasattr(value, "__set__"):
                setter_descriptors.add(field)

        # init class variables
        cls._pytree__initializing = False
        cls._pytree__class_is_mutable = mutable
        cls._pytree__static_fields = frozenset(static_fields)
        cls._pytree__setter_descriptors = frozenset(setter_descriptors)

        if hasattr(jax.tree_util, "register_pytree_with_keys"):
            jax.tree_util.register_pytree_with_keys(
                cls,
                partial(
                    cls._pytree__flatten,
                    cls._pytree__static_fields,
                    with_key_paths=True,
                ),
                cls._pytree__unflatten,
            )
        else:
            jax.tree_util.register_pytree_node(
                cls,
                partial(
                    cls._pytree__flatten,
                    cls._pytree__static_fields,
                    with_key_paths=False,
                ),
                cls._pytree__unflatten,
            )

        # flax serialization support
        if importlib.util.find_spec("flax") is not None:
            from flax import serialization

            serialization.register_serialization_state(
                cls,
                partial(cls._to_flax_state_dict, cls._pytree__static_fields),
                partial(cls._from_flax_state_dict, cls._pytree__static_fields),
            )

    @classmethod
    def _pytree__flatten(
        cls,
        static_field_names: tp.FrozenSet[str],
        pytree: "Pytree",
        *,
        with_key_paths: bool,
    ) -> tp.Tuple[
        tp.List[tp.Any],
        tp.Tuple[tp.List[str], tp.List[tp.Tuple[str, tp.Any]]],
    ]:
        static_fields = []
        node_names = []
        node_values = []
        # sort to ensure deterministic order
        for field in sorted(vars(pytree)):
            value = getattr(pytree, field)
            if field in static_field_names:
                static_fields.append((field, value))
            else:
                if with_key_paths:
                    value = (jax.tree_util.GetAttrKey(field), value)
                node_names.append(field)
                node_values.append(value)

        return node_values, (node_names, static_fields)

    @classmethod
    def _pytree__unflatten(
        cls: tp.Type[P],
        metadata: tp.Tuple[tp.List[str], tp.List[tp.Tuple[str, tp.Any]]],
        node_values: tp.List[tp.Any],
    ) -> P:
        node_names, static_fields = metadata
        node_fields = dict(zip(node_names, node_values))
        pytree = cls.__new__(cls)
        pytree.__dict__.update(node_fields, **dict(static_fields))
        return pytree

    @classmethod
    def _to_flax_state_dict(
        cls, static_field_names: tp.FrozenSet[str], pytree: "Pytree"
    ) -> tp.Dict[str, tp.Any]:
        from flax import serialization

        state_dict = {
            name: serialization.to_state_dict(getattr(pytree, name))
            for name in pytree.__dict__
            if name not in static_field_names
        }
        return state_dict

    @classmethod
    def _from_flax_state_dict(
        cls,
        static_field_names: tp.FrozenSet[str],
        pytree: P,
        state: tp.Dict[str, tp.Any],
    ) -> P:
        """Restore the state of a data class."""
        from flax import serialization

        state = state.copy()  # copy the state so we can pop the restored fields.
        updates = {}
        for name in pytree.__dict__:
            if name in static_field_names:
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

    def replace(self: P, **kwargs: tp.Any) -> P:
        """
        Replace the values of the fields of the object with the values of the
        keyword arguments. If the object is a dataclass, `dataclasses.replace`
        will be used. Otherwise, a new object will be created with the same
        type as the original object.
        """
        if dataclasses.is_dataclass(self):
            return dataclasses.replace(self, **kwargs)

        unknown_keys = set(kwargs) - set(vars(self))
        if unknown_keys:
            raise ValueError(
                f"Trying to replace unknown fields {unknown_keys} "
                f"for '{type(self).__name__}'"
            )

        pytree = copy(self)
        pytree.__dict__.update(kwargs)
        return pytree

    if not tp.TYPE_CHECKING:

        def __setattr__(self: P, field: str, value: tp.Any):
            if (
                not self._pytree__initializing
                and not self._pytree__class_is_mutable
                and field not in self._pytree__setter_descriptors
            ):
                raise AttributeError(
                    f"{type(self)} is immutable, trying to update field {field}"
                )

            object.__setattr__(self, field, value)


def _inherited_static_fields(cls: type) -> tp.Set[str]:
    static_fields = set()
    for parent_class in cls.mro():
        if (
            parent_class is not cls
            and parent_class is not Pytree
            and issubclass(parent_class, Pytree)
        ):
            static_fields.update(parent_class._pytree__static_fields)
    return static_fields
