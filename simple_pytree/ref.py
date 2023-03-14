import contextlib
import dataclasses
import functools
import threading
import typing as tp

import jax

from simple_pytree import ids, tracers
from simple_pytree.pytree import field

A = tp.TypeVar("A")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])


@dataclasses.dataclass(frozen=True)
class _RefContext:
    level: int


@dataclasses.dataclass
class _Context(threading.local):
    ref_context_stack: tp.List[_RefContext] = dataclasses.field(
        default_factory=lambda: [_RefContext(0)]
    )
    is_crossing_barrier: bool = False
    # NOTE: `barrier_cache` is not used for now but left as an optimization
    # opportunity for the future. `unflatten_pytree_ref` already has the logic
    # to use the cache, using `barrier_cache` would activate it.
    barrier_cache: tp.Optional[tp.Dict[ids.Id, "Ref[tp.Any]"]] = None

    @property
    def current_ref_context(self) -> _RefContext:
        return self.ref_context_stack[-1]


_CONTEXT = _Context()


@contextlib.contextmanager
def incremented_ref():
    _CONTEXT.ref_context_stack.append(
        _RefContext(_CONTEXT.current_ref_context.level + 1)
    )
    try:
        yield
    finally:
        _CONTEXT.ref_context_stack.pop()


def clone_references(pytree: tp.Any) -> tp.Any:
    cache: tp.Dict[ids.Id, Ref[tp.Any]] = {}

    def clone_ref(pytree: tp.Any):
        if isinstance(pytree, PytreeRef):
            if pytree.id not in cache:
                cache[pytree.id] = Ref(pytree.value, id=pytree.id)
            return PytreeRef(cache[pytree.id])
        return pytree

    return jax.tree_map(clone_ref, pytree, is_leaf=lambda x: isinstance(x, PytreeRef))


@contextlib.contextmanager
def barrier_cache():
    _CONTEXT.barrier_cache = {}
    try:
        yield
    finally:
        _CONTEXT.barrier_cache = None


@contextlib.contextmanager
def crossing_barrier():
    _CONTEXT.is_crossing_barrier = True
    try:
        yield
    finally:
        _CONTEXT.is_crossing_barrier = False


def _update_ref_context(pytree: tp.Any) -> tp.Any:
    for pytree in jax.tree_util.tree_leaves(
        pytree, is_leaf=lambda x: isinstance(x, PytreeRef)
    ):
        if isinstance(pytree, PytreeRef):
            pytree.ref._trace_level = tracers.current_trace_level()
            pytree.ref._ref_context = _CONTEXT.current_ref_context


def cross_barrier(
    decorator, *decorator_args, **decorator_kwargs
) -> tp.Callable[[F], F]:
    @functools.wraps(decorator)
    def decorator_wrapper(f):
        @functools.wraps(f)
        def inner_wrapper(*args, **kwargs):
            _CONTEXT.is_crossing_barrier = False
            # _CONTEXT.barrier_cache = None # Note: barrier_cache is not used for now
            with incremented_ref():
                _update_ref_context((args, kwargs))
                out = f(*args, **kwargs)
            _CONTEXT.is_crossing_barrier = True
            # _CONTEXT.barrier_cache = {} # Note: barrier_cache is not used for now
            return out

        decorated = decorator(inner_wrapper, *decorator_args, **decorator_kwargs)

        @functools.wraps(f)
        def outer_wrapper(*args, **kwargs):
            args, kwargs = clone_references((args, kwargs))
            with crossing_barrier():
                out = decorated(*args, **kwargs)
            out = clone_references(out)
            return out

        return outer_wrapper

    return decorator_wrapper


class Ref(tp.Generic[A]):
    def __init__(self, value: A, id: tp.Optional[ids.Id] = None):
        self._value = value
        self._ref_context = _CONTEXT.current_ref_context
        self._id = ids.uuid() if id is None else id
        self._trace_level = tracers.current_trace_level()

    @property
    def id(self) -> ids.Id:
        return self._id

    @property
    def value(self) -> A:
        return self._value

    @value.setter
    def value(self, value: A):
        if (
            self._ref_context is not _CONTEXT.current_ref_context
            and not _CONTEXT.is_crossing_barrier
        ):
            raise ValueError("Cannot mutate ref from different context")
        if (
            self._trace_level != tracers.current_trace_level()
            and not _CONTEXT.is_crossing_barrier
        ):
            raise ValueError("Cannot mutate ref from different trace level")
        self._value = value


class PytreeRef(tp.Generic[A]):
    def __init__(self, ref_or_value: tp.Union[Ref[A], A]):
        if isinstance(ref_or_value, Ref):
            self._ref = ref_or_value
        else:
            self._ref = Ref(ref_or_value)

    @property
    def ref(self) -> Ref[A]:
        return self._ref

    @property
    def id(self) -> ids.Id:
        return self.ref.id

    @property
    def value(self) -> A:
        return self.ref.value

    @value.setter
    def value(self, value: A):
        self.ref.value = value


def flatten_pytree_ref(pytree: PytreeRef[A]) -> tp.Tuple[tp.Tuple[A], Ref[A]]:
    return (pytree.value,), pytree.ref


def unflatten_pytree_ref(ref: Ref[A], children: tp.Tuple[A]) -> PytreeRef[A]:
    value = children[0]
    if _CONTEXT.barrier_cache is not None:
        if ref.id not in _CONTEXT.barrier_cache:
            _CONTEXT.barrier_cache[ref.id] = Ref(value, id=ref.id)

        ref = _CONTEXT.barrier_cache[ref.id]
    else:
        ref.value = value
    return PytreeRef(ref)


jax.tree_util.register_pytree_node(PytreeRef, flatten_pytree_ref, unflatten_pytree_ref)


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
        return getattr(obj, f"_ref_{self.name}").value

    def __set__(self, obj, value: tp.Union[A, Ref[A], PytreeRef[A], "RefField[A]"]):
        if isinstance(value, RefField):
            return

        if hasattr(obj, f"_ref_{self.name}"):
            if isinstance(value, (Ref, PytreeRef, RefField)):
                raise AttributeError(f"Cannot change reference of {self.name}")
            getattr(obj, f"_ref_{self.name}").value = value
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
