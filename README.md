
<!-- codecov badge -->
[![codecov](https://codecov.io/gh/cgarciae/simple-pytree/branch/main/graph/badge.svg?token=3IKEUAU3C8)](https://codecov.io/gh/cgarciae/simple-pytree)


# Simple Pytree

A _dead simple_ Python package for creating custom JAX pytree objects.

* Strives to be minimal, the implementation is just ~100 lines of code
* Has no dependencies other than JAX
* Its compatible with both `dataclasses` and regular classes
* It has no intention of supporting Neural Network use cases (e.g. partitioning)

<details><summary>What about Equinox, Treeo, etc?</summary>

Most pytree-based neural network libraries start simple but end up adding
a lot of features that are not needed for simple pytree objects. `flax.struct.PytreeNode`
is the simplest one out there, but it has two downsides:

1. Forces you to use `dataclasses`, which is not a bad thing but not always
what you want.
2. It requires you to install `flax` just to use it.

</details>

## Installation

```bash
pip install simple-pytree
```

## Usage

```python
import jax
from simple_pytree import Pytree

class Foo(Pytree):
    def __init__(self, x, y):
        self.x = x
        self.y = y

foo = Foo(1, 2)
foo = jax.tree_map(lambda x: -x, foo)

assert foo.x == -1 and foo.y == -2
```

### Static fields
You can mark fields as static by assigning `static_field()` to a class attribute with the same name 
as the instance attribute:

```python
import jax
from simple_pytree import Pytree, static_field

class Foo(Pytree):
    y = static_field()
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

foo = Foo(1, 2)
foo = jax.tree_map(lambda x: -x, foo) # y is not modified

assert foo.x == -1 and foo.y == 2
```

Static fields are not included in the pytree leaves, they
are passed as pytree metadata instead.

### Dataclasses
You can seamlessly use the `dataclasses.dataclass` decorator with `Pytree` classes.
Since `static_field` returns instances of `dataclasses.Field` these it will work as expected:

```python
import jax
from dataclasses import dataclass
from simple_pytree import Pytree, static_field

@dataclass
class Foo(Pytree):
    x: int
    y: int = static_field(2) # with default value
    
foo = Foo(1)
foo = jax.tree_map(lambda x: -x, foo) # y is not modified

assert foo.x == -1 and foo.y == 2
```

### Mutability
`Pytree` objects are immutable by default after `__init__`:

```python
from simple_pytree import Pytree, static_field

class Foo(Pytree):
    y = static_field()
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

foo = Foo(1, 2)
foo.x = 3 # AttributeError
```
If you want to make them mutable, you can use the `mutable` argument in class definition:

```python
from simple_pytree import Pytree, static_field

class Foo(Pytree, mutable=True):
    y = static_field()
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

foo = Foo(1, 2)
foo.x = 3 # OK
```

### Replacing fields

If you want to make a copy of a `Pytree` object with some fields modified, you can use the `.replace()` method:

```python
from simple_pytree import Pytree, static_field

class Foo(Pytree):
    y = static_field()
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

foo = Foo(1, 2)
foo = foo.replace(x=10)

assert foo.x == 10 and foo.y == 2
```

`replace` works for both mutable and immutable `Pytree` objects. If the class
is a `dataclass`, `replace` internally use `dataclasses.replace`.

