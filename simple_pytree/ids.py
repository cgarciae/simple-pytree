# Taken from flax/ids.py 🏴‍☠️

import threading


class UUIDManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._id = 0

    def __call__(self):
        with self._lock:
            self._id += 1
            return Id(self._id)


uuid = UUIDManager()


class Id:
    def __init__(self, rawid):
        self.id = rawid

    def __eq__(self, other):
        return isinstance(other, Id) and other.id == self.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"Id({self.id})"

    def __deepcopy__(self, memo):
        del memo
        return uuid()

    def __copy__(self):
        return uuid()
