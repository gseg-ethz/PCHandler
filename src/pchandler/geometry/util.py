import copy
from functools import wraps


def bypass_immutable(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        original_state: bool = getattr(self, '_immutable', False)
        self.set_mutability(False)
        try:
            return method(self, *args, **kwargs)
        finally:
            self.set_mutability(original_state)
    return wrapper

def enforce_immutability(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, '_immutable', False):
            raise AttributeError()
        return method(self, *args, **kwargs)
    return wrapper


def return_copy(deep=True):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            if not self._mutability:
                return result
            return copy.deepcopy(result) if deep else copy.copy(result)
        return wrapper
    return decorator


