from functools import wraps
import copy
from typing import Any

def bypass_immutable(method):
    @wraps(method)
    def wrapper(self, *args: Any, **kwargs: dict[str, Any]) -> Any:
        original_state: bool = getattr(self, '_immutable', False)
        self.set_immutability(False)
        try:
            return method(self, *args, **kwargs)
        finally:
            self.set_immutability(original_state)
    return wrapper


def return_copy(deep=True):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args: Any, **kwargs: dict[str, Any]) -> Any:
            result:Any = method(self, *args, **kwargs)
            if not self._immutable:
                return result
            return copy.deepcopy(result) if deep else copy.copy(result)
        return wrapper
    return decorator