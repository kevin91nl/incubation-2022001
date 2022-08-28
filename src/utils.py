from typing import Any, Dict, TypeVar, Optional, Type
from exception_types import ClassNotFoundException
import inspect


B = TypeVar("B", bound="Any")


def load_class_instance(module: Any, protocol: Type[B], class_name: str) -> B:
    candidates: Dict[str, B] = {
        name: cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if cls not in [protocol] and protocol in cls.__mro__
    }
    class_object: Optional[B] = candidates.get(class_name)
    if not class_object:
        raise ClassNotFoundException(protocol, class_name, list(candidates.keys()))
    return class_object()
