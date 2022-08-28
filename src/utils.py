from typing import Any, Dict, TypeVar, Optional, Type, Protocol
from exception_types import ClassNotFoundException
import inspect


B = TypeVar("B", bound="Any")


def load_class_instance(
    module: Any, protocol: Type[B], class_name: str, config: Optional[Any] = None
) -> B:
    candidates: Dict[str, B] = {}
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if cls not in [protocol]:
            if protocol in cls.__mro__:
                candidates[name] = cls
    class_object: Optional[B] = candidates.get(class_name)
    if not class_object:
        raise ClassNotFoundException(protocol, class_name, list(candidates.keys()))
    class_instance: ConfigLoader = class_object()
    if config:
        class_instance.load_config(config)
    return class_instance  # type: ignore


class ConfigLoader(Protocol):
    def load_config(self, config: Any) -> Any:
        ...
