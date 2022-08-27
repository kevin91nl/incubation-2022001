from typing import Any, Dict, TypeVar
import inspect


B = TypeVar("B")


def load_class_instances(module: Any, abc_class: Any, base_class: B) -> Dict[str, B]:
    candidates: Dict[str, B] = {}
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if cls not in [abc_class, base_class]:
            if issubclass(cls, abc_class):
                candidates[name] = cls
    return candidates
