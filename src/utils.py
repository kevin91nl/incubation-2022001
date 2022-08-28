"""A collection of utility functions."""

from typing import Any, Dict, TypeVar, Optional, Type, List
from exception_types import ClassNotFoundException
import inspect


B = TypeVar("B", bound="Any")


def list_class_instances(
    module: Any, protocol: Type[B], exclude: Optional[List[B]] = None
) -> Dict[str, Type[B]]:
    """List all classes in a module that implement a given protocol.

    Parameters
    ----------
    module : Any
        The module to search.
    protocol : Type[B]
        The protocol to search for.
    exclude : Optional[List[B]]
        A list of classes to exclude from the search.

    Returns
    -------
    Dict[str, Type[B]]
        A mapping from class name to class object.
    """
    return {
        name: cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if cls not in [protocol]
        and protocol in cls.__mro__
        and cls not in (exclude or [])
    }


def load_class_instance(module: Any, protocol: Type[B], class_name: str) -> B:
    """Load a class instance from a module.

    Parameters
    ----------
    module : Any
        The module to search.
    protocol : Type[B]
        The protocol to search for.
    class_name : str
        The name of the class to load.

    Returns
    -------
    B
        The class instance.

    Raises
    ------
    ClassNotFoundException
        _description_
    """
    candidates: Dict[str, B] = list_class_instances(module, protocol)  # type: ignore
    class_object: Optional[B] = candidates.get(class_name)
    if not class_object:
        raise ClassNotFoundException(protocol, class_name, list(candidates.keys()))
    return class_object()
