"""Definitions of exception types."""

from typing import Any, List, Optional


class ClassNotFoundException(BaseException):
    """Exception when a dynamically loaded class is not found."""

    def __init__(
        self,
        abc_class: Any,
        class_name: str,
        candidates: Optional[List[str]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the exception.

        Parameters
        ----------
        abc_class : Any
            The superclass (ABC class) for the class that was expected to be found.
        class_name : str
            The name of the class.
        candidates : Optional[List[str]]
            A list of candidate class names.
        args : Any
            Additional positional arguments.
        kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.abc_class = abc_class
        self.class_name = class_name
        self.candidates = candidates

    def __str__(self) -> str:
        """Build a string representation of the exception.

        Returns
        -------
        str
            The string representation of the exception.
        """
        option_list = self.candidates or []
        output = f'{self.abc_class.__name__} subclass "{self.class_name}" not found; Options: {option_list}'
        return output


class UnsupportedTokenTypeException(BaseException):
    """Exception when a token type is not supported."""

    def __init__(self, token_type: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the exception.

        Parameters
        ----------
        token_type : Any
            The token type.
        args : Any
            Additional positional arguments.
        kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.token_type = token_type

    def __str__(self) -> str:
        """Build a string representation of the exception.

        Returns
        -------
        str
            The string representation of the exception.
        """
        return f'Token type "{self.token_type}" not supported'
