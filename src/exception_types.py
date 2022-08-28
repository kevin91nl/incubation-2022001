from typing import Any, List, Optional


class ClassNotFoundException(Exception):
    def __init__(
        self, abc_class: Any, class_name: str, candidates: Optional[List[str]]
    ):
        super().__init__()
        self.abc_class = abc_class
        self.class_name = class_name
        self.candidates = candidates

    def __str__(self):
        option_list = self.candidates or []
        output = f'{self.abc_class.__name__} subclass "{self.class_name}" not found; Options: {option_list}'
        return output
