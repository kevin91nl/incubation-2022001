from abc import ABC, abstractmethod
from typing import List


class TextGenerationPipeline(ABC):
    @abstractmethod
    def predict(self, input_texts: List[str]) -> List[str]:
        """The predict method takes a list of texts as input and returns
        a list of generated texts as output."""
        ...
