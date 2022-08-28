"""A class for running the code."""

from pipeline.information_extraction_models import InformationExtractionModel
from pipeline.tokenizer_models import Tokenizer


class Runner:
    """A class for running the code."""

    def __init__(self, tokenizer: Tokenizer, model: InformationExtractionModel) -> None:
        """Initialize the runner.

        Parameters
        ----------
        tokenizer : Tokenizer
            The tokenizer.
        model : InformationExtractionModel
            The model.
        """
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model

    def run(self) -> None:
        """Run the code."""
        tokens = self._tokenizer.encode(["Hello [CLS] world"])
        print(self._model.predict(tokens))
