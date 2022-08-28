"""A class for running the code."""

from dataset import ConfiguredDataset
from pipeline.information_extraction_models import InformationExtractionModel
from pipeline.tokenizer_models import Tokenizer
from typing import Optional


class Runner:
    """A class for running the code."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        model: InformationExtractionModel,
        train_dataset: Optional[ConfiguredDataset] = None,
        test_dataset: Optional[ConfiguredDataset] = None,
        validation_dataset: Optional[ConfiguredDataset] = None,
    ) -> None:
        """Initialize the runner.

        Parameters
        ----------
        tokenizer : Tokenizer
            The tokenizer.
        model : InformationExtractionModel
            The model.
        train_dataset : Optional[ConfiguredDataset]
            The train dataset.
        test_dataset : Optional[ConfiguredDataset]
            The test dataset.
        validation_dataset : Optional[ConfiguredDataset]
            The validation dataset.
        """
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._validation_dataset = validation_dataset

    def run(self) -> None:
        """Run the code."""
        tokens = self._tokenizer.encode(["Hello [CLS] world"])
        print(self._model.predict(tokens))
