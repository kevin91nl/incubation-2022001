"""A class for running the code."""

from dataset import DatasetBatchTransformer, DatasetItem
from pipeline.language_models import LanguageModel
from pipeline.tokenizer_models import Tokenizer
from typing import List, Optional
from omegaconf import DictConfig
from tqdm import trange
from torch.utils.data import DataLoader


class Runner:
    """A class for running the code."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        model: LanguageModel,
        batch_transformer: DatasetBatchTransformer,
        train_dataloader: Optional[DataLoader] = None,  # type: ignore
        test_dataloader: Optional[DataLoader] = None,  # type: ignore
        validation_dataloader: Optional[DataLoader] = None,  # type: ignore
    ) -> None:
        """Initialize the runner.

        Parameters
        ----------
        tokenizer : Tokenizer
            The tokenizer.
        model : InformationExtractionModel
            The model.
        batch_transformer : DatasetBatchTransformer
            The batch transformer.
        train_dataloader : Optional[DataLoader]
            The train dataloader.
        test_dataloader : Optional[DataLoader]
            The test dataloader.
        validation_dataloader : Optional[DataLoader]
            The validation dataloader.
        """
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model
        self._batch_transformer = batch_transformer
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._validation_dataloader = validation_dataloader

    def run(self, config: DictConfig) -> None:
        """Run the code.

        Parameters
        ----------
        config : DictConfig
            The configuration.

        Raises
        ------
        MethodCallMissingException
            If the optimizer is not set up.
        """
        if self._train_dataloader:
            for _ in trange(config.experiment.params.epoch_count):
                self._model.setup_training(config)
                for batch in self._train_dataloader:
                    batch: List[DatasetItem] = batch
                    input_data = self._batch_transformer.transform(batch)
                    token_ids = self._tokenizer.encode(input_data)
                    outputs = self._model.predict(token_ids)
                    print(token_ids)
                    print("---")
                    return
