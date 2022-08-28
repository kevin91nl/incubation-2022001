"""Definition of the models."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from omegaconf import DictConfig
from exception_types import PreprocessingStepMissingException
from pipeline.tokenizer_models import GPT2TokenRepresentation, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model as HuggingFaceGPT2Model


class InformationExtractionModel(ABC):
    """Abstract class for information extraction models."""

    def load_config(self, config: DictConfig) -> None:
        """Load the configuration.

        Parameters
        ----------
        config : DictConfig
            The configuration.
        """
        ...

    def handle_tokenizer(self, tokenizer: Any) -> None:
        """Handle the tokenizer.

        Parameters
        ----------
        tokenizer : Any
            The tokenizer.
        """
        ...

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Predict given the inputs.

        Parameters
        ----------
        inputs : Any
            The inputs.

        Returns
        -------
        Any
            The predictions.
        """
        ...


class GPT2Model(InformationExtractionModel):
    """The GPT2 model."""

    _model: Optional[HuggingFaceGPT2Model] = None

    def load_config(self, config: DictConfig) -> None:
        """Load the configuration.

        Parameters
        ----------
        config : DictConfig
            The configuration.
        """
        model: HuggingFaceGPT2Model = HuggingFaceGPT2Model.from_pretrained("gpt2")  # type: ignore
        self._model = model

    def handle_tokenizer(self, tokenizer: GPT2Tokenizer) -> None:
        """Handle the tokenizer.

        Parameters
        ----------
        tokenizer : GPT2Tokenizer
            The tokenizer.

        Raises
        ------
        PreprocessingStepMissingException
            If the required preprocessing step is not executed.
        """
        if self._model is None:
            raise PreprocessingStepMissingException(self.load_config)
        self._model.resize_token_embeddings(tokenizer.vocab_size)

    def _get_model(self) -> HuggingFaceGPT2Model:
        """Get the model.

        Returns
        -------
        HuggingFaceGPT2Model
            The model.
        """
        assert (
            self._model is not None
        ), "Model not set; Please use the load_config() method first"
        return self._model

    def predict(self, inputs: GPT2TokenRepresentation) -> Any:
        """Predict given the inputs.

        Parameters
        ----------
        inputs : GPT2TokenRepresentation
            The inputs.

        Returns
        -------
        Any
            The predictions.
        """
        return self._model(**inputs.__dict__)  # type: ignore
