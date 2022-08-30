"""Definition of the models."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from omegaconf import DictConfig
from exception_types import MethodCallMissingException
from pipeline.tokenizer_models import GPT2TokenRepresentation, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model as HuggingFaceGPT2Model
import torch
from transformers import AdamW  # type: ignore


class Model(ABC):
    """Abstract class for models."""

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

    @abstractmethod
    def train(self, input_batch: Any, target_batch: Any) -> None:
        """Train the model.

        Parameters
        ----------
        input_batch : Any
            The input batch.
        target_batch : Any
            The target batch.
        """
        ...


class LanguageModel(Model):
    """Abstract class for langugage models."""

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

    def setup_training(self, config: DictConfig) -> None:
        """Setup the model for training."""
        ...


class GPT2Model(LanguageModel):
    """The GPT2 model."""

    _model: Optional[HuggingFaceGPT2Model] = None
    _optimizer: Optional[torch.optim.Optimizer] = None

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
        MethodCallMissingException
            If the required preprocessing step is not executed.
        """
        if self._model is None:
            raise MethodCallMissingException(self.load_config)
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

    def setup_optimizer(self, config: DictConfig):
        """Setup the optimizer.

        Parameters
        ----------
        config : DictConfig
            The configuration.
        """
        model_params = list(self._model.named_parameters())  # type: ignore
        optimizer_params = [{"params": [param for _, param in model_params]}]
        self._optimizer = AdamW(optimizer_params, lr=config.pipeline.optimizer.learning_rate)  # type: ignore

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

    def named_parameters(self) -> Any:
        """Get the named parameters.

        Returns
        -------
        Any
            The named parameters.
        """
        return self._model.named_parameters()  # type: ignore

    def setup_training(self, config: DictConfig) -> None:
        """Setup the model for training.

        Parameters
        ----------
        config : DictConfig
            The configuration.

        Raises
        ------
        MethodCallMissingException
            If the required preprocessing step is not executed.
        """
        if self._model is None:
            raise MethodCallMissingException(self.load_config)
        if self._optimizer is None:
            self.setup_optimizer(config)
        self._model.train()

    def train(self, input_batch: Any, target_batch: Any) -> None:
        pass
