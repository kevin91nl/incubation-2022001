from abc import ABC, abstractmethod
from typing import Any, Optional

from omegaconf import DictConfig
from pipeline.tokenizer_models import GPT2TokenRepresentation
from transformers.models.gpt2.modeling_gpt2 import GPT2Model as HuggingFaceGPT2Model


class InformationExtractionModel(ABC):
    def load_config(self, config: DictConfig) -> Any:
        ...

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        ...


class GPT2Model(InformationExtractionModel):
    _model: Optional[HuggingFaceGPT2Model] = None

    def load_config(self, config: DictConfig) -> Any:
        model: HuggingFaceGPT2Model = HuggingFaceGPT2Model.from_pretrained("gpt2")  # type: ignore
        self._model = model

    def _get_model(self) -> HuggingFaceGPT2Model:
        assert (
            self._model is not None
        ), "Model not set; Please use the load_config() method first"
        return self._model

    def predict(self, inputs: GPT2TokenRepresentation) -> Any:
        return self._model(**inputs.__dict__)  # type: ignore
