from dataclasses import dataclass
from typing import Dict, Union, Any, List, Optional
from abc import abstractmethod, ABC
from omegaconf import DictConfig
from transformers.tokenization_utils_base import AddedToken  # type: ignore
import torch
import numpy as np  # NOQA
from transformers.models.gpt2.tokenization_gpt2 import (
    GPT2Tokenizer as HuggingFaceGPT2Tokenizer,
)

_primary_token_classes = Union[int, List[int], "np.ndarray[Any, Any]", "torch.Tensor"]


TokenRepresentation = _primary_token_classes
TextRepresentation = List[str]


@dataclass
class GPT2TokenRepresentation:
    input_ids: _primary_token_classes
    attention_mask: _primary_token_classes


class Tokenizer(ABC):
    def load_config(self, config: DictConfig) -> Any:
        ...

    @abstractmethod
    def encode(self, text: TextRepresentation) -> Any:
        ...

    @abstractmethod
    def decode(self, token_ids: Any) -> TextRepresentation:
        ...


class GPT2Tokenizer(Tokenizer):
    _tokenizer: Optional[HuggingFaceGPT2Tokenizer] = None
    _config: Optional[DictConfig] = None

    def load_config(self, config: DictConfig) -> Any:
        tokenizer: HuggingFaceGPT2Tokenizer = HuggingFaceGPT2Tokenizer.from_pretrained("gpt2")  # type: ignore
        tokenizer.add_special_tokens(config.special_tokens)
        self._tokenizer = tokenizer
        self._config = config

    def _get_tokenizer(self) -> HuggingFaceGPT2Tokenizer:
        assert (
            self._tokenizer is not None
        ), "Tokenizer not set; Please use the load_config() method first"
        return self._tokenizer

    def _add_special_tokens(self, tokens: Dict[str, Union[str, AddedToken]]) -> None:
        self._get_tokenizer().add_special_tokens(tokens)

    def encode(self, text: TextRepresentation) -> GPT2TokenRepresentation:
        assert not isinstance(text, str)
        assert self._config is not None
        tokenizer = self._get_tokenizer()
        result = tokenizer(
            text,
            return_tensors="pt",
            truncation=self._config.truncation,
            padding=self._config.padding,
            max_length=self._config.max_length,
        )
        return GPT2TokenRepresentation(**result)  # type: ignore

    def decode(self, token_ids: GPT2TokenRepresentation) -> TextRepresentation:
        return self._get_tokenizer().batch_decode(token_ids.input_ids, skip_special_tokens=self._config.skip_special_tokens)  # type: ignore


class CharTokenizer(Tokenizer):
    def encode(self, text: TextRepresentation) -> TokenRepresentation:
        assert not isinstance(text, str)
        max_length = max(len(t) for t in text)
        encoding = torch.zeros(len(text), max_length, dtype=torch.int64)
        for i, t in enumerate(text):
            encoding[i, : len(t)] = torch.LongTensor(list(map(ord, t))) + 1
        return encoding[0] if isinstance(text, str) else encoding

    def _decode_int(self, token_ids: int) -> str:
        return chr(token_ids - 1) if token_ids != 0 else ""

    def _decode_list(self, token_ids: List[int]) -> str:
        if len(token_ids) > 0 and isinstance(token_ids[0], List):
            return [self.decode(item) for item in token_ids]  # type: ignore
        else:
            return "".join(map(self._decode_int, token_ids))

    def _decode_torch(self, token_ids: torch.Tensor) -> TextRepresentation:
        results = token_ids.tolist()  # type: ignore
        return self.decode(results)  # type: ignore

    def decode(self, token_ids: TokenRepresentation) -> TextRepresentation:
        decoders = {
            int: self._decode_int,
            List: self._decode_list,
            torch.Tensor: self._decode_torch,
        }
        for decode_type, decoder_fn in decoders.items():
            if isinstance(token_ids, decode_type):
                return decoder_fn(token_ids)  # type: ignore
        raise ValueError(f"Unsupported token_ids type: {type(token_ids)}")
