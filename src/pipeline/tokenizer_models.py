from abc import abstractmethod, ABC
from typing import Dict, Union, Any, List

from transformers.tokenization_utils_base import AddedToken  # type: ignore
from config import TokenizerConfig
import torch
import numpy as np  # NOQA


TokenRepresentation = Union[int, List[int], "np.ndarray[Any, Any]", "torch.Tensor"]
TextRepresentation = Union[str, List[str]]


class Tokenizer(ABC):
    @abstractmethod
    def __init__(self, config: TokenizerConfig) -> None:
        ...

    @abstractmethod
    def encode(self, text: TextRepresentation) -> TokenRepresentation:
        ...

    @abstractmethod
    def decode(self, token_ids: TokenRepresentation) -> TextRepresentation:
        ...


class BaseTokenizer(Tokenizer):
    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        self._config = config


class GPT2Tokenizer(BaseTokenizer):
    def __init__(self, config: TokenizerConfig):
        from transformers.models.gpt2.tokenization_gpt2 import (
            GPT2Tokenizer as HuggingFaceGPT2Tokenizer,
        )

        self._tokenizer: HuggingFaceGPT2Tokenizer = HuggingFaceGPT2Tokenizer.from_pretrained("gpt2")  # type: ignore
        self._tokenizer.add_special_tokens(config.special_tokens)
        super().__init__(config)

    def _add_special_tokens(self, tokens: Dict[str, Union[str, AddedToken]]) -> None:
        assert self._tokenizer is not None
        self._tokenizer.add_special_tokens(tokens)

    def encode(self, text: TextRepresentation) -> TokenRepresentation:
        return self._tokenizer(text)["input_ids"]  # type: ignore

    def decode(self, token_ids: TokenRepresentation) -> TextRepresentation:
        return self._tokenizer.decode(token_ids, self._config.skip_special_tokens)  # type: ignore


class CharTokenizer(BaseTokenizer):
    def encode(self, text: TextRepresentation) -> TokenRepresentation:
        batch = [text] if isinstance(text, str) else text
        max_length = max(len(t) for t in batch)
        encoding = torch.zeros(len(batch), max_length, dtype=torch.int64)
        for i, t in enumerate(batch):
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
