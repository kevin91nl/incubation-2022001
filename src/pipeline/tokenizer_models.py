"""Tokenizer definitions."""

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

from exception_types import UnsupportedTokenTypeException

_primary_token_classes = Union[int, List[int], "np.ndarray[Any, Any]", "torch.Tensor"]


TokenRepresentation = _primary_token_classes
TextRepresentation = List[str]


@dataclass
class GPT2TokenRepresentation:
    """The GPT2 token representation."""

    input_ids: _primary_token_classes
    attention_mask: _primary_token_classes


class Tokenizer(ABC):
    """Abstract class for tokenizers."""

    def load_config(self, config: DictConfig) -> Any:
        """Load the configuration.

        Parameters
        ----------
        config : DictConfig
            The configuration.

        Returns
        -------
        Any
            The loaded configuration.
        """
        ...

    @abstractmethod
    def encode(self, text: TextRepresentation) -> Any:
        """Encode the text.

        Parameters
        ----------
        text : TextRepresentation
            The text.

        Returns
        -------
        Any
            The encoded text.
        """
        ...

    @abstractmethod
    def decode(self, token_ids: Any) -> TextRepresentation:
        """Decode the token ids.

        Parameters
        ----------
        token_ids : Any
            The token ids.

        Returns
        -------
        TextRepresentation
            The decoded text.
        """
        ...


class GPT2Tokenizer(Tokenizer):
    """The GPT2 tokenizer."""

    _tokenizer: Optional[HuggingFaceGPT2Tokenizer] = None
    _config: Optional[DictConfig] = None

    def load_config(self, config: DictConfig) -> None:
        """Load the configuration.

        Parameters
        ----------
        config : DictConfig
            The configuration.
        """
        tokenizer: HuggingFaceGPT2Tokenizer = HuggingFaceGPT2Tokenizer.from_pretrained("gpt2")  # type: ignore
        tokenizer.add_special_tokens(config.special_tokens)
        self._tokenizer = tokenizer
        self._config = config

    def _get_tokenizer(self) -> HuggingFaceGPT2Tokenizer:
        """Get the tokenizer.

        Returns
        -------
        HuggingFaceGPT2Tokenizer
            The tokenizer.
        """
        assert (
            self._tokenizer is not None
        ), "Tokenizer not set; Please use the load_config() method first"
        return self._tokenizer

    def _add_special_tokens(self, tokens: Dict[str, Union[str, AddedToken]]) -> None:
        """Add special tokens to the tokenizer.

        Parameters
        ----------
        tokens : Dict[str, Union[str, AddedToken]]
            The tokens to add.
        """
        self._get_tokenizer().add_special_tokens(tokens)

    def encode(self, text: TextRepresentation) -> GPT2TokenRepresentation:
        assert not isinstance(text, str)
        """Encode the text.

        Parameters
        ----------
        text : TextRepresentation
            The text.

        Returns
        -------
        GPT2TokenRepresentation
            The encoded text.
        """
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
        """Decode the token ids.

        Parameters
        ----------
        token_ids : GPT2TokenRepresentation
            The token ids.

        Returns
        -------
        TextRepresentation
            The decoded text.
        """
        return self._get_tokenizer().batch_decode(token_ids.input_ids, skip_special_tokens=self._config.skip_special_tokens)  # type: ignore


class CharTokenizer(Tokenizer):
    """A character-based tokenizer.

    Examples
    --------
    >>> text = "Hello world!"
    >>> tokenizer = CharTokenizer()

    Generate the token IDs for all the characters in the text.
    >>> token_ids = tokenizer.encode([text])
    >>> print(token_ids)
    tensor([[ 73, 102, 109, 109, 112,  33, 120, 112, 115, 109, 101,  34]])

    Now, decode the token IDs back to the original text.
    >>> tokenizer.decode(token_ids)
    ['Hello world!']
    """

    def encode(self, text: TextRepresentation) -> TokenRepresentation:
        """Encode the text.

        Parameters
        ----------
        text : TextRepresentation
            The text.

        Returns
        -------
        TokenRepresentation
            The encoded text.
        """
        assert not isinstance(text, str)
        max_length = max(len(t) for t in text)
        encoding = torch.zeros(len(text), max_length, dtype=torch.int64)
        for i, t in enumerate(text):
            encoding[i, : len(t)] = torch.LongTensor(list(map(ord, t))) + 1
        return encoding[0] if isinstance(text, str) else encoding

    def _decode_int(self, token_ids: int) -> str:
        """Decode the token id where the input is an integer.

        Parameters
        ----------
        token_ids : int
            The token id.

        Returns
        -------
        str
            The decoded token.
        """
        return chr(token_ids - 1) if token_ids != 0 else ""

    def _decode_list(self, token_ids: List[int]) -> str:
        """Decode the token ids where the input is a list.

        Parameters
        ----------
        token_ids : List[int]
            The token ids.

        Returns
        -------
        str
            The decoded text.
        """
        if len(token_ids) > 0 and isinstance(token_ids[0], List):
            return [self.decode(item) for item in token_ids]  # type: ignore
        else:
            return "".join(map(self._decode_int, token_ids))

    def _decode_torch(self, token_ids: torch.LongTensor) -> TextRepresentation:
        """Decode the token ids where the input is a torch tensor.

        Parameters
        ----------
        token_ids : LongTensor
            The token ids.

        Returns
        -------
        TextRepresentation
            The decoded text.
        """
        results = token_ids.tolist()  # type: ignore
        return self.decode(results)  # type: ignore

    def decode(self, token_ids: TokenRepresentation) -> TextRepresentation:
        """Decode the token ids.

        Parameters
        ----------
        token_ids : TokenRepresentation
            The token ids.

        Returns
        -------
        TextRepresentation
            The decoded text.

        Raises
        ------
        UnsupportedTokenTypeException
            If the input is not a supported type.
        """
        decoders = {
            int: self._decode_int,
            List: self._decode_list,
            torch.Tensor: self._decode_torch,
        }
        for decode_type, decoder_fn in decoders.items():
            if isinstance(token_ids, decode_type):
                return decoder_fn(token_ids)  # type: ignore
        raise UnsupportedTokenTypeException(type(token_ids))
