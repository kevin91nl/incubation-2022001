from typing import Dict, Union, Any, List, Optional
from abc import abstractmethod, ABC
from transformers.tokenization_utils_base import AddedToken  # type: ignore
from config import TokenizerConfig
import torch
import numpy as np  # NOQA
from transformers.models.gpt2.tokenization_gpt2 import (
    GPT2Tokenizer as HuggingFaceGPT2Tokenizer,
)


TokenRepresentation = Union[int, List[int], "np.ndarray[Any, Any]", "torch.Tensor"]
TextRepresentation = Union[str, List[str]]


class Tokenizer(ABC):
    def load_config(self, config: TokenizerConfig) -> Any:
        ...

    @abstractmethod
    def encode(self, text: TextRepresentation) -> TokenRepresentation:
        ...

    @abstractmethod
    def decode(self, token_ids: TokenRepresentation) -> TextRepresentation:
        ...


class GPT2Tokenizer(Tokenizer):
    _tokenizer: Optional[HuggingFaceGPT2Tokenizer] = None

    def load_config(self, config: TokenizerConfig) -> Any:
        tokenizer: HuggingFaceGPT2Tokenizer = HuggingFaceGPT2Tokenizer.from_pretrained("gpt2")  # type: ignore
        tokenizer.add_special_tokens(config.special_tokens)
        self._tokenizer = tokenizer

    def _get_tokenizer(self) -> HuggingFaceGPT2Tokenizer:
        assert (
            self._tokenizer is not None
        ), "Tokenizer not set; Please use the load_config() method first"
        return self._tokenizer

    def _add_special_tokens(self, tokens: Dict[str, Union[str, AddedToken]]) -> None:
        self._get_tokenizer().add_special_tokens(tokens)

    def encode(self, text: TextRepresentation) -> TokenRepresentation:
        return self._get_tokenizer()(text)["input_ids"]  # type: ignore

    def decode(self, token_ids: TokenRepresentation) -> TextRepresentation:
        return self._get_tokenizer().decode(token_ids, self._config.skip_special_tokens)  # type: ignore


class CharTokenizer(Tokenizer):
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


def very_complex(
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z
):
    if 1 > 2:
        if 3 > 4:
            if 5 > 6:
                return 7
    if a > b:
        if c > d > e:
            if very_complex(
                z,
                y,
                x,
                w,
                v,
                u,
                t,
                s,
                r,
                q,
                p,
                o,
                n,
                m,
                l,
                k,
                j,
                i,
                h,
                g,
                f,
                e,
                d,
                c,
                b,
                a,
            ):
                return 8
    if 1 > 2:
        if 3 > 4:
            if 5 > 6:
                return 7
    if a > b:
        if c > d > e:
            if very_complex(
                z,
                y,
                x,
                w,
                v,
                u,
                t,
                s,
                r,
                q,
                p,
                o,
                n,
                m,
                l,
                k,
                j,
                i,
                h,
                g,
                f,
                e,
                d,
                c,
                b,
                a,
            ):
                return 8
