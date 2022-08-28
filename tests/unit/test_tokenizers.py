from pipeline.tokenizer_models import CharTokenizer, Tokenizer
from hypothesis import example, given, strategies as st, settings
import pytest


tokenizers = [CharTokenizer()]


@pytest.mark.parametrize("tokenizer", tokenizers)
@given(text=st.text(min_size=3))
@settings(max_examples=100, deadline=None)
@example(text="")
def test_all_tokenizers(tokenizer: Tokenizer, text: str):
    assert tokenizer.decode(tokenizer.encode([text]))[0] == text


@pytest.mark.parametrize("tokenizer", tokenizers)
def test_no_string_allowed(tokenizer: Tokenizer):
    with pytest.raises(Exception):
        tokenizer.encode("")  # type: ignore


@pytest.mark.parametrize("tokenizer", tokenizers)
def test_tokenizer_encode_raises_error_when_text_is_none(tokenizer: Tokenizer):
    with pytest.raises(Exception):
        tokenizer.encode(None)  # type: ignore


@pytest.mark.parametrize("tokenizer", tokenizers)
def test_tokenizer_decode_raises_error_when_text_is_none(tokenizer: Tokenizer):
    with pytest.raises(Exception):
        tokenizer.decode(None)
