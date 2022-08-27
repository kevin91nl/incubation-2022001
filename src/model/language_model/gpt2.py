from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from model.language_model.text_generation_pipeline import TextGenerationModel


@dataclass
class _TokenizerParams:
    padding: bool = True
    truncation: bool = True
    max_length: int = 512
    return_tensors: str = "pt"


@dataclass
class _ModelParams:
    return_dict_in_generate: bool = True
    output_scores: bool = True
    output_attentions: bool = True
    output_hidden_states: bool = True


@dataclass
class _ModelOutput:
    sequences: Any
    scores: Tuple[Any, ...]
    attentions: Tuple[Tuple[Any], ...]
    hidden_states: Tuple[Tuple[Any], ...]


class GPT2(TextGenerationModel):
    def __init__(
        self,
        tokenizer_params: Optional[_TokenizerParams] = None,
        model_params: Optional[_ModelParams] = None,
    ):
        super().__init__()
        # Execute the import here to only import it when it's actually used since
        # it's quite slow
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # type: ignore
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # type: ignore
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")  # type: ignore
        self.tokenizer_params = (
            tokenizer_params if tokenizer_params is not None else _TokenizerParams()
        )
        self.model_params = model_params if model_params is not None else _ModelParams()

    def predict(self, input_texts: List[str]) -> List[str]:
        # The process of generating a prediction
        input_data = self.tokenizer(input_texts, **self.tokenizer_params.__dict__)  # type: ignore
        # Only allow input tokens in the output:

        outputs: _ModelOutput = _ModelOutput(**self.model.generate(**input_data, **self.model_params.__dict__))  # type: ignore
        output_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)  # type: ignore

        # Ideas to test further:
        # input_data = model.tokenizer('Coffee is a great: choose DRINK or FOOD', **model.tokenizer_params.__dict__)
        # outputs = model.model(**input_data, labels=input_data["input_ids"])
        # This should perhaps be a mapping, not sure: outputs.logits[:, :, input_data["input_ids"].reshape(-1)].argmax(dim=-1)
        # model.tokenizer.batch_decode(outputs.logits.argmax(dim=-1))

        return output_texts


# "'book a flight from bejing to new york tomorrow morning' ; arrival refers to"
