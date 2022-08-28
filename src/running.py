from pipeline.information_extraction_models import InformationExtractionModel
from pipeline.tokenizer_models import Tokenizer


class Runner:
    def __init__(self, tokenizer: Tokenizer, model: InformationExtractionModel):
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model

    def run(self):
        tokens = self._tokenizer.encode(["Hello world"])
        print(tokens)
        print(self._model.predict(tokens))
