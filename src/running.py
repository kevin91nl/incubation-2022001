from pipeline.tokenizer_models import Tokenizer


class Runner:
    def __init__(self, tokenizer: Tokenizer):
        super().__init__()
        self._tokenizer = tokenizer

    def run(self):
        tokens = self._tokenizer.encode("Hello world")
        print(tokens)
