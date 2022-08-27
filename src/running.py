from pipeline.tokenizer_models import BaseTokenizer


class Runner:
    def __init__(self, tokenizer: BaseTokenizer):
        super().__init__()
        self._tokenizer = tokenizer

    def run(self):
        tokens = self._tokenizer.encode("Hello world")
        print(tokens)
