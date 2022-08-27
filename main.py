import hydra
from hydra.core.config_store import ConfigStore
from config import Config, TokenizerConfig
from omegaconf import DictConfig, OmegaConf
from exception_types import ClassNotFoundException

from running import Runner
from types import SimpleNamespace
import pipeline.tokenizer_models
from pipeline.tokenizer_models import Tokenizer, BaseTokenizer
from utils import load_class_instances

config_store = ConfigStore.instance()
config_store.store(name="config", node=Config)


def load_tokenizer(config: TokenizerConfig) -> BaseTokenizer:
    # Dynamically load the tokenizer class based on the config
    tokenizers = load_class_instances(
        module=pipeline.tokenizer_models, abc_class=Tokenizer, base_class=BaseTokenizer
    )
    if config.name in tokenizers:
        return tokenizers[config.name](config=config)  # type: ignore
    else:
        raise ClassNotFoundException(Tokenizer, config.name, list(tokenizers.keys()))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def app(dict_config: DictConfig) -> None:
    if missing_keys := OmegaConf.missing_keys(dict_config):
        raise RuntimeError(f"Got missing keys in the config: {missing_keys}")
    config: Config = SimpleNamespace(**dict_config)  # type: ignore

    tokenizer = load_tokenizer(config.pipeline.tokenizer)
    runner = Runner(tokenizer=tokenizer)
    runner.run()


if __name__ == "__main__":
    app()
