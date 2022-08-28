import hydra
from hydra.core.config_store import ConfigStore
from config import Config
from omegaconf import DictConfig, OmegaConf

from running import Runner
from types import SimpleNamespace
import pipeline.tokenizer_models
from pipeline.tokenizer_models import Tokenizer
from utils import load_class_instance

config_store = ConfigStore.instance()
config_store.store(name="config", node=Config)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def app(dict_config: DictConfig) -> None:
    if missing_keys := OmegaConf.missing_keys(dict_config):
        raise RuntimeError(f"Got missing keys in the config: {missing_keys}")
    config: Config = SimpleNamespace(**dict_config)  # type: ignore

    tokenizer = load_class_instance(
        module=pipeline.tokenizer_models,
        protocol=Tokenizer,
        class_name=config.pipeline.tokenizer.name,
        config=config.pipeline.tokenizer,
    )

    runner = Runner(tokenizer=tokenizer)
    runner.run()


if __name__ == "__main__":
    app()
