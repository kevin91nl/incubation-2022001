"""The entrypoint."""

import hydra
from omegaconf import DictConfig
from dataset import ConfiguredDataset
from pipeline.information_extraction_models import InformationExtractionModel
import pipeline.information_extraction_models
from running import Runner
import pipeline.tokenizer_models
from pipeline.tokenizer_models import Tokenizer
from utils import load_class_instance


@hydra.main(config_path="conf", config_name="config", version_base=None)
def app(config: DictConfig) -> None:
    """Run the application.

    Parameters
    ----------
    config : DictConfig
        The configuration.
    """
    tokenizer = load_class_instance(
        module=pipeline.tokenizer_models,
        protocol=Tokenizer,
        class_name=config.pipeline.tokenizer.name,
    )
    tokenizer.load_config(config.pipeline.tokenizer)

    model = load_class_instance(
        module=pipeline.information_extraction_models,
        protocol=InformationExtractionModel,
        class_name=config.pipeline.model.name,
    )
    model.load_config(config.pipeline.model)
    model.handle_tokenizer(tokenizer)

    train_dataset = ConfiguredDataset(config.dataset.train_dataset)
    test_dataset = ConfiguredDataset(config.dataset.test_dataset)
    validation_dataset = ConfiguredDataset(config.dataset.validation_dataset)

    runner = Runner(
        tokenizer=tokenizer,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        validation_dataset=validation_dataset,
    )
    runner.run()


if __name__ == "__main__":
    app()
