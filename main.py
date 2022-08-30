"""The entrypoint."""

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from dataset import ConfiguredDataset, ProvisionDatasetBatchTransformer, collate_fn
from pipeline.language_models import LanguageModel
import pipeline.language_models
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
        module=pipeline.language_models,
        protocol=LanguageModel,
        class_name=config.pipeline.model.name,
    )
    model.load_config(config.pipeline.model)
    model.handle_tokenizer(tokenizer)

    train_dataset = ConfiguredDataset(config.dataset.train_dataset)
    test_dataset = ConfiguredDataset(config.dataset.test_dataset)
    validation_dataset = ConfiguredDataset(config.dataset.validation_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.experiment.params.batch_size,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=len(test_dataset), collate_fn=collate_fn
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=len(validation_dataset), collate_fn=collate_fn
    )

    batch_transformer = ProvisionDatasetBatchTransformer()

    runner = Runner(
        tokenizer=tokenizer,
        model=model,
        batch_transformer=batch_transformer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        validation_dataloader=validation_dataloader,
    )
    runner.run(config)


if __name__ == "__main__":
    app()
