"""Script for generating a synthetic dataset."""

from dataclasses import asdict
from typing import List
import hydra
from omegaconf import DictConfig
import os
from dataset import DatasetItem, ProvisionFrame, ProvisionType
import pandas as pd
import random


def fill_in_pattern(pattern: str, provision: ProvisionFrame) -> str:
    """Fill in the pattern with the given provision.

    Parameters
    ----------
    pattern : str
        The pattern.
    provision : ProvisionFrame
        The provision.

    Returns
    -------
    str
        The filled in pattern.
    """
    pattern = pattern.replace("{bearer}", provision.bearer)
    pattern = pattern.replace("{action}", provision.action)
    pattern = pattern.replace("{other_party}", provision.other_party)
    return pattern


def generate_provision(config: DictConfig) -> ProvisionFrame:
    """Generate a single provision.

    Parameters
    ----------
    config : DictConfig
        The configuration of the dataset generation.

    Returns
    -------
    ProvisionFrame
        The generated provision.
    """
    return ProvisionFrame(
        provision_type=random.choice(list(ProvisionType)).value,
        bearer=random.choice(config.bearers),
        action=random.choice(config.actions),
        other_party=random.choice(config.other_parties),
    )


def generate_dataset_item(config: DictConfig) -> DatasetItem:
    """Generate a single dataset item.

    Parameters
    ----------
    config : DictConfig
        The configuration of the dataset.

    Returns
    -------
    DatasetItem
        The generated dataset item.
    """
    provisions = [generate_provision(config) for _ in range(random.randint(0, 2))]
    context_patterns = []
    if len(provisions) == 0:
        if len(config.patterns["other"]) > 0:
            pattern = random.choice(config.patterns["other"])
            provision = generate_provision(config)
            pattern = fill_in_pattern(pattern, provision)
            context_patterns.append(pattern)
    else:
        for provision in provisions:
            pattern = random.choice(config.patterns[provision.provision_type])
            pattern = fill_in_pattern(pattern, provision)
            context_patterns.append(pattern)
    return DatasetItem(context=" ".join(context_patterns), provisions=provisions)


def generate_dataset_items(
    number_of_items: int, config: DictConfig
) -> List[DatasetItem]:
    """Generate the given number of dataset items.

    Parameters
    ----------
    number_of_items : int
        The number of items to generate.
    config : DictConfig
        The configuration of the dataset generation.

    Returns
    -------
    List[DatasetItem]
        The generated dataset items.
    """
    return [generate_dataset_item(config) for _ in range(number_of_items)]


def write_dataset_to_path(dataset: List[DatasetItem], path: str) -> None:
    """Write the dataset to the given path.

    Parameters
    ----------
    dataset : List[DatasetItem]
        The dataset.
    path : str
        The path to write the dataset to.
    """
    base_dir = os.path.dirname(path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    converted_items = [asdict(item) for item in dataset]
    pd.DataFrame.from_records(converted_items).to_json(
        path, orient="records", lines=True
    )


@hydra.main(
    config_path=os.path.join("..", "conf"),
    config_name="generate_synthetic_dataset",
    version_base=None,
)
def generate_synthetic_dataset(config: DictConfig) -> None:
    """Generate a synthetic dataset.

    Parameters
    ----------
    config : DictConfig
        The configuration.
    """
    random.seed(config.dataset_generation.seed)
    for dataset in config.dataset:
        dataset_config = config.dataset[dataset]
        items = generate_dataset_items(dataset_config.size, config.dataset_generation)
        write_dataset_to_path(items, dataset_config.path)


if __name__ == "__main__":
    generate_synthetic_dataset()
