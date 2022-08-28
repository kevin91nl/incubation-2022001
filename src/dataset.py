"""Dataset loaders."""
from dataclasses import dataclass
from omegaconf import DictConfig
from torch.utils.data import Dataset
import pandas as pd
from typing import List
from enum import Enum


class ProvisionType(Enum):
    """The provision type."""

    permission = "permission"
    prohibition = "prohibition"
    duty = "duty"


@dataclass
class ProvisionFrame:
    """The frame consisting of a provision type (text classification) and entities (named entity recognition)."""

    provision_type: str
    bearer: str
    action: str
    other_party: str


@dataclass
class DatasetItem:
    """A dataset item, consisting of the context and zero or more provisions."""

    context: str
    provisions: List[ProvisionFrame]


class ConfiguredDataset(Dataset):  # type: ignore
    """A configured dataset."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        config : DictConfig
            The configuration of the dataset.
        """
        super().__init__()
        self.data = pd.read_json(config.path, orient="records", lines=True)

    def __getitem__(self, index: int) -> DatasetItem:
        """Get the item at the given index.

        Parameters
        ----------
        index : int
            The index.

        Returns
        -------
        DatasetItem
            The item.
        """
        return DatasetItem(**self.data.iloc[index % len(self)].to_dict())

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.data)
