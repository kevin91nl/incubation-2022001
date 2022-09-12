"""Dataset loaders."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from omegaconf import DictConfig
from torch.utils.data import Dataset
import pandas as pd
from typing import Any, Dict, List
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
        self.data: pd.DataFrame = pd.read_json(
            config.path, orient="records", lines=True
        )

    def __getitem__(self, index: int) -> Any:
        """Get the item at the given index.

        Parameters
        ----------
        index : int
            The index.

        Returns
        -------
        Any
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


def collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collate the batch.

    Parameters
    ----------
    batch : List[Dict[str, Any]]
        The batch.

    Returns
    -------
    List[Dict[str, Any]]
        The collated batch.
    """
    return batch


class DatasetBatchTransformer(ABC):
    """A dataset batch transformer."""

    @abstractmethod
    def transform(self, batch: List[Any]) -> Any:
        """Transform the batch.

        Parameters
        ----------
        batch : List[Any]
            The batch.

        Returns
        -------
        Any
            The transformed batch.
        """
        ...


class ProvisionDatasetBatchTransformer(DatasetBatchTransformer):
    """A batch transformer for the provision dataset."""

    def transform(self, batch: List[DatasetItem]) -> Any:
        """Transform the batch.

        Parameters
        ----------
        batch : List[DatasetItem]
            The batch.

        Returns
        -------
        Any
            The transformed batch.
        """
        return [item.context for item in batch]
