from dataclasses import asdict
from hypothesis import given, strategies as st, settings
from dataset import ConfiguredDataset, DatasetItem, ProvisionFrame
from tempfile import NamedTemporaryFile
from typing import List
import pandas as pd
from omegaconf import DictConfig


@given(
    st.lists(
        st.builds(
            DatasetItem, provisions=st.lists(st.builds(ProvisionFrame), min_size=1)
        ),
        min_size=2,
    )
)
@settings(deadline=None)
def test_dataset_item(items: List[DatasetItem]):
    file = NamedTemporaryFile()
    records = [asdict(item) for item in items]
    df = pd.DataFrame.from_records(records)
    df.to_json(file.name, orient="records", lines=True)
    dataset = ConfiguredDataset(DictConfig({"path": file.name}))
    assert len(dataset) == len(items)
