from dataclasses import dataclass, field
from typing import Any, Dict, List

from omegaconf import MISSING


@dataclass
class PathConfig:
    log: str = MISSING
    data: str = MISSING


@dataclass
class FileConfig:
    train_data: str = MISSING
    train_labels: str = MISSING
    test_data: str = MISSING
    test_labels: str = MISSING


@dataclass
class ParamConfig:
    epoch_count: int = MISSING
    lr: float = MISSING
    batch_size: int = MISSING


@dataclass
class TokenizerConfig:
    name: str = MISSING
    special_tokens: Dict[str, Any] = field(default_factory=dict)
    skip_special_tokens: bool = True


@dataclass
class ModelConfig:
    name: str = MISSING


@dataclass
class PipelineConfig:
    tokenizer: TokenizerConfig = MISSING
    model: ModelConfig = MISSING


@dataclass
class ExperimentConfig:
    name: str = MISSING


@dataclass
class Config:
    experiment: ExperimentConfig = ExperimentConfig()
    paths: PathConfig = PathConfig()
    files: FileConfig = FileConfig()
    params: ParamConfig = ParamConfig()
    pipeline: PipelineConfig = PipelineConfig()

    defaults: List[str] = field(default_factory=lambda: ["_self_", "defaults"])
