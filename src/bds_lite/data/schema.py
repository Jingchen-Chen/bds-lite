from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    processed_dir: Path
    in_channels: int
    num_classes: int
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"


@dataclass(frozen=True)
class SampleKeys:
    image: str = "image"
    mask: str = "mask"
    boundary: str = "boundary"
    sdf: str = "sdf"
    spacing: str = "spacing"
