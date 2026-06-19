from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzSegmentationDataset(Dataset):
    """Dataset for preprocessed 2D segmentation samples stored as `.npz` files."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Any | None = None,
        require_boundary: bool = False,
        require_sdf: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.require_boundary = require_boundary
        self.require_sdf = require_sdf
        self.files = sorted((self.root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz files found under {self.root / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index]
        with np.load(path) as sample:
            image = sample["image"].astype(np.float32)
            mask = sample["mask"].astype(np.int64)
            item: dict[str, Any] = {
                "image": image,
                "mask": mask,
                "id": path.stem,
            }
            if "boundary" in sample:
                item["boundary"] = sample["boundary"].astype(np.float32)
            elif self.require_boundary:
                raise KeyError(f"{path} is missing required 'boundary' array")
            if "sdf" in sample:
                item["sdf"] = sample["sdf"].astype(np.float32)
            elif self.require_sdf:
                raise KeyError(f"{path} is missing required 'sdf' array")
            if "spacing" in sample:
                item["spacing"] = sample["spacing"].astype(np.float32)

        if self.transform is not None:
            item = self.transform(item)

        output: dict[str, torch.Tensor | str] = {
            "image": torch.from_numpy(np.ascontiguousarray(item["image"])).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(item["mask"])).long(),
            "id": str(item["id"]),
        }
        if "boundary" in item:
            boundary = item["boundary"]
            if boundary.ndim == 2:
                boundary = boundary[None, ...]
            output["boundary"] = torch.from_numpy(np.ascontiguousarray(boundary)).float()
        if "sdf" in item:
            sdf = item["sdf"]
            if sdf.ndim == 2:
                sdf = sdf[None, ...]
            output["sdf"] = torch.from_numpy(np.ascontiguousarray(sdf)).float()
        if "spacing" in item:
            output["spacing"] = torch.from_numpy(np.ascontiguousarray(item["spacing"])).float()
        return output
