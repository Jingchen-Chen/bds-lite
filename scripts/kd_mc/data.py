"""Training data with cached multiclass teacher probabilities (N, K, 224, 224) float16."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.revision.training_support import ManifestDataset

CACHE_NAME = "train_teacher_prob_fp16.npy"
IDS_NAME = "train_ids.json"


class TeacherCacheDatasetMC(ManifestDataset):
    """ManifestDataset plus `teacher_prob` (K, 224, 224) in manifest order; no random draws."""

    def __init__(self, root, split, manifest, classes, channels, boundary, sdf, cache_dir: Path):
        super().__init__(root, split, manifest, classes, channels, boundary, sdf)
        self.cache_path = Path(cache_dir) / CACHE_NAME
        ids = json.loads((Path(cache_dir) / IDS_NAME).read_text())
        if ids != manifest["sample_ids"]:
            raise ValueError("teacher cache order differs from the training manifest")
        shape = np.load(self.cache_path, mmap_mode="r").shape
        if shape != (len(ids), classes, 224, 224):
            raise ValueError(f"teacher cache shape {shape} is wrong")
        self._cache = None

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self._cache is None:
            self._cache = np.load(self.cache_path, mmap_mode="r")
        item["teacher_prob"] = torch.from_numpy(np.array(self._cache[index], dtype=np.float16))
        return item

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state
