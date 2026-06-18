"""Split manifests must not leak subjects between train / val / test.

These tests guard the publication claim that splits are subject-disjoint by
construction. If anyone regenerates the split manifests with a per-slice
shuffle rather than a per-subject shuffle, these tests will fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPLITS = REPO / "splits"


def _stem_to_subject(stem: str, dataset: str) -> str:
    if dataset == "acdc":
        return stem.split("_frame")[0]
    if dataset == "synapse":
        return stem.split("_slice")[0]
    return stem


def _load(dataset: str, split: str) -> set[str]:
    path = SPLITS / dataset / f"{split}.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text())
    return {_stem_to_subject(stem, dataset) for stem in payload["sample_ids"]}


@pytest.mark.parametrize("dataset", ["acdc", "synapse"])
def test_subject_level_splits_are_disjoint(dataset: str) -> None:
    train = _load(dataset, "train")
    val = _load(dataset, "val")
    test = _load(dataset, "test")

    assert train and val and test, f"{dataset} splits should be non-empty"
    assert not (train & val), f"{dataset} train/val subject overlap: {sorted(train & val)[:5]}"
    assert not (train & test), f"{dataset} train/test subject overlap: {sorted(train & test)[:5]}"
    assert not (val & test), f"{dataset} val/test subject overlap: {sorted(val & test)[:5]}"


def test_isic2018_image_level_splits_are_disjoint() -> None:
    train = _load("isic2018", "train")
    val = _load("isic2018", "val")
    assert train and val
    assert not (train & val), "ISIC2018 train/val image-id overlap detected"


@pytest.mark.parametrize("dataset", ["acdc", "isic2018", "synapse"])
def test_manifests_record_seed_and_count(dataset: str) -> None:
    for split in ("train", "val", "test"):
        path = SPLITS / dataset / f"{split}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        assert payload["dataset"] == dataset
        assert payload["split"] == split
        assert payload["seed"] == 2026
        assert payload["count"] == len(payload["sample_ids"])
