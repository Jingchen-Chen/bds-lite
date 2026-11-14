"""Tests for the multi-seed aggregation script."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path("scripts/prepare_seed_averaged_results.py")


def _load_script() -> object:
    spec = importlib.util.spec_from_file_location("prepare_seed_averaged_results", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@pytest.fixture()
def script_mod() -> object:
    return _load_script()


def test_read_mean_row_returns_none_for_missing_file(tmp_path: Path, script_mod: object) -> None:
    result = script_mod._read_mean_row(tmp_path / "nonexistent.csv")  # type: ignore[attr-defined]
    assert result is None


def test_read_mean_row_returns_mean_row(tmp_path: Path, script_mod: object) -> None:
    csv_path = tmp_path / "eval_val.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label", "dsc", "iou", "hd95", "assd", "boundary_f1"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "label": "1",
                "dsc": "0.8",
                "iou": "0.7",
                "hd95": "5.0",
                "assd": "2.0",
                "boundary_f1": "0.6",
            }
        )
        writer.writerow(
            {
                "label": "mean",
                "dsc": "0.8",
                "iou": "0.7",
                "hd95": "5.0",
                "assd": "2.0",
                "boundary_f1": "0.6",
            }
        )

    result = script_mod._read_mean_row(csv_path)  # type: ignore[attr-defined]
    assert result is not None
    assert abs(result["dsc"] - 0.8) < 1e-9


def test_main_writes_partial_results(
    tmp_path: Path, script_mod: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpts = tmp_path / "checkpoints"
    tables = tmp_path / "tables"
    ckpts.mkdir()
    tables.mkdir()

    exp_dir = ckpts / "my_exp_seed2026"
    exp_dir.mkdir()
    csv_path = exp_dir / "eval_val.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label", "dsc", "iou", "hd95", "assd", "boundary_f1"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "label": "mean",
                "dsc": "0.85",
                "iou": "0.78",
                "hd95": "10.0",
                "assd": "3.5",
                "boundary_f1": "0.5",
            }
        )

    monkeypatch.setattr(script_mod, "CHECKPOINTS_DIR", ckpts)
    monkeypatch.setattr(script_mod, "OUTPUT_DIR", tables)
    monkeypatch.setattr(
        script_mod,
        "SEED_GROUPS",
        {"my_exp": ["my_exp_seed2026", "my_exp_seed2027"]},
    )

    script_mod.main()  # type: ignore[attr-defined]

    out = tables / "seed_averaged_results.csv"
    assert out.exists()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    dsc_row = next(r for r in rows if r["metric"] == "dsc")
    assert abs(float(dsc_row["mean"]) - 0.85) < 1e-6
    assert "my_exp_seed2027" in dsc_row["missing"]
