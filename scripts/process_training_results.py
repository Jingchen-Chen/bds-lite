#!/usr/bin/env python
"""Post-training processing: update all result tables after a training chain completes.

Runs update_baseline_table, prepare_seed_averaged_results, prepare_cross_dataset_report,
prepare_paper_main_table, and prepare_publication_strategy_report in sequence.

Usage:
    python scripts/process_training_results.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_and_run(script_path: str) -> None:
    path = Path(script_path)
    print(f"\n{'=' * 60}")
    print(f"Running {path.name}")
    print("=" * 60)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if hasattr(mod, "main"):
        mod.main()  # type: ignore[attr-defined]


def main() -> None:
    scripts = [
        "scripts/update_baseline_table.py",
        "scripts/prepare_seed_averaged_results.py",
        "scripts/prepare_cross_dataset_report.py",
        "scripts/prepare_paper_main_table.py",
        "scripts/prepare_publication_strategy_report.py",
    ]
    for s in scripts:
        _load_and_run(s)
    print("\nAll post-training updates complete.")


if __name__ == "__main__":
    main()
