from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

DATA_MODULES = (
    "bds_lite.data",
    "bds_lite.data.boundary",
    "bds_lite.data.mask_to_sdf",
    "bds_lite.data.converters",
    "bds_lite.data.datasets",
    "bds_lite.data.schema",
)


def test_data_modules_import_in_place() -> None:
    for module_name in DATA_MODULES:
        module = importlib.import_module(module_name)
        assert module is not None


def test_data_modules_import_from_fresh_install(tmp_path: Path) -> None:
    install_dir = tmp_path / "install"
    repo_root = Path(__file__).resolve().parents[2]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(install_dir),
            str(repo_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    code = "\n".join(
        [
            "import importlib",
            f"modules = {DATA_MODULES!r}",
            "for name in modules:",
            "    importlib.import_module(name)",
        ]
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(install_dir)
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
