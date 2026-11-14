"""Entry point for the expansion runs (revision/EXPANSION_PROTOCOL.md).

Runs the validated adapters unchanged. The only difference from scripts/revision/train_run.py and
scripts/comparators/train_comparator.py is that training seeds 4-10 are accepted; every other locked
protocol check is applied by the original require_protocol. Neither adapter file is edited, so their
source hashes, and the provenance of all earlier runs, stay valid.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

from scripts.revision.run_queue import require_protocol as locked_protocol

ALLOWED_SEEDS = range(1, 11)


def require_protocol(config: dict) -> None:
    seed = config["project"]["seed"]
    if seed not in ALLOWED_SEEDS:
        raise ValueError("expansion training seed must be 1-10")
    checked = copy.deepcopy(config)
    checked["project"]["seed"] = 1
    locked_protocol(checked)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True, type=Path)
    args, _ = parser.parse_known_args(sys.argv[1:])
    config = yaml.safe_load(args.config.read_text())
    if "comparator" in config:
        from scripts.comparators import train_comparator as adapter
    else:
        from scripts.revision import train_run as adapter
    adapter.require_protocol = require_protocol
    adapter.main()


if __name__ == "__main__":
    main()
