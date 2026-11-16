"""Detect reused tensor states across path/config aliases without deleting artifacts."""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = []
    groups = defaultdict(list)
    for record in csv.DictReader((ROOT / "results/audit/checkpoints.csv").open()):
        path = ROOT / record["path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("state_dict"))
        if state is None:
            raise ValueError(f"unknown checkpoint state schema: {path}")
        digest = hashlib.sha256()
        for key, value in sorted(state.items()):
            digest.update(key.encode())
            digest.update(str(value.shape).encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.contiguous().numpy().tobytes())
        fingerprint = digest.hexdigest()
        groups[fingerprint].append(record["path"])
        rows.append(
            dict(
                path=record["path"],
                model=record["model"],
                dataset=record["dataset"],
                seed=record["seed"],
                epoch=record["epoch"],
                state_sha256=fingerprint,
                tensor_keys=len(state),
                gate_keys=sum("gate" in key for key in state),
            )
        )
    with (ROOT / "results/validation/checkpoint_states.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / "results/validation/checkpoint_state_duplicates.csv").open("w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["state_sha256", "path", "group_size"])
        for fingerprint, paths in groups.items():
            if len(paths) > 1:
                for path in paths:
                    writer.writerow([fingerprint, path, len(paths)])
    print(
        len(rows),
        "checkpoint paths;",
        len(groups),
        "distinct state dictionaries;",
        sum(len(paths) for paths in groups.values() if len(paths) > 1),
        "paths in duplicate groups",
    )


if __name__ == "__main__":
    main()
