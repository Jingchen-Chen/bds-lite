"""One-time test evaluation (pre-registration section 7). Refuses a second invocation.

Requires integrity.py to pass. Uses the unchanged scripts/revision/train_run.py::evaluate with the
frozen ManifestDataset loader (batch 16, 4 workers, no shuffle), writing only under
results/acdc_confirm_eval/<run>/ (parent-study run folders are not touched):
- official_test (if PRIMARY_SET == "official_test"): the 34 replication runs and the 6
  exploratory runs, on patients 101-150 (NPZ hashes checked against the preprocessing record);
- internal_test: the 34 replication runs, on the original 20-patient split (already seen).

    .venv/bin/python scripts/acdc_confirm/evaluate_test.py
"""

from __future__ import annotations

import fcntl
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from bds_lite.training import build_model  # noqa: E402
from scripts.acdc_confirm.common import (  # noqa: E402
    EVAL,
    EXPLORATORY,
    OFFICIAL_MANIFEST,
    OFFICIAL_PREPROCESSING,
    OFFICIAL_PROCESSED,
    PRIMARY_SET,
    RUNS,
    failed_runs,
    registered_runs,
    run_name,
)
from scripts.acdc_confirm.integrity import check  # noqa: E402
from scripts.kd.common import GPU_LOCK, sha, write_json  # noqa: E402
from scripts.revision.train_run import evaluate  # noqa: E402
from scripts.revision.training_support import ManifestDataset  # noqa: E402

RECORD = EVAL / "evaluation_record.json"


def loader(root: Path, manifest: dict) -> DataLoader:
    data = ManifestDataset(root, "test", manifest, 4, 1)
    return DataLoader(data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)


def internal_loader(config: dict) -> DataLoader:
    record = config["revision"]["manifests"]["test"]
    if sha(ROOT / record["path"]) != record["sha256"]:
        raise ValueError("internal test manifest changed")
    manifest = json.loads((ROOT / record["path"]).read_text())
    return loader(ROOT / config["dataset"]["processed_dir"], manifest)


def official_loader() -> DataLoader:
    record = json.loads(OFFICIAL_PREPROCESSING.read_text())
    if sha(OFFICIAL_MANIFEST) != record["manifest_file_sha256"]:
        raise ValueError("official test manifest changed")
    for sample, digest in record["npz_sha256"].items():
        if sha(OFFICIAL_PROCESSED / "test" / f"{sample}.npz") != digest:
            raise ValueError(f"official test NPZ changed: {sample}")
    return loader(OFFICIAL_PROCESSED, json.loads(OFFICIAL_MANIFEST.read_text()))


def load(folder: Path, config: dict):
    model = build_model(config)
    model.load_state_dict(
        torch.load(folder / "best.pt", map_location="cpu", weights_only=False)["model"],
        strict=True,
    )
    return model.cuda().eval()


def main() -> None:
    if EVAL.exists():
        raise FileExistsError(f"{EVAL} exists: the test evaluation runs exactly once")
    result = check()
    if not result["passed"]:
        raise RuntimeError(f"integrity checks failed: {result['problems']}")
    jobs = []  # (run name, run folder, split)
    failed = failed_runs()
    new = [run_name(a, s) for a, s in registered_runs() if run_name(a, s) not in failed]
    if PRIMARY_SET == "official_test":
        exploratory = [n for arm in ("decoder", "kd_band") for n in EXPLORATORY[arm]]
        for name in exploratory:
            digest = {**EXPLORATORY["decoder"], **EXPLORATORY["kd_band"]}[name]
            if sha(RUNS / name / "best.pt") != digest:
                raise ValueError(f"exploratory checkpoint changed: {name}")
        jobs += [(n, "official_test") for n in new + exploratory]
    elif PRIMARY_SET != "internal_test":
        raise ValueError(PRIMARY_SET)
    jobs += [(n, "internal_test") for n in new]
    EVAL.mkdir(parents=True)
    log = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        primary_set=PRIMARY_SET,
        failed_runs_not_evaluated=sorted(failed),
        runs=[],
    )
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        official = official_loader() if PRIMARY_SET == "official_test" else None
        for name, split in jobs:
            folder = RUNS / name
            config = json.loads((folder / "config.json").read_text())
            out = EVAL / name
            out.mkdir(exist_ok=True)
            data = official if split == "official_test" else internal_loader(config)
            metrics = evaluate(load(folder, config), data, 4, out, split)
            log["runs"].append(
                dict(
                    run=name,
                    split=split,
                    best_sha256=sha(folder / "best.pt"),
                    metrics=metrics,
                    artifacts={
                        f: sha(out / f) for f in [f"preds_{split}.npz", f"per_case_{split}.csv"]
                    },
                )
            )
            print(name, split, "evaluated once", flush=True)
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(RECORD, log)


if __name__ == "__main__":
    main()
