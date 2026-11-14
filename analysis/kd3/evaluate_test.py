"""One-time test evaluation of the new arms of one dataset (protocol amendments; see below).

Refuses unless analysis/kd3/integrity.py passes for the dataset (teacher, cache, all three
D+KD-band runs) and no test output exists. Each selected checkpoint (three students and the
teacher) is evaluated once with the unchanged scripts/revision/train_run.py::evaluate, on the same
test loader the existing arms used: ISIC2018 official test (batch 8, no workers, as train_run.py);
ACDC/Synapse the registered test manifest (ManifestDataset, the config's batch size and workers).
Writes preds_test.npz, per_case_test.csv and test_metrics.json beside each run.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from analysis.kd3.integrity import RUNS, check, names  # noqa: E402
from bds_lite.training import build_model  # noqa: E402
from scripts.kd.common import GPU_LOCK, sha, write_json  # noqa: E402
from scripts.revision.train_run import OfficialTestDataset, evaluate  # noqa: E402
from scripts.revision.training_support import ManifestDataset  # noqa: E402


def test_loader(config: dict) -> DataLoader:
    if config["dataset"]["name"] == "isic2018":
        return DataLoader(OfficialTestDataset(), batch_size=8, shuffle=False, num_workers=0)
    record = config["revision"]["manifests"]["test"]
    if sha(ROOT / record["path"]) != record["sha256"]:
        raise ValueError("test manifest changed")
    manifest = json.loads((ROOT / record["path"]).read_text())
    data = ManifestDataset(
        ROOT / config["dataset"]["processed_dir"],
        "test",
        manifest,
        config["dataset"]["num_classes"],
        config["dataset"]["in_channels"],
    )
    return DataLoader(
        data,
        batch_size=config["trainer"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )


def load_model(folder: Path, config: dict):
    payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    if config["kd"]["role"] == "teacher":
        if config["dataset"]["name"] == "isic2018":
            from scripts.kd.teacher import TeacherNet

            model = TeacherNet(pretrained=False)
        else:
            from scripts.kd_mc.teacher import build_teacher

            model = build_teacher(config, pretrained=False)
    else:
        model = build_model(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.cuda().eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["isic2018", "acdc", "synapse"])
    args = parser.parse_args()
    result = check(args.dataset)
    if not result["passed"]:
        raise RuntimeError(f"integrity checks failed: {result['problems']}")
    n = names(args.dataset)
    folders = [RUNS / s for s in n["students"]] + [RUNS / n["teacher"]]
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for folder in folders:
            config = json.loads((folder / "config.json").read_text())
            model = load_model(folder, config)
            metrics = evaluate(
                model, test_loader(config), config["dataset"]["num_classes"], folder, "test"
            )
            write_json(
                folder / "test_metrics.json",
                dict(
                    best_sha256=sha(folder / "best.pt"),
                    metrics=metrics,
                    artifacts={n: sha(folder / n) for n in ["preds_test.npz", "per_case_test.csv"]},
                ),
            )
            print(folder.name, "evaluated once", flush=True)


if __name__ == "__main__":
    main()
