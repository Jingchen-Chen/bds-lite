"""Protocol step 8 (and 9 for D+KD-full): official-test evaluation of the new arms. NOT for Phase 1.

Refuses unless every D+KD-band run, seeds 1-10, is complete (150 epochs, artifact hashes, no test
output), and, with --phase2b, every D+KD-full run as well. Each selected checkpoint is evaluated
once with the unchanged scripts/revision/train_run.py::evaluate (same evaluator as all existing
arms), writing preds_test.npz, per_case_test.csv and test_metrics.json beside the run. The teacher
is evaluated too (protocol 9.8). Existing outputs are never overwritten.
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
from bds_lite.training import build_model  # noqa: E402
from scripts.kd.common import GPU_LOCK, RUNS, sha, write_json  # noqa: E402
from scripts.revision.train_run import OfficialTestDataset, evaluate  # noqa: E402

BAND = [RUNS / f"isic2018_kd_band_seed{s}" for s in range(1, 11)]
FULL = [RUNS / f"isic2018_kd_full_seed{s}" for s in range(1, 11)]
TEACHER = RUNS / "isic2018_kd_teacher_seed100"


def verified(folder: Path) -> dict:
    record = json.loads((folder / "final_metrics.json").read_text())
    if record["epochs"] != 150 or record["test_evaluated"] is not False:
        raise ValueError(f"incomplete or already evaluated: {folder.name}")
    for name, digest in record["artifacts"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"artifact hash mismatch: {folder.name}/{name}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2b", action="store_true")
    args = parser.parse_args()
    folders = FULL if args.phase2b else [*BAND, TEACHER]
    for folder in [*BAND, *(FULL if args.phase2b else [])]:
        verified(folder)
    test = OfficialTestDataset()
    loader = DataLoader(test, batch_size=8, shuffle=False, num_workers=0)
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for folder in folders:
            if (folder / "test_metrics.json").exists() or (folder / "per_case_test.csv").exists():
                raise FileExistsError(f"test outputs exist: {folder.name}")
            payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
            config = json.loads((folder / "config.json").read_text())
            if config["kd"]["role"] == "teacher":
                from scripts.kd.teacher import TeacherNet

                model = TeacherNet(pretrained=False)
            else:
                model = build_model(config)
            model.load_state_dict(payload["model"], strict=True)
            model = model.cuda().eval()
            metrics = evaluate(model, loader, config["dataset"]["num_classes"], folder, "test")
            write_json(
                folder / "test_metrics.json",
                dict(
                    best_sha256=sha(folder / "best.pt"),
                    metrics=metrics,
                    artifacts={n: sha(folder / n) for n in ["preds_test.npz", "per_case_test.csv"]},
                ),
            )
            print(folder.name, "evaluated", flush=True)


if __name__ == "__main__":
    main()
