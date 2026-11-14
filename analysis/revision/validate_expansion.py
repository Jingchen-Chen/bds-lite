# ruff: noqa: E501
"""Section 10 checks of revision/EXPANSION_PROTOCOL.md. Prints pass or fail only, never a test mean.

Per run: 150 contiguous epochs; hashes; registered config and sources; seed; selected epoch equals
the first maximum of the validation curve; the stored checkpoint is the selected one; no nonfinite
training loss; expected per-case row counts. Across runs: distinct trajectories across seeds.
For one run per arm: every per-image DSC and IoU, and HD95 and ASSD on 40 images, recomputed from the
saved prediction maps with code independent of the training evaluator.
Output: revision/validation/expansion_validation.json. Exit code 1 on any failure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
ROWS = {"isic2018": 1000, "acdc": 366 * 3, "synapse": 1568 * 8}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_run(entry: dict) -> tuple[list[str], dict]:
    run = ROOT / entry["output"]
    problems: list[str] = []
    final_path = run / "final_metrics.json"
    if not final_path.exists():
        return ["missing final_metrics.json"], {}
    final = json.loads(final_path.read_text())
    config_path = ROOT / entry["config"]
    if (
        sha(config_path) != entry["config_sha256"]
        or final["config_sha256"] != entry["config_sha256"]
    ):
        problems.append("config hash mismatch")
    config = yaml.safe_load(config_path.read_text())
    seed = int(re.search(r"seed(\d+)$", entry["run_id"]).group(1))
    if config["project"]["seed"] != seed:
        problems.append("seed mismatch")
    for name, checksum in final["artifacts"].items():
        if sha(run / name) != checksum:
            problems.append(f"artifact hash mismatch: {name}")
    history = json.loads((run / "history.json").read_text())
    epochs = [h["epoch"] for h in history]
    if epochs != list(range(1, 151)):
        problems.append("epochs not contiguous 1-150")
    curve = np.array([h["val"]["val_dsc"] for h in history])
    selected = int(np.argmax(curve)) + 1
    if selected != final["selected_epoch"]:
        problems.append("selected epoch is not the first validation maximum")
    if not math.isclose(curve.max(), final["best_val_dsc"], abs_tol=1e-12):
        problems.append("best validation DSC differs from curve")
    if abs(final["best_val_dsc"] - final["metrics"]["val"]["dsc"]["mean"]) > 1e-6:
        problems.append("stored checkpoint does not reproduce the selected validation DSC")
    if not all(math.isfinite(h["train"].get("total", 0.0)) for h in history):
        problems.append("nonfinite training loss")
    provenance = json.loads((run / "provenance.json").read_text())
    registered = {s["path"]: s["sha256"] for s in entry["sources"]}
    for path, checksum in provenance["source_hashes"].items():
        if path in registered and registered[path] != checksum:
            problems.append(f"source differs from registration: {path}")
    dataset = config["dataset"]["name"]
    with (run / "per_case_test.csv").open() as stream:
        count = sum(1 for _ in csv.DictReader(stream))
    if count != ROWS[dataset]:
        problems.append(f"per-case rows {count} != {ROWS[dataset]}")
    return problems, dict(
        arm=config["revision"]["arm"], dataset=dataset, seed=seed, first=curve[:3].tolist()
    )


def surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask)


def recompute(entry: dict, dataset: str) -> list[str]:
    """Independent DSC, IoU on every image and HD95, ASSD on 40 images, for ISIC2018 runs."""
    from analysis.revision.recover_predictions import Cohort

    run = ROOT / entry["output"]
    saved = np.load(run / "preds_test.npz", allow_pickle=True)
    rows = {r["id"]: r for r in csv.DictReader((run / "per_case_test.csv").open())}
    cohort = Cohort(dict(dataset="isic2018", split="official_test"))
    problems = []
    for index, (sample, pred) in enumerate(zip(saved["ids"], saved["predictions"], strict=True)):
        _, mask, _ = cohort.get(str(sample))
        p, g = pred > 0, np.asarray(mask) > 0
        inter = float((p & g).sum())
        dsc = (2 * inter + 1e-6) / (p.sum() + g.sum() + 1e-6)
        iou = (inter + 1e-6) / ((p | g).sum() + 1e-6)
        row = rows[str(sample)]
        if abs(dsc - float(row["dsc"])) > 1e-9 or abs(iou - float(row["iou"])) > 1e-9:
            problems.append(f"overlap mismatch {sample}")
        if index < 40 and p.any() and g.any():
            sp, sg = surface(p), surface(g)
            distances = np.r_[
                ndi.distance_transform_edt(~sg)[sp], ndi.distance_transform_edt(~sp)[sg]
            ]
            if abs(np.percentile(distances, 95) - float(row["hd95"])) > 1e-2:
                problems.append(f"HD95 mismatch {sample}")
            if abs(distances.mean() - float(row["assd"])) > 1e-3:
                problems.append(f"ASSD mismatch {sample}")
    return problems[:5]


def main() -> int:
    queue = yaml.safe_load((ROOT / "revision/queue_expansion.yaml").read_text())
    report: dict = dict(runs=len(queue["runs"]), failures={})
    meta = {}
    for entry in queue["runs"]:
        problems, info = check_run(entry)
        if problems:
            report["failures"][entry["run_id"]] = problems
        meta[entry["run_id"]] = info
    groups: dict[tuple, list] = defaultdict(list)
    for info in meta.values():
        if info:
            groups[(info["dataset"], info["arm"])].append(tuple(info["first"]))
    report["duplicate_trajectories"] = [str(k) for k, v in groups.items() if len(set(v)) != len(v)]
    samples = {}
    for arm in ("m0_unet_seed4", "m2_aux_boundary_only_seed4", "m6_full_seed4", "gsl_seed4"):
        entry = next(e for e in queue["runs"] if e["run_id"] == f"isic2018_{arm}")
        if (ROOT / entry["output"] / "final_metrics.json").exists():
            samples[entry["run_id"]] = recompute(entry, "isic2018")
    report["independent_recomputation"] = samples
    failed = bool(
        report["failures"]
        or report["duplicate_trajectories"]
        or any(samples.values())
        or len(samples) < 4
    )
    report["status"] = "FAILED" if failed else "PASSED"
    (ROOT / "revision/validation/expansion_validation.json").write_text(
        json.dumps(report, indent=1) + "\n"
    )
    print(json.dumps(report, indent=1)[:2500])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
