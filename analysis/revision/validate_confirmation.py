# ruff: noqa: E501
"""Validation of the 70 Experiment C runs. Prints pass or fail only, never a test metric.

Per run, the checks of validate_expansion.check_run: 150 contiguous epochs; config, artifact and source
hashes; seed; selected epoch equals the first validation maximum; the stored checkpoint reproduces
it; finite training loss; expected per-case rows. Across runs: the registered 70 are all present and
no two seeds of an arm share a trajectory. For one fresh run per dataset and arm, DSC and IoU for
every image and label, and HD95 and ASSD on 40 image-label pairs, are recomputed from the saved
prediction maps with code independent of the training evaluator.

Usage: validate_confirmation.py [--partial] [--self-test RUN_DIR DATASET]
--partial checks finished runs only and writes nothing, so it can run during training.
--self-test exercises the recomputation on an already reported run.
Output: revision/validation/confirmation_validation.json. Exit code 1 on any failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from analysis.revision.validate_expansion import check_run, surface  # noqa: E402

EXPECTED = 70
SURFACE_PAIRS = 40


def reference(dataset: str, sample: str, cohort) -> np.ndarray:
    if dataset == "isic2018":
        return (np.asarray(cohort.get(sample)[1]) > 0).astype(np.uint8)
    with np.load(ROOT / f"data/processed/{dataset}/test/{sample}.npz") as stored:
        return stored["mask"].astype(np.uint8)


def recompute(run: Path, dataset: str) -> list[str]:
    cohort = None
    if dataset == "isic2018":
        from analysis.revision.recover_predictions import Cohort

        cohort = Cohort(dict(dataset="isic2018", split="official_test"))
    saved = np.load(run / "preds_test.npz", allow_pickle=True)
    rows = defaultdict(dict)
    with (run / "per_case_test.csv").open() as stream:
        for row in csv.DictReader(stream):
            rows[row["id"]][int(row.get("label") or 1)] = row
    problems: list[str] = []
    surfaces = 0
    checked = 0
    for sample, prediction in zip(saved["ids"], saved["predictions"], strict=True):
        sample = str(sample)
        mask = reference(dataset, sample, cohort)
        for label, row in rows[sample].items():
            p, g = prediction == label, mask == label
            if dataset == "isic2018":
                p, g = prediction > 0, mask > 0
            inter = float((p & g).sum())
            dsc = (2 * inter + 1e-6) / (p.sum() + g.sum() + 1e-6)
            iou = (inter + 1e-6) / ((p | g).sum() + 1e-6)
            checked += 1
            if abs(dsc - float(row["dsc"])) > 1e-7 or abs(iou - float(row["iou"])) > 1e-7:
                problems.append(f"overlap mismatch {sample}/{label}")
            if surfaces < SURFACE_PAIRS and p.any() and g.any():
                surfaces += 1
                sp, sg = surface(p), surface(g)
                distances = np.r_[
                    ndi.distance_transform_edt(~sg)[sp], ndi.distance_transform_edt(~sp)[sg]
                ]
                if abs(np.percentile(distances, 95) - float(row["hd95"])) > 1e-2:
                    problems.append(f"HD95 mismatch {sample}/{label}")
                if abs(distances.mean() - float(row["assd"])) > 1e-3:
                    problems.append(f"ASSD mismatch {sample}/{label}")
    if surfaces < SURFACE_PAIRS or checked == 0:
        problems.append(f"too few pairs: {checked} overlap, {surfaces} surface")
    return problems[:5]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--self-test", nargs=2, metavar=("RUN_DIR", "DATASET"))
    args = parser.parse_args()
    if args.self_test:
        problems = recompute(ROOT / args.self_test[0], args.self_test[1])
        print("PASSED" if not problems else f"FAILED {problems}")
        return 1 if problems else 0

    queue = yaml.safe_load((ROOT / "revision/queue_confirmation.yaml").read_text())
    runs = queue["runs"]
    report: dict = dict(registered=len(runs), finished=0, failures={})
    if len(runs) != EXPECTED:
        report["failures"]["queue"] = [f"{len(runs)} runs registered, {EXPECTED} expected"]
    meta: dict = {}
    for entry in runs:
        finished = (ROOT / entry["output"] / "final_metrics.json").exists()
        if not finished and args.partial:
            continue
        problems, info = check_run(entry)
        report["finished"] += int(finished)
        if problems:
            report["failures"][entry["run_id"]] = problems
        meta[entry["run_id"]] = info
    groups: dict[tuple, list] = defaultdict(list)
    for info in meta.values():
        if info:
            groups[(info["dataset"], info["arm"])].append(tuple(info["first"]))
    report["duplicate_trajectories"] = [str(k) for k, v in groups.items() if len(set(v)) != len(v)]
    samples: dict = {}
    if not args.partial:
        chosen: dict[tuple, dict] = {}
        for entry in runs:
            info = meta.get(entry["run_id"])
            if info:
                chosen.setdefault((info["dataset"], info["arm"]), entry)
        for (dataset, _), entry in sorted(chosen.items()):
            samples[entry["run_id"]] = recompute(ROOT / entry["output"], dataset)
        report["independent_recomputation"] = samples
    failed = bool(
        report["failures"]
        or report["duplicate_trajectories"]
        or any(samples.values())
        or (not args.partial and len(samples) != 10)
    )
    report["status"] = "FAILED" if failed else "PASSED"
    if not args.partial:
        (ROOT / "revision/validation/confirmation_validation.json").write_text(
            json.dumps(report, indent=1) + "\n"
        )
    print(json.dumps(report, indent=1)[:2500])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
