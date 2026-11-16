"""Recompute recovered development overlap metrics and audit split compatibility."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from bds_lite.evaluation.metrics import dice_score, iou_score

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "recovered/followup"
OUT = ROOT / "results/validation"


def dump_csv(path, rows):
    with path.open("w") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    split_rows = []
    for dataset in ["isic2018", "acdc", "synapse"]:
        for split in ["train", "val", "test"]:
            p = F / f"splits/{dataset}/{split}.json"
            if not p.exists():
                continue
            record = json.loads(p.read_text())
            expected = set(record["sample_ids"])
            actual = {p.stem for p in (ROOT / f"data/processed/{dataset}/{split}").glob("*.npz")}
            split_rows.append(
                dict(
                    dataset=dataset,
                    split=split,
                    manifest_n=len(expected),
                    primary_n=len(actual),
                    intersection=len(actual & expected),
                    exact_match=actual == expected,
                    manifest_sha256=hashlib.file_digest(p.open("rb"), "sha256").hexdigest(),
                )
            )
    dump_csv(OUT / "split_compatibility.csv", split_rows)
    rows = []
    val_ids = set(json.loads((F / "splits/isic2018/val.json").read_text())["sample_ids"])
    masks = {}
    for sid in val_ids:
        with np.load(F / f"data/followup/isic2018/val/{sid}.npz") as z:
            masks[sid] = z["mask"]
    for arm in "USBF":
        for seed in [1, 2, 3]:
            run = F / f"results/followup_20261018_v2/isic2018_{arm}_seed{seed}"
            cases = list(csv.DictReader((run / "per_case.csv").open()))
            cfg = json.loads((run / "config.json").read_text())
            state = json.loads((run / "status.json").read_text())
            assert len(cases) == 519 and {r["id"] for r in cases} == val_ids
            assert {p.stem for p in (run / "predictions").glob("*.npy")} == val_ids
            errors = []
            dice = []
            iou = []
            for case in cases:
                pred = np.load(run / f"predictions/{case['id']}.npy")
                assert pred.shape == (224, 224) and pred.dtype == np.uint8
                d = dice_score(pred, masks[case["id"]], 1)
                j = iou_score(pred, masks[case["id"]], 1)
                errors.extend([abs(d - float(case["dsc"])), abs(j - float(case["iou"]))])
                dice.append(d)
                iou.append(j)
            assert max(errors) < 1e-12
            rows.append(
                dict(
                    run=run.name,
                    arm=arm,
                    seed=seed,
                    cases=len(cases),
                    dsc=float(np.mean(dice)),
                    iou=float(np.mean(iou)),
                    max_overlap_error=max(errors),
                    best_pt=(run / "best.pt").exists(),
                    last_pt=(run / "last.pt").exists(),
                    completed=state["state"],
                    epochs=state["epoch"],
                    seconds=state["elapsed_seconds"],
                    saved_batch=cfg["trainer"]["batch_size"],
                    saved_lr=cfg["trainer"]["lr"],
                    result_scope="development; not official test; training not reproduced",
                )
            )
    dump_csv(OUT / "followup_overlap_recomputed.csv", rows)
    print(
        json.dumps(
            {
                "runs_checked": len(rows),
                "case_predictions_checked": sum(r["cases"] for r in rows),
                "max_overlap_error": max(r["max_overlap_error"] for r in rows),
                "split_checks": split_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
