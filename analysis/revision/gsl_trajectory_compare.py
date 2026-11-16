# ruff: noqa: E501
"""Compare the retrained U-Net+GSL trajectories (revision queue_b, locked partitions, revision
adapter, src/bds_lite GSL loss) with the submitted Phase16 matched-GSL trajectories
(recovered/followup/results/gsl/runs/phase16_matched_gsl_<ds>_seed<n>/metrics.jsonl; trained with
scripts/gsl/gsl_memory_bounded.py, checkpoints not available locally, see GITHUB_AUDIT.md U1).

Purpose: establish whether the retrain is (a) a numerical reproduction of Phase16 (epoch-1 val
DSC equal within AMP noise, trajectories overlapping), or (b) an independent realisation of the
same protocol (divergence from epoch 1, as observed for p1b vs family O), and how far the
best-validation values differ. Validation DSC is the training-time aggregate of each pipeline
(same locked validation partition for ACDC/Synapse; ISIC2018 locked 519-image validation).
Config note: Phase16 records data.normalize=zscore and the revision configs none_stored; the key
is not read by src/bds_lite/data (normalisation is applied once at conversion, converters.py),
so both pipelines consume the same stored arrays.

Outputs:
  results/tables/families/gsl_trajectory_comparison.csv  (one row per dataset/seed)
  results/tables/families/gsl_trajectory_epochs.csv      (long format, common epochs)
  results/tables/families/gsl_table2_reproduction.csv     (submitted U-Net+GSL Table-2 rows vs the
      retrained runs' test predictions, original reeval_main_seeds.py aggregation; ISIC2018 rows use
      the 519-image validation set as in the submission)
Runs not yet trained are reported with status=missing; partially trained runs are compared on
the epochs available (status=partial).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE16 = ROOT / "recovered/followup/results/gsl/runs"
REVISION = ROOT / "results/revision_runs"
OUT = ROOT / "results/tables/families"
EPOCHS = 150
REPRODUCTION_TOL = 2e-3  # epoch-1 |dDSC| below this is treated as numerical agreement


def phase16(dataset: str, seed: int) -> dict[int, dict]:
    path = PHASE16 / f"phase16_matched_gsl_{dataset}_seed{seed}/metrics.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {int(r["epoch"]): r for r in rows}


def revision(dataset: str, seed: int) -> tuple[dict[int, dict], bool]:
    run = REVISION / f"{dataset}_gsl_seed{seed}"
    path = run / "history.json"
    if not path.exists():
        return {}, False
    rows = json.loads(path.read_text())
    return {int(r["epoch"]): r for r in rows}, (run / "final_metrics.json").exists()


def table2_value(per_case: Path, metric: str) -> float:
    """Original aggregation (see reproduce_submitted_tables.table2_value): per-class nanmean over
    samples, then mean over foreground classes; both-empty classes score 1 for DSC/IoU/BF1."""
    by_class: dict[str, list[float]] = {}
    for r in csv.DictReader(per_case.open()):
        both_empty = r["empty_prediction"] == "True" and r["empty_target"] == "True"
        if metric in ("dsc", "iou", "boundary_f1"):
            value = 1.0 if both_empty else float(r[metric])
        else:
            value = float(r[metric]) if r[metric] not in ("", "None") else np.nan
        by_class.setdefault(r["label"], []).append(value)
    return float(np.mean([np.nanmean(v) for v in by_class.values()]))


def table2_reproduction() -> list[dict]:
    out = []
    source = ROOT / "docs/provenance/sources/github_release/matched_gsl_summary.csv"
    for r in csv.DictReader(source.open()):
        split = "val" if r["dataset"] == "isic2018" else "test"
        if r["split"] != split:
            continue
        published = [float(x) for x in r["seed_values"].split("|")]
        mine = []
        for seed in (1, 2, 3):
            run = REVISION / f"{r['dataset']}_gsl_seed{seed}"
            per_case = run / f"per_case_{split}.csv"
            ok = (run / "final_metrics.json").exists() and per_case.exists()
            mine.append(table2_value(per_case, r["metric"]) if ok else float("nan"))
        diffs = [abs(a - b) for a, b in zip(published, mine, strict=True) if np.isfinite(b)]
        out.append(
            dict(
                dataset=r["dataset"],
                split=split,
                metric=r["metric"],
                published=";".join(f"{x:.6f}" for x in published),
                reproduced=";".join(f"{x:.6f}" for x in mine),
                seeds_available=len(diffs),
                max_abs_diff=max(diffs) if diffs else float("nan"),
            )
        )
    with (OUT / "gsl_table2_reproduction.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    return out


def main() -> None:
    summary, long_rows = [], []
    for dataset in ("isic2018", "acdc", "synapse"):
        for seed in (1, 2, 3):
            ref = phase16(dataset, seed)
            new, finished = revision(dataset, seed)
            common = sorted(set(ref) & set(new))
            row = dict(
                dataset=dataset, seed=seed, phase16_epochs=len(ref), revision_epochs=len(new)
            )
            if not ref or not new or not common:
                row["status"] = "missing" if not new else "no_common_epochs"
                summary.append(row)
                continue
            a = np.array([ref[e]["val"]["val_dsc"] for e in common])
            b = np.array([new[e]["val"]["val_dsc"] for e in common])
            for e, x, y in zip(common, a, b, strict=True):
                long_rows.append(
                    dict(
                        dataset=dataset,
                        seed=seed,
                        epoch=e,
                        phase16_val_dsc=x,
                        revision_val_dsc=y,
                        diff=y - x,
                    )
                )
            ref_best_epoch = max(ref, key=lambda e: ref[e]["val"]["val_dsc"])
            new_best_epoch = max(new, key=lambda e: new[e]["val"]["val_dsc"])
            d1 = float(b[0] - a[0]) if common[0] == 1 else float("nan")
            row.update(
                status="complete" if finished and len(new) == EPOCHS else "partial",
                epoch1_phase16_val_dsc=float(a[0]),
                epoch1_revision_val_dsc=float(b[0]),
                epoch1_diff=d1,
                epoch1_numerically_equal=bool(abs(d1) < REPRODUCTION_TOL),
                epoch1_train_total_phase16=ref[common[0]]["train"].get("total"),
                epoch1_train_total_revision=new[common[0]]["train"].get("total"),
                mean_abs_diff_common=float(np.mean(np.abs(b - a))),
                max_abs_diff_common=float(np.max(np.abs(b - a))),
                last10_mean_diff=float(np.mean((b - a)[-10:])),
                phase16_best_val_dsc=ref[ref_best_epoch]["val"]["val_dsc"],
                phase16_best_epoch=ref_best_epoch,
                revision_best_val_dsc=new[new_best_epoch]["val"]["val_dsc"],
                revision_best_epoch=new_best_epoch,
                best_diff=new[new_best_epoch]["val"]["val_dsc"]
                - ref[ref_best_epoch]["val"]["val_dsc"],
            )
            summary.append(row)
    fields = list(dict.fromkeys(k for r in summary for k in r))
    with (OUT / "gsl_trajectory_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    with (OUT / "gsl_trajectory_epochs.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["dataset", "seed", "epoch", "phase16_val_dsc", "revision_val_dsc", "diff"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(long_rows)
    for r in table2_reproduction():
        print(r["dataset"], r["metric"], r["seeds_available"], f"{r['max_abs_diff']:.2e}")
    for r in summary:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
