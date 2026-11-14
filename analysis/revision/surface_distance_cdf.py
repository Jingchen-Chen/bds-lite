"""Descriptive complete surface-distance CDFs from saved test masks; no inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from analysis.revision.error_decomposition import load_targets
from bds_lite.evaluation.metrics import surface, surface_distances

ROOT = Path(__file__).resolve().parents[2]
MAX_SQUARED = 2 * 223**2
THRESHOLDS = [0, 1, 2, 3, 5, 10, 20, 50, 100]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def default_entries():
    entries = []
    for queue in ["main_prediction_queue.json", "prediction_recovery_queue.json"]:
        for e in json.loads((ROOT / "revision" / queue).read_text())["runs"]:
            if e["split"] not in ["test", "official_test"]:
                continue
            entries.append(
                dict(
                    dataset=e["dataset"],
                    split=e["split"],
                    family=e["family"],
                    seed=e["seed"],
                    classes=e["classes"],
                    path=e["output"] + "/predictions.npz",
                )
            )
    for family in "USF":
        for seed in [1, 2, 3]:
            entries.append(
                dict(
                    dataset="isic2018",
                    split="official_test",
                    family=family,
                    seed=seed,
                    classes=2,
                    path=f"revision/official_isic/test/{family}_seed{seed}/preds_test.npz",
                )
            )
    return entries


def calculate(entry, targets, writer):
    path = ROOT / entry["path"]
    with np.load(path) as z:
        ids, predictions = z["ids"].astype(str), z["predictions"]
    if len(set(ids)) != len(ids) or set(ids) != set(targets):
        raise ValueError("prediction and target identity mismatch")
    counts = np.zeros(MAX_SQUARED + 1, dtype=np.int64)
    balanced = np.zeros(MAX_SQUARED + 1, dtype=np.float64)
    valid, both_empty, one_empty, tail_cases, spurious_tail, all_tail = 0, 0, 0, 0, 0, 0
    for sid, prediction in zip(ids, predictions, strict=True):
        for label in range(1, entry["classes"]):
            p, g = prediction == label, targets[sid] == label
            distances = surface_distances(p, g)
            row = dict(
                dataset=entry["dataset"],
                family=entry["family"],
                seed=entry["seed"],
                id=sid,
                label=label,
                empty_prediction=not p.any(),
                empty_target=not g.any(),
                valid=bool(len(distances)),
                q50=None,
                q95=None,
                q99=None,
                max=None,
                fraction_gt20=None,
                spurious_components=0,
                spurious_pred_surface_gt20=0,
                all_pred_surface_gt20=0,
            )
            if not len(distances):
                both_empty += int(not p.any() and not g.any())
                one_empty += int(p.any() != g.any())
            else:
                valid += 1
                # Exact squared distances on a unit pixel lattice are integers.
                squared = distances.astype(np.float64) ** 2
                bins = np.rint(squared).astype(int)
                if np.max(np.abs(squared - bins)) >= 0.05 or bins.max() > MAX_SQUARED:
                    raise AssertionError("distance histogram is not the expected224 pixel grid")
                histogram = np.bincount(bins, minlength=MAX_SQUARED + 1)
                counts += histogram
                balanced += histogram / len(distances)
                q50, q95, q99 = map(float, np.quantile(distances, [0.5, 0.95, 0.99]))
                row.update(
                    q50=q50,
                    q95=q95,
                    q99=q99,
                    max=float(distances.max()),
                    fraction_gt20=float(np.mean(distances > 20)),
                )
                tail_cases += int(q95 > 20)
                labels, n = ndi.label(p)
                touching = set(np.unique(labels[g])) - {0}
                spurious_ids = set(range(1, n + 1)) - touching
                ps = surface(p)
                remote = ps & (ndi.distance_transform_edt(~surface(g)) > 20)
                spurious = np.isin(labels, list(spurious_ids))
                row.update(
                    spurious_components=len(spurious_ids),
                    spurious_pred_surface_gt20=int((remote & spurious).sum()),
                    all_pred_surface_gt20=int(remote.sum()),
                )
                spurious_tail += row["spurious_pred_surface_gt20"]
                all_tail += row["all_pred_surface_gt20"]
            writer.writerow(row)
    cumulative = np.cumsum(counts) / counts.sum() if counts.sum() else np.zeros_like(balanced)
    equal_rows = np.cumsum(balanced) / valid if valid else np.zeros_like(balanced)
    summary = dict(
        dataset=entry["dataset"],
        split=entry["split"],
        family=entry["family"],
        seed=entry["seed"],
        valid_image_classes=valid,
        both_empty=both_empty,
        one_empty=one_empty,
        surface_points=int(counts.sum()),
        q95_gt20_rows=tail_cases,
        q95_gt20_rate=tail_cases / valid if valid else None,
        spurious_pred_tail20=spurious_tail,
        all_pred_tail20=all_tail,
        spurious_share_pred_tail20=spurious_tail / all_tail if all_tail else None,
        prediction_source=entry["path"],
        prediction_sha256=sha(path),
    )
    cdf_rows = []
    for squared in np.flatnonzero(counts):
        cdf_rows.append(
            dict(
                dataset=entry["dataset"],
                family=entry["family"],
                seed=entry["seed"],
                distance_squared=int(squared),
                distance_px=float(np.sqrt(squared)),
                points=int(counts[squared]),
                pooled_cdf=float(cumulative[squared]),
                equal_image_class_cdf=float(equal_rows[squared]),
            )
        )
    threshold_rows = [
        dict(
            dataset=entry["dataset"],
            family=entry["family"],
            seed=entry["seed"],
            threshold_px=t,
            pooled_cdf=float(cumulative[t * t]),
            equal_image_class_cdf=float(equal_rows[t * t]),
        )
        for t in THRESHOLDS
    ]
    return summary, cdf_rows, threshold_rows, equal_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-list", type=Path, help="JSON list: dataset/split/family/seed/classes/path"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "revision/surface_cdf")
    args = parser.parse_args()
    entries = json.loads(args.run_list.read_text()) if args.run_list else default_entries()
    args.output.mkdir(exist_ok=False)
    cache, summaries, full_cdfs, thresholds = {}, [], [], []
    curves = defaultdict(list)
    fields = [
        "dataset",
        "family",
        "seed",
        "id",
        "label",
        "empty_prediction",
        "empty_target",
        "valid",
        "q50",
        "q95",
        "q99",
        "max",
        "fraction_gt20",
        "spurious_components",
        "spurious_pred_surface_gt20",
        "all_pred_surface_gt20",
    ]
    with (args.output / "per_image_class.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for entry in entries:
            key = entry["dataset"], entry["split"]
            with np.load(ROOT / entry["path"]) as z:
                ids = z["ids"].astype(str)
            if key not in cache:
                cache[key] = load_targets(*key, ids)
            summary, cdf, selected, curve = calculate(entry, cache[key], writer)
            summaries.append(summary)
            full_cdfs.extend(cdf)
            thresholds.extend(selected)
            curves[entry["dataset"], entry["family"]].append(curve)
            print(
                entry["dataset"],
                entry["family"],
                entry["seed"],
                summary["valid_image_classes"],
                "valid,",
                summary["one_empty"],
                "one empty",
                flush=True,
            )
    write_csv(args.output / "summary.csv", summaries)
    write_csv(args.output / "full_cdf.csv", full_cdfs)
    write_csv(args.output / "threshold_cdf.csv", thresholds)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, layout="constrained")
    for ax, dataset in zip(axes, ["isic2018", "acdc", "synapse"], strict=True):
        for (ds, family), values in curves.items():
            if ds != dataset:
                continue
            mean = np.mean(values, axis=0)
            # Plot exact CDF on every observed squared lattice distance up to100px.
            index = np.arange(10001)
            ax.plot(np.sqrt(index), mean[index], label=family)
        ax.set(
            xlim=(0, 100), ylim=(0, 1.01), title=dataset, xlabel="Surface distance (processed px)"
        )
        ax.set_xscale("symlog", linthresh=2)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("CDF (equal valid image/class rows, then seeds)")
    for suffix in ["pdf", "png"]:
        fig.savefig(args.output / f"surface_distance_cdf.{suffix}", dpi=180)
    plt.close(fig)
    (args.output / "scope.json").write_text(
        json.dumps(
            dict(
                description="descriptive saved-test masks; no inference or hypothesis test",
                entries=entries,
                cohorts=len(entries),
                surface="inner erosion XOR, connectivity1",
                weighting="point-pooled and equal valid image/class CDF; plot seed-mean of latter",
                units="processed224 pixel grid, not verified millimetres",
                empties="separate both/one-empty counts; neither assigned a measured distance",
                tail="exploratory fixed20px descriptive threshold, no clinical meaning",
                spurious="4-connected predicted component with no GT intersection",
                interpretation="spurious share is descriptive, not causal HD95 attribution",
                classes="numeric Synapse IDs; legacy GSL and Attention2026 are separate references",
                source_sha256=sha(Path(__file__)),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        len(entries),
        "cohorts completed,",
        sum(r["valid_image_classes"] for r in summaries),
        "valid image-class surfaces",
    )


if __name__ == "__main__":
    main()
