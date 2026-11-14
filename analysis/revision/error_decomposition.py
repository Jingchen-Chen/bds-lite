"""WP1c/f descriptive collapse and 2px contour-band errors, all recovered methods."""

import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]


def load_targets(dataset, split, ids):
    targets = {}
    if split.startswith("official"):
        name = "Validation" if split == "official_validation" else "Test"
        path = (
            ROOT / f"data/revision/isic2018_official/archives/ISIC2018_Task1_{name}_GroundTruth.zip"
        )
        with zipfile.ZipFile(path) as masks:
            names = {
                Path(n).stem.removesuffix("_segmentation"): n
                for n in masks.namelist()
                if n.endswith(".png") and not n.startswith("__MACOSX/")
            }
            for sid in ids:
                image = (
                    Image.open(io.BytesIO(masks.read(names[sid])))
                    .convert("L")
                    .resize((224, 224), Image.Resampling.NEAREST)
                )
                targets[sid] = (np.asarray(image) > 0).astype(np.uint8)
    else:
        for sid in ids:
            with np.load(ROOT / f"data/processed/{dataset}/{split}/{sid}.npz") as z:
                targets[sid] = z["mask"]
    return targets


def main():
    groups = defaultdict(list)
    for name in ["prediction_recovery_queue.json", "main_prediction_queue.json"]:
        for entry in json.loads((ROOT / "revision" / name).read_text())["runs"]:
            folder = ROOT / entry["output"]
            if not (folder / "evaluation_complete.json").exists():
                raise ValueError("required prediction recovery incomplete")
            groups[entry["dataset"], entry["split"]].append(
                dict(
                    run_id=entry["run_id"],
                    family=entry["family"],
                    seed=entry["seed"],
                    path=folder / "predictions.npz",
                    classes=entry["classes"],
                )
            )
    for split in ["validation", "test"]:
        for arm in "USF":
            for seed in [1, 2, 3]:
                groups["isic2018", "official_" + split].append(
                    dict(
                        run_id=f"followup_{arm}_seed{seed}",
                        family=f"followup_{arm}",
                        seed=seed,
                        path=ROOT
                        / f"revision/official_isic/{split}/{arm}_seed{seed}/preds_{split}.npz",
                        classes=2,
                    )
                )
    # Existing development predictions use one .npy per sample, not new markers.
    for arm in "USBF":
        for seed in [1, 2, 3]:
            groups["isic2018", "val"].append(
                dict(
                    run_id=f"followup_{arm}_seed{seed}",
                    family=f"followup_{arm}",
                    seed=seed,
                    path=ROOT
                    / "recovered/followup/results/followup_20261018_v2"
                    / f"isic2018_{arm}_seed{seed}/predictions",
                    classes=2,
                )
            )
    output = ROOT / "revision/error_analysis"
    output.mkdir(exist_ok=True)
    aggregations = {}
    total_rows = 0
    fields = [
        "dataset",
        "split",
        "run_id",
        "family",
        "seed",
        "id",
        "label",
        "area_bin",
        "gt_area",
        "pred_area",
        "dsc",
        "missed_eligible",
        "missed",
        "false_alarm_eligible",
        "false_alarm",
        "fp_near",
        "fp_far",
        "fn_near",
        "fn_far",
        "pred_components",
        "spurious_components",
    ]
    with (output / "per_image_class.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for (dataset, split), entries in groups.items():
            models = []
            ids = None
            for entry in entries:
                if entry["path"].is_dir():
                    predictions = {p.stem: np.load(p) for p in sorted(entry["path"].glob("*.npy"))}
                else:
                    with np.load(entry["path"]) as z:
                        predictions = dict(zip(z["ids"], z["predictions"], strict=True))
                if ids is None:
                    ids = set(predictions)
                if set(predictions) != ids:
                    raise ValueError("cohort IDs differ across methods")
                models.append((entry, predictions))
            targets = load_targets(dataset, split, sorted(ids))
            for sid, target in targets.items():
                for label in range(1, entries[0]["classes"]):
                    gt = target == label
                    area = int(gt.sum())
                    fraction = area / gt.size
                    area_bin = (
                        "empty"
                        if not area
                        else "<1%"
                        if fraction < 0.01
                        else "1-2%"
                        if fraction < 0.02
                        else "2-5%"
                        if fraction < 0.05
                        else ">5%"
                    )
                    inner = gt ^ ndi.binary_erosion(gt)
                    # No GT contour: all false foreground belongs to the no-contour/far bin.
                    near = (
                        ndi.distance_transform_edt(~inner) <= 2
                        if inner.any()
                        else np.zeros_like(gt)
                    )
                    for entry, predictions in models:
                        pred = predictions[sid] == label
                        pred_area = int(pred.sum())
                        fp = pred & ~gt
                        fn = gt & ~pred
                        labels, ncomponents = ndi.label(pred)
                        touching = np.unique(labels[gt])
                        touching = touching[touching > 0]
                        row = dict(
                            dataset=dataset,
                            split=split,
                            run_id=entry["run_id"],
                            family=entry["family"],
                            seed=entry["seed"],
                            id=sid,
                            label=label,
                            area_bin=area_bin,
                            gt_area=area,
                            pred_area=pred_area,
                            dsc=(2 * int((pred & gt).sum()) + 1e-6) / (pred_area + area + 1e-6),
                            missed_eligible=int(area > 0),
                            missed=int(area > 0 and pred_area == 0),
                            false_alarm_eligible=int(area == 0),
                            false_alarm=int(area == 0 and pred_area > 0),
                            fp_near=int((fp & near).sum()),
                            fp_far=int((fp & ~near).sum()),
                            fn_near=int((fn & near).sum()),
                            fn_far=int((fn & ~near).sum()),
                            pred_components=ncomponents,
                            spurious_components=ncomponents - len(touching),
                        )
                        if row["fp_near"] + row["fp_far"] != int(fp.sum()) or row["fn_near"] + row[
                            "fn_far"
                        ] != int(fn.sum()):
                            raise ValueError("error partition is not exhaustive")
                        writer.writerow(row)
                        total_rows += 1
                        key = (
                            dataset,
                            split,
                            entry["run_id"],
                            entry["family"],
                            entry["seed"],
                            label,
                            area_bin,
                        )
                        if key not in aggregations:
                            aggregations[key] = dict(
                                zip(
                                    [
                                        "dataset",
                                        "split",
                                        "run_id",
                                        "family",
                                        "seed",
                                        "label",
                                        "area_bin",
                                    ],
                                    key,
                                    strict=True,
                                ),
                                images=0,
                                dsc_sum=0.0,
                                missed_eligible=0,
                                missed=0,
                                false_alarm_eligible=0,
                                false_alarm=0,
                                fp_near=0,
                                fp_far=0,
                                fn_near=0,
                                fn_far=0,
                                spurious_components=0,
                            )
                        summary = aggregations[key]
                        summary["images"] += 1
                        summary["dsc_sum"] += row["dsc"]
                        for name in [
                            "missed_eligible",
                            "missed",
                            "false_alarm_eligible",
                            "false_alarm",
                            "fp_near",
                            "fp_far",
                            "fn_near",
                            "fn_far",
                            "spurious_components",
                        ]:
                            summary[name] += row[name]
            print(dataset, split, len(ids), "images", len(models), "models", flush=True)
    summaries = []
    for row in aggregations.values():
        row["mean_dsc"] = row.pop("dsc_sum") / row["images"]
        row["missed_structure_rate"] = (
            row["missed"] / row["missed_eligible"] if row["missed_eligible"] else None
        )
        row["false_alarm_rate"] = (
            row["false_alarm"] / row["false_alarm_eligible"]
            if row["false_alarm_eligible"]
            else None
        )
        summaries.append(row)
    with (output / "summary_by_seed_class_area.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    (output / "scope.json").write_text(
        json.dumps(
            dict(
                image_class_rows=total_rows,
                cohorts=len(groups),
                boundary_band="distance to GT inner contour <=2 processed pixels",
                empty_gt_policy="no contour; false foreground counted far, not measured distance",
                connectivity=4,
                inference="descriptive per-seed rates; no seed-image independence claim",
                organ_names="numeric IDs until Synapse mapping verified",
            ),
            indent=2,
        )
        + "\n"
    )
    print("Complete", total_rows, "image-class rows")


if __name__ == "__main__":
    main()
