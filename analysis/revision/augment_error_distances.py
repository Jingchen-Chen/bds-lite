"""Join validated saved distance metrics into error bins without repeating inference/EDT."""

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = {}
    for name in ["prediction_recovery_queue.json", "main_prediction_queue.json"]:
        for entry in json.loads((ROOT / "revision" / name).read_text())["runs"]:
            source[entry["dataset"], entry["split"], entry["run_id"]] = (
                ROOT / entry["output"] / "per_case.csv"
            )
    for split in ["validation", "test"]:
        for arm in "USF":
            for seed in [1, 2, 3]:
                source["isic2018", "official_" + split, f"followup_{arm}_seed{seed}"] = (
                    ROOT / f"revision/official_isic/{split}/{arm}_seed{seed}/per_case.csv"
                )
    for arm in "USBF":
        for seed in [1, 2, 3]:
            source["isic2018", "val", f"followup_{arm}_seed{seed}"] = (
                ROOT
                / "recovered/followup/results/followup_20261018_v2"
                / f"isic2018_{arm}_seed{seed}/per_case.csv"
            )
    index = {}
    for key, path in source.items():
        for row in csv.DictReader(path.open()):
            if "label" not in row and key[0] != "isic2018":
                raise ValueError("missing class column in multiclass source")
            label = row.get("label", "1")
            index[(*key, row["id"], label)] = (row["hd95"], row["assd"])
    directory = ROOT / "revision/error_analysis"
    original = directory / "per_image_class.csv"
    temporary = original.with_suffix(".enriched.tmp")
    aggregations = defaultdict(lambda: [0.0, 0.0, 0])
    count = 0
    with original.open() as stream, temporary.open("w") as target:
        reader = csv.DictReader(stream)
        fields = list(dict.fromkeys([*reader.fieldnames, "hd95", "assd", "distance_valid"]))
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            h, a = index[row["dataset"], row["split"], row["run_id"], row["id"], row["label"]]
            valid = h not in ["", "nan"] and a not in ["", "nan"]
            row.update(
                hd95=float(h) if valid else None,
                assd=float(a) if valid else None,
                distance_valid=int(valid),
            )
            writer.writerow(row)
            count += 1
            key = row["dataset"], row["split"], row["run_id"], row["label"], row["area_bin"]
            if valid:
                total = aggregations[key]
                total[0] += float(h)
                total[1] += float(a)
                total[2] += 1
    temporary.replace(original)
    path = directory / "summary_by_seed_class_area.csv"
    summaries = list(csv.DictReader(path.open()))
    for row in summaries:
        total = aggregations[
            row["dataset"], row["split"], row["run_id"], row["label"], row["area_bin"]
        ]
        row.update(
            mean_hd95_valid=total[0] / total[2] if total[2] else None,
            mean_assd_valid=total[1] / total[2] if total[2] else None,
            valid_distance_cases=total[2],
            missing_distance_cases=int(row["images"]) - total[2],
        )
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    scope = json.loads((directory / "scope.json").read_text())
    if count != scope["image_class_rows"]:
        raise ValueError("distance enrichment changed row count")
    scope["distance_stratification"] = (
        "Exact cohort/run/sample/class join; validated distance rows and explicit valid counts"
    )
    (directory / "scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print(
        "Enriched", count, "rows and", len(summaries), "seed/class/area groups; no inference replay"
    )


if __name__ == "__main__":
    main()
