"""WP1e: manifest-controlled processed-grid descriptors; no model/test selection."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]


def main():
    output = ROOT / "revision/structural_descriptors"
    output.mkdir(exist_ok=True)
    groups = defaultdict(list)
    fields = [
        "dataset",
        "split",
        "id",
        "class",
        "area",
        "area_fraction",
        "components_4connected",
        "inner_perimeter_pixels",
        "morphological_band_pixels",
        "perimeter_area",
        "area_bin",
        "empty",
    ]
    with (output / "per_sample_class.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for dataset, classes in [("isic2018", 2), ("acdc", 4), ("synapse", 9)]:
            # Training masks suffice for dataset characterization; held-out labels untouched.
            manifest = ROOT / f"recovered/followup/splits/{dataset}/train.json"
            ids = json.loads(manifest.read_text())["sample_ids"]
            for sid in ids:
                path = ROOT / f"data/processed/{dataset}/train/{sid}.npz"
                with np.load(path) as z:
                    mask = z["mask"]
                if mask.shape != (224, 224) or not np.isin(mask, np.arange(classes)).all():
                    raise ValueError(f"invalid mask: {path}")
                for label in range(1, classes):
                    m = mask == label
                    area = int(m.sum())
                    perimeter = int((m ^ ndi.binary_erosion(m)).sum())
                    band = int((ndi.binary_dilation(m) ^ ndi.binary_erosion(m)).sum())
                    fraction = area / m.size
                    row = dict(
                        dataset=dataset,
                        split="train",
                        id=sid,
                        **{"class": label},
                        area=area,
                        area_fraction=fraction,
                        components_4connected=int(ndi.label(m)[1]),
                        inner_perimeter_pixels=perimeter,
                        morphological_band_pixels=band,
                        perimeter_area=perimeter / area if area else None,
                        area_bin="empty"
                        if not area
                        else "<1%"
                        if fraction < 0.01
                        else "1-2%"
                        if fraction < 0.02
                        else "2-5%"
                        if fraction < 0.05
                        else ">5%",
                        empty=not area,
                    )
                    writer.writerow(row)
                    groups[dataset, label].append(row)
    summaries = []
    for (dataset, label), rows in groups.items():
        nonempty = [r for r in rows if not r["empty"]]
        la = float(np.mean([r["perimeter_area"] for r in nonempty])) if nonempty else None
        summaries.append(
            dict(
                dataset=dataset,
                split="train",
                **{"class": label},
                images=len(rows),
                empty_class_fraction=float(np.mean([r["empty"] for r in rows])),
                mean_area_fraction=float(np.mean([r["area_fraction"] for r in rows])),
                mean_components_all=float(np.mean([r["components_4connected"] for r in rows])),
                mean_boundary_fraction=float(
                    np.mean([r["morphological_band_pixels"] / (224 * 224) for r in rows])
                ),
                mean_perimeter_area_nonempty=la,
                first_order_dsc_drop_per_1px=la / 2 if la is not None else None,
            )
        )
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    print(
        f"{sum(r['images'] for r in summaries)} image-class rows; {len(summaries)} class summaries"
    )


if __name__ == "__main__":
    main()
