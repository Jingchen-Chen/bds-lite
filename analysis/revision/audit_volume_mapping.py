"""Read-only case/frame completeness and original-shape provenance audit."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = []
    for dataset in ["acdc", "synapse"]:
        for split in ["train", "val", "test"]:
            groups = defaultdict(list)
            for path in (ROOT / f"data/processed/{dataset}/{split}").glob("*.npz"):
                pattern = (
                    r"(patient\d+)_frame(\d+)_slice_?(\d+)"
                    if dataset == "acdc"
                    else r"(case\d+)_slice_?(\d+)"
                )
                match = re.fullmatch(pattern, path.stem)
                if not match:
                    raise ValueError(f"unparsed sample ID: {path}")
                volume = "_".join(match.groups()[:-1])
                index = int(match.groups()[-1])
                groups[volume].append(index)
            for volume, indices in sorted(groups.items()):
                ordered = sorted(indices)
                row = dict(
                    dataset=dataset,
                    split=split,
                    volume=volume,
                    slices=len(ordered),
                    first_index=ordered[0],
                    last_index=ordered[-1],
                    contiguous=len(ordered) == len(set(ordered))
                    and ordered == list(range(ordered[0], ordered[-1] + 1)),
                    original_shape=None,
                    original_geometry="unverified",
                )
                if dataset == "synapse" and split == "test":
                    raw = ROOT / f"data/raw/Synapse/test_vol_h5/{volume}.npy.h5"
                    with h5py.File(raw, "r") as handle:
                        shape = handle["label"].shape
                        row["original_shape"] = "x".join(map(str, shape))
                        row["raw_attrs"] = json.dumps(dict(handle.attrs))
                        row["full_original_z"] = ordered == list(range(shape[0]))
                rows.append(row)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / "revision/validation/volume_mapping.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    official = set((ROOT / "revision/sources/transunet/test_vol.txt").read_text().split())
    local = {
        row["volume"] for row in rows if row["dataset"] == "synapse" and row["split"] == "test"
    }
    acdc = [row for row in rows if row["dataset"] == "acdc" and row["split"] == "test"]
    patients = defaultdict(list)
    for row in acdc:
        patients[row["volume"].split("_")[0]].append(row["volume"])
    result = dict(
        synapse_test_cases=sorted(local),
        official_cases=sorted(official),
        exact_case_match=local == official,
        synapse_full_z=all(row["full_original_z"] for row in rows if "full_original_z" in row),
        acdc_test_patient_count=len(patients),
        acdc_test_frame_count=len(acdc),
        acdc_two_annotated_frames=all(len(v) == 2 for v in patients.values()),
        acdc_ed_es_assignment="not verified without original frame metadata",
        acdc_native_dimensions_and_spacing="not preserved in legacy NPZs",
        no_missing_within_observed_z=all(row["contiguous"] for row in rows),
        physical_units_ready=False,
    )
    (ROOT / "revision/validation/volume_mapping_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
