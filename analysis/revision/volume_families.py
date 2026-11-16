# ruff: noqa: E501
"""Stacked-volume evaluation and statistics for every method in the family registry.

Reuses the frozen evaluator functions of evaluate_volumes.py (metrics, distances, protocol
protocols/volume/volume_evaluation_protocol.md) unchanged; only cohort discovery differs: any
method/seed with a complete per-case CSV in family_statistics.per_case_path and a
prediction archive next to it (submitted models S0/S6, replication R0/R6, new runs).
Outputs results/volume/registry_<method>/<dataset>_seed<s>/per_volume_class.csv.
Statistics (registered conventions, prereg_revision.md): per patient (ACDC, frames averaged)
or case (Synapse), classes averaged over volumes with a defined value, seeds averaged; paired
Wilcoxon, 10,000-draw bootstrap; Holm over {dsc, hd95, assd_pooled} within dataset and family.
Index-grid units (ACDC 224 grid; Synapse original H5 grid); no millimetre claim.
Output: results/tables/families/volume_family_statistics.csv and volume_method_summary.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import wilcoxon

from analysis.revision.evaluate_volumes import metrics, validate_backend
from analysis.revision.family_statistics import METHODS, SEEDS, bootstrap, holm
from analysis.revision.figure_qualitative import prediction_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/volume"
CLASSES = {"acdc": 4, "synapse": 9}
ENDPOINTS = ["dsc", "hd95", "assd_pooled"]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(method: str, dataset: str, seed: int) -> Path | None:
    archive = prediction_file(method, dataset, seed)
    if archive is None:
        return None
    folder = OUT / f"registry_{method}" / f"{dataset}_seed{seed}"
    done = folder / "volume_complete.json"
    if done.exists() and json.loads(done.read_text())["prediction_sha256"] == sha(archive):
        return folder / "per_volume_class.csv"
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(archive) as z:
        predictions = dict(zip([str(i) for i in z["ids"]], z["predictions"], strict=True))
    groups = defaultdict(list)
    pattern = (
        r"(patient\d+)_frame(\d+)_slice_?(\d+)" if dataset == "acdc" else r"(case\d+)_slice_?(\d+)"
    )
    for sid in predictions:
        match = re.fullmatch(pattern, sid)
        groups["_".join(match.groups()[:-1])].append((int(match.groups()[-1]), sid))
    rows = []
    for volume, indices in sorted(groups.items()):
        indices.sort()
        pred = np.stack([predictions[s] for _, s in indices])
        if dataset == "synapse":
            with h5py.File(ROOT / f"data/raw/Synapse/test_vol_h5/{volume}.npy.h5", "r") as handle:
                target = handle["label"][:]
            pred = ndi.zoom(pred, (1, target.shape[1] / 224, target.shape[2] / 224), order=0)
        else:
            target = np.stack(
                [np.load(ROOT / f"data/processed/acdc/test/{s}.npz")["mask"] for _, s in indices]
            )
        for label in range(1, CLASSES[dataset]):
            rows.append(
                dict(
                    method=method,
                    dataset=dataset,
                    seed=seed,
                    volume=volume,
                    unit=volume.split("_")[0],
                    **metrics(pred, target, label),
                )
            )
    with (folder / "per_volume_class.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    done.write_text(
        json.dumps(
            dict(
                prediction=str(archive.relative_to(ROOT)),
                prediction_sha256=sha(archive),
                protocol_sha256=sha(ROOT / "protocols/volume/volume_evaluation_protocol.md"),
            ),
            indent=1,
        )
        + "\n"
    )
    print("volume", method, dataset, seed, len(groups), flush=True)
    return folder / "per_volume_class.csv"


def unit_values(method: str, dataset: str) -> dict[str, dict[str, float]] | None:
    per_seed = []
    for seed in SEEDS:
        path = evaluate(method, dataset, seed)
        if path is None:
            return None
        acc = defaultdict(lambda: defaultdict(list))
        for r in csv.DictReader(path.open()):
            for e in ENDPOINTS:
                if r[e] not in ("", "None") and not (
                    r["empty_prediction"] == "True" and r["empty_target"] == "True"
                ):
                    acc[r["unit"]][e].append(float(r[e]))
        per_seed.append(
            {
                u: {e: float(np.mean(v)) if v else math.nan for e, v in d.items()}
                for u, d in acc.items()
            }
        )
    units = sorted(per_seed[0])
    return {
        u: {e: float(np.nanmean([s[u].get(e, math.nan) for s in per_seed])) for e in ENDPOINTS}
        for u in units
    }


def compare(a: str, b: str, dataset: str) -> list[dict]:
    va, vb = unit_values(a, dataset), unit_values(b, dataset)
    if va is None or vb is None:
        return []
    rows = []
    for e in ENDPOINTS:
        sign = -1.0 if e != "dsc" else 1.0
        d = np.array(
            [
                sign * (va[u][e] - vb[u][e])
                for u in va
                if math.isfinite(va[u][e]) and math.isfinite(vb[u][e])
            ]
        )
        low, high = bootstrap(d)
        p = float(wilcoxon(d).pvalue) if np.any(d) else 1.0
        rows.append(
            dict(
                dataset=dataset,
                method=a,
                reference=b,
                endpoint=e,
                n_units=d.size,
                mean_oriented_delta=float(d.mean()),
                ci95_low=low,
                ci95_high=high,
                p_raw=p,
            )
        )
    holm(rows)
    return rows


def main() -> None:
    (ROOT / "results/validation/volume_distance_backend_registry.json").write_text(
        json.dumps(validate_backend(), indent=1) + "\n"
    )
    out, summary = [], []
    pairs = [("P", "S6", "S0"), ("P-replication", "R6", "R0"), ("P-matched", "M6", "M0")]
    pairs += [("A", m, "M6") for m in ["M1", "M2", "M3", "M4", "M5"]] + [
        ("C", m, "M0") for m in ["GSL", "C1", "C3", "C4"]
    ]
    for dataset in CLASSES:
        for fam, a, b in pairs:
            for r in compare(a, b, dataset):
                r["family"] = fam
                out.append(r)
        for method in METHODS:
            v = unit_values(method, dataset)
            if v:
                summary.append(
                    dict(
                        dataset=dataset,
                        method=method,
                        **{e: float(np.nanmean([x[e] for x in v.values()])) for e in ENDPOINTS},
                    )
                )
    for name, data in [
        ("volume_family_statistics.csv", out),
        ("volume_method_summary.csv", summary),
    ]:
        with (ROOT / "results/tables/families" / name).open("w") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(dict.fromkeys(k for r in data for k in r)),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(data)
    print(len(out), "volume contrasts")


if __name__ == "__main__":
    main()
