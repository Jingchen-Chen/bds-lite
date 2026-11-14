"""Independent sampled E3 prediction-to-mask audit; no models or inference.

Reads completed E3 evaluation artifacts and four deterministic official-test masks.
Writes only revision/evidence_freeze/final_prediction_audit.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("dsc", "iou", "boundary_f1", "hd95", "assd")
TOLERANCE = 1e-10


def independent_metrics(pred: np.ndarray, target: np.ndarray, label: int) -> dict:
    p, q = pred == label, target == label
    intersection = np.count_nonzero(p & q)
    result = {
        "dsc": (2 * intersection + 1e-6) / (p.sum() + q.sum() + 1e-6),
        "iou": (intersection + 1e-6) / (np.count_nonzero(p | q) + 1e-6),
    }
    bp = ndi.binary_dilation(p) ^ ndi.binary_erosion(p)
    bq = ndi.binary_dilation(q) ^ ndi.binary_erosion(q)
    if np.array_equal(bp, bq):
        bf = 1.0
    elif not bp.any() or not bq.any():
        bf = 0.0
    else:
        precision = np.count_nonzero(bp & ndi.binary_dilation(bq, iterations=2)) / (bp.sum() + 1e-6)
        recall = np.count_nonzero(bq & ndi.binary_dilation(bp, iterations=2)) / (bq.sum() + 1e-6)
        bf = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result["boundary_f1"] = bf
    sp, sq = p ^ ndi.binary_erosion(p), q ^ ndi.binary_erosion(q)
    if not sp.any() or not sq.any():
        result.update(hd95=None, assd=None)
    else:
        distances = np.concatenate(
            [ndi.distance_transform_edt(~sq)[sp], ndi.distance_transform_edt(~sp)[sq]]
        ).astype(np.float32)
        result.update(hd95=float(np.percentile(distances, 95)), assd=float(distances.mean()))
    result.update(empty_prediction=not p.any(), empty_target=not q.any(), target_area=int(q.sum()))
    return result


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    record_path = ROOT / "results/followup_eval/evaluation_record_e3.json"
    record = json.loads(record_path.read_text())
    manifest_path = ROOT / "revision/acdc_confirm/official_test_manifest.json"
    official_ids = sorted(json.loads(manifest_path.read_text())["sample_ids"])
    wanted = {
        f"acdc_followup_{arm}_seed{seed}"
        for arm in ("decoder_aug", "kd_band_aug")
        for seed in range(201, 211)
    }
    if {r["run"] for r in record["runs"]} != wanted or len(record["runs"]) != 20:
        raise ValueError("unexpected completed E3 run set")
    sources = {record_path, manifest_path}
    errors = dict.fromkeys(METRICS, 0.0)
    undefined = dict.fromkeys(METRICS, 0)
    audited = []
    for entry in record["runs"]:
        folder = ROOT / "results/followup_eval" / entry["run"]
        csv_path, prediction_path = (
            folder / "per_case_official_test.csv",
            folder / "preds_official_test.npz",
        )
        sources.update((csv_path, prediction_path))
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        lookup = {(r["id"], int(r["label"])): r for r in rows}
        if len(lookup) != len(rows) or set(lookup) != {
            (sample, label) for sample in official_ids for label in (1, 2, 3)
        }:
            raise ValueError(f"case/label identity mismatch: {entry['run']}")
        with np.load(prediction_path, allow_pickle=False) as saved:
            ids = [str(x) for x in saved["ids"]]
            predictions = saved["predictions"]
        if ids != official_ids or predictions.shape != (1076, 224, 224):
            raise ValueError(f"prediction identity/shape mismatch: {entry['run']}")
        if predictions.dtype != np.uint8 or predictions.max() > 3:
            raise ValueError(f"prediction dtype/label mismatch: {entry['run']}")
        selected = np.linspace(0, 1075, 4, dtype=int)
        for index in selected:
            sample = ids[index]
            mask_path = ROOT / f"data/processed/acdc_official_test/test/{sample}.npz"
            sources.add(mask_path)
            with np.load(mask_path, allow_pickle=False) as data:
                mask = data["mask"]
            if mask.shape != (224, 224) or mask.min() < 0 or mask.max() > 3:
                raise ValueError(f"reference shape/labels invalid: {sample}")
            for label in (1, 2, 3):
                got = independent_metrics(predictions[index], mask, label)
                row = lookup[sample, label]
                for flag in ("empty_prediction", "empty_target"):
                    if bool(got[flag]) != (row[flag] == "True"):
                        raise ValueError(f"emptiness mismatch: {entry['run']} {sample} {label}")
                if got["target_area"] != int(row["target_area"]):
                    raise ValueError(f"target area mismatch: {sample} {label}")
                for metric in METRICS:
                    value = got[metric]
                    recorded = row[metric]
                    if value is None:
                        if recorded not in ("", "nan", "None"):
                            raise ValueError(f"undefined metric mismatch: {sample} {metric}")
                        undefined[metric] += 1
                    else:
                        error = abs(value - float(recorded))
                        if not np.isfinite(error) or error > TOLERANCE:
                            raise ValueError(f"metric mismatch: {sample} {metric}: {error}")
                        errors[metric] = max(errors[metric], error)
        audited.append({"run": entry["run"], "sample_indices": selected.tolist()})
    result = {
        "status": "passed",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent NumPy/SciPy recalculation; no model execution or inference.",
        "runs_checked_for_identity_shape_dtype_labels": len(audited),
        "sampled_slices_per_run": 4,
        "sampled_class_rows": 240,
        "metric_comparisons": 1200,
        "tolerance": TOLERANCE,
        "maximum_absolute_error": errors,
        "undefined_agreements": undefined,
        "runs": audited,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)},
        "limits": "Samples four slices per run; not exhaustive recalculation of all E3 pixels.",
    }
    path = ROOT / "revision/evidence_freeze/final_prediction_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("runs", "source_sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
