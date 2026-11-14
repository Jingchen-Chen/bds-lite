"""Preprocessing impact check on the internal VALIDATION patients only (no test data).

The 6 exploratory ACDC models are evaluated, with the unchanged train_run.py::evaluate, on the
10 validation patients (186 slices) prepared two ways:
(a) the stored data/processed/acdc/val (third-party stage 1 + frozen stage 2);
(b) the same slices rebuilt from native NIfTI with official_test.slices (reconstructed stage 1 +
    frozen stage 2), written to data/processed/acdc_reconstructed_val/val/.
(a) must reproduce each run's recorded per_case_val.csv exactly. Reports per-model differences in
run-level Boundary F1 and DSC (patient-clustered, as the estimator) and per-row maxima.
Outputs: results/acdc_confirm_preproc_check/, revision/acdc_confirm/preprocessing_impact.json.

    .venv/bin/python scripts/acdc_confirm/preprocessing_impact.py
"""

from __future__ import annotations

import csv
import fcntl
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import expansion_statistics as ex  # noqa: E402
import family_statistics as fs  # noqa: E402

from bds_lite.training import build_model  # noqa: E402
from scripts.acdc_confirm.common import EXPLORATORY, RUNS, WORKSPACE  # noqa: E402
from scripts.acdc_confirm.official_test import DEV_RAW, slices  # noqa: E402
from scripts.kd.common import GPU_LOCK, sha  # noqa: E402
from scripts.revision.train_run import evaluate  # noqa: E402
from scripts.revision.training_support import ManifestDataset  # noqa: E402

MANIFEST = ROOT / "recovered/followup/splits/acdc/val.json"
STORED = ROOT / "data/processed/acdc"
REBUILT = ROOT / "data/processed/acdc_reconstructed_val"
OUT = ROOT / "results/acdc_confirm_preproc_check"
REPORT = WORKSPACE / "preprocessing_impact.json"


def rebuild(manifest: dict) -> dict:
    folder = REBUILT / "val"
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(folder)
    folder.mkdir(parents=True, exist_ok=True)
    frames = sorted({re.match(r"(patient\d+_frame\d+)", s)[1] for s in manifest["sample_ids"]})
    written, worst = [], 0.0
    for frame in frames:
        pid = frame.split("_")[0]
        for z, image, mask in slices(
            DEV_RAW / pid / f"{frame}.nii.gz", DEV_RAW / pid / f"{frame}_gt.nii.gz"
        ):
            sample_id = f"{frame}_slice_{z}"
            ref = np.load(STORED / "val" / f"{sample_id}.npz")
            if not np.array_equal(mask, ref["mask"]):
                raise ValueError(f"rebuilt mask differs: {sample_id}")
            worst = max(worst, float(np.abs(image - ref["image"]).max()))
            np.savez_compressed(
                folder / f"{sample_id}.npz", image=image, mask=mask, sample_id=np.asarray(sample_id)
            )
            written.append(sample_id)
    if sorted(written) != manifest["sample_ids"]:
        raise ValueError("rebuilt validation IDs differ from the manifest")
    return dict(slices=len(written), image_max_abs_diff=worst, masks_identical=True)


def rows(path: Path) -> dict:
    with path.open() as stream:
        return {(r["id"], r["label"]): r for r in csv.DictReader(stream)}


def run_mean(path: Path, metric: str) -> float:
    values = ex.unit_values(path, "acdc")[metric]
    return float(np.mean([v for v in values.values() if np.isfinite(v)]))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    rebuilt = rebuild(manifest)
    loaders = {
        name: DataLoader(
            ManifestDataset(root, "val", manifest, 4, 1),
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        for name, root in [("stored", STORED), ("reconstructed", REBUILT)]
    }
    models = {**EXPLORATORY["kd_band"], **EXPLORATORY["decoder"]}
    report = dict(validation_patients=10, rebuild=rebuilt, models=[])
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, digest in models.items():
            folder = RUNS / name
            if sha(folder / "best.pt") != digest:
                raise ValueError(f"checkpoint changed: {name}")
            config = json.loads((folder / "config.json").read_text())
            model = build_model(config)
            payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            model = model.cuda().eval()
            paths = {}
            for variant, data in loaders.items():
                out = OUT / name / variant
                out.mkdir(parents=True, exist_ok=False)
                evaluate(model, data, 4, out, "val")
                paths[variant] = out / "per_case_val.csv"
            recorded, stored, recon = (
                rows(folder / "per_case_val.csv"),
                rows(paths["stored"]),
                rows(paths["reconstructed"]),
            )
            reproduces = recorded == stored
            preds = {v: np.load(OUT / name / v / "preds_val.npz")["predictions"] for v in loaders}
            entry = dict(
                run=name,
                stored_reproduces_recorded_per_case_val=reproduces,
                changed_prediction_pixels=int((preds["stored"] != preds["reconstructed"]).sum()),
                total_prediction_pixels=int(preds["stored"].size),
            )
            for metric in ["boundary_f1", "dsc"]:
                a, b = run_mean(paths["stored"], metric), run_mean(paths["reconstructed"], metric)
                diffs = [
                    abs(fs._value(recon[k], metric) - fs._value(stored[k], metric))
                    for k in stored
                    if np.isfinite(fs._value(stored[k], metric))
                    and np.isfinite(fs._value(recon[k], metric))
                ]
                entry[metric] = dict(
                    stored=a,
                    reconstructed=b,
                    difference=b - a,
                    max_abs_row_difference=max(diffs),
                )
            report["models"].append(entry)
            print(json.dumps(entry), flush=True)
    report["max_abs_run_level_difference"] = {
        m: max(abs(e[m]["difference"]) for e in report["models"]) for m in ["boundary_f1", "dsc"]
    }
    report["all_stored_reproduce_recorded"] = all(
        e["stored_reproduces_recorded_per_case_val"] for e in report["models"]
    )
    report["test_data_used"] = False
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
