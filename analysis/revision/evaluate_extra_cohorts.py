"""Evaluate fixed extra checkpoint cohorts with the exact revision evaluator (CPU).

Cohorts (no training, no selection; checkpoints fixed beforehand):
  isic_M3_B : ISIC followup arm B (BDS-Lite - SDF) seeds 1-3 -> official test.
  famO_unet : submitted-manuscript family O U-Net (outputs/runs) on all test sets.
  famO_full : submitted-manuscript family O BDS-Lite full on all test sets.

Inputs use the same datasets as the new runs (train_run.OfficialTestDataset for the
official ISIC test; strict ManifestDataset for ACDC/Synapse test) and the same per-row
metric function (recover_predictions.case_metrics). Family-O BDS-Lite keys
projection_head.* are renamed to seg_to_boundary_feature.* (pure key rename,
scripts/adapt_legacy_bdslite_for_eval.py); a parity check against family O's own
saved development-split evaluations guards the graph semantics before use.
Outputs: revision/extra_cohorts/<cohort>/<dataset>_seed<s>/test/{per_case.csv,
predictions.npz, provenance.json}.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.revision.recover_predictions import IdentityGate, case_metrics
from bds_lite.models import BDSLiteUNet, UNet
from bds_lite.training import build_model
from bds_lite.training.runner import inference_forward
from scripts.revision.train_run import OfficialTestDataset
from scripts.revision.training_support import ManifestDataset

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision/extra_cohorts"
CLASSES = {"isic2018": 2, "acdc": 4, "synapse": 9}
CHANNELS = {"isic2018": 3, "acdc": 1, "synapse": 1}
FOLLOWUP = ROOT / "recovered/followup/results/followup_20261018_v2"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint(cohort: str, dataset: str, seed: int) -> Path:
    if cohort == "isic_M3_B":
        return FOLLOWUP / f"isic2018_B_seed{seed}/best.pt"
    arm = {"famO_unet": "unet", "famO_full": "bds_lite_full", "famO_full_gateid": "bds_lite_full"}[
        cohort
    ]
    return ROOT / f"outputs/runs/{dataset}_{arm}_seed{seed}/best.pt"


def load(cohort: str, dataset: str, seed: int) -> torch.nn.Module:
    payload = torch.load(checkpoint(cohort, dataset, seed), map_location="cpu", weights_only=False)
    state = payload["model"]
    if cohort == "isic_M3_B":
        model = build_model(payload["config"])
    elif cohort == "famO_unet":
        model = UNet(CHANNELS[dataset], CLASSES[dataset], 32)
    else:  # famO_full, famO_full_gateid
        model = BDSLiteUNet(CHANNELS[dataset], CLASSES[dataset], 32, 1, True, 0.25)
        state = {
            k.replace("projection_head.", "seg_to_boundary_feature.", 1): v
            for k, v in state.items()
        }
    model.load_state_dict(state, strict=True)
    if cohort == "famO_full_gateid":
        # F10/WP1i: same trained weights, gate replaced by unit modulation (g = 0.5)
        model.boundary_gate = IdentityGate()
    return model.eval()


def dataset_for(dataset: str, split: str):
    if dataset == "isic2018" and split == "test":
        return OfficialTestDataset()
    manifest = json.loads((ROOT / f"recovered/followup/splits/{dataset}/{split}.json").read_text())
    return ManifestDataset(
        ROOT / f"data/processed/{dataset}", split, manifest, CLASSES[dataset], CHANNELS[dataset]
    )


@torch.inference_mode()
def predict(model, data) -> tuple[list[str], np.ndarray, list[np.ndarray]]:
    loader = DataLoader(data, batch_size=16, shuffle=False, num_workers=0)
    ids, preds, masks = [], [], []
    for batch in loader:
        logits = inference_forward(model, batch["image"].float())["seg_logits"]
        preds.append(logits.argmax(1).numpy().astype(np.uint8))
        ids.extend(batch["id"])
        masks.extend(batch["mask"].numpy())
    return ids, np.concatenate(preds), masks


def parity(cohort: str) -> list[dict]:
    """Family O: current-code ISIC development-split DSC vs family O's saved evaluation."""
    arm = {"famO_unet": "unet", "famO_full": "bds_lite_full"}[cohort]
    rows = []
    for seed in (1, 2, 3):
        saved = {
            r["sample_id"]: float(r["dsc"])
            for r in csv.DictReader(
                (ROOT / f"outputs/evaluations/isic2018_{arm}_seed{seed}_test.csv").open()
            )
            if r["class_id"] == "1"
        }
        ids, preds, masks = predict(load(cohort, "isic2018", seed), dataset_for("isic2018", "val"))
        now = {i: case_metrics(p, m, 1)["dsc"] for i, p, m in zip(ids, preds, masks, strict=True)}
        common = sorted(set(saved) & set(now))
        diff = max(abs(saved[i] - now[i]) for i in common)
        rows.append(dict(cohort=cohort, seed=seed, n=len(common), max_abs_dsc_diff=diff))
        print(rows[-1], flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        required=True,
        choices=["isic_M3_B", "famO_unet", "famO_full", "famO_full_gateid"],
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cohort in ("famO_unet", "famO_full"):
        check = parity(args.cohort)
        (OUT / f"{args.cohort}_parity.json").write_text(json.dumps(check, indent=1) + "\n")
        # Family O was trained/evaluated with pre-release code on a different ISIC split
        # (splits/isic2018_split.json, seed 1: 1815/389/390); its saved CSV overlaps the
        # locked development split in 77 images. Observed |dDSC| <= 7.8e-4 is consistent
        # with a different numeric evaluation path, not a different graph (a semantic
        # mismatch, e.g. wrong gate input, gives O(0.1) differences). Tolerance 2e-3.
        if any(r["max_abs_dsc_diff"] > 2e-3 or r["n"] < 50 for r in check):
            raise ValueError("family O parity failed; graph semantics differ; not evaluated")
    datasets = ["isic2018"] if args.cohort == "isic_M3_B" else ["isic2018", "acdc", "synapse"]
    for dataset in datasets:
        data = dataset_for(dataset, "test")
        for seed in (1, 2, 3):
            folder = OUT / args.cohort / f"{dataset}_seed{seed}" / "test"
            if (folder / "provenance.json").exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            ids, preds, masks = predict(load(args.cohort, dataset, seed), data)
            rows = [
                dict(id=i, **case_metrics(p, m, label))
                for i, p, m in zip(ids, preds, masks, strict=True)
                for label in range(1, CLASSES[dataset])
            ]
            with (folder / "per_case.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            np.savez_compressed(folder / "predictions.npz", ids=np.array(ids), predictions=preds)
            ckpt = checkpoint(args.cohort, dataset, seed)
            (folder / "provenance.json").write_text(
                json.dumps(
                    dict(
                        checkpoint=str(ckpt.relative_to(ROOT)),
                        checkpoint_sha256=sha(ckpt),
                        n_samples=len(ids),
                        evaluator="recover_predictions.case_metrics; CPU fp32; argmax",
                        per_case_sha256=sha(folder / "per_case.csv"),
                    ),
                    indent=1,
                )
                + "\n"
            )
            print("done", args.cohort, dataset, seed, len(ids), flush=True)


if __name__ == "__main__":
    main()
