# ruff: noqa: E501
"""Are the local submitted checkpoints the archived (Zenodo-manifest) checkpoints?

The local outputs/runs/*/best.pt files differ in file SHA-256 from the deleted
ZENODO_MANIFEST.csv of the public release (commit 1b0c4f6), although byte sizes match.
The manifest also lists SHA-256 for every per-sample prediction array behind the
submitted Table 3 (outputs/evaluations/predictions/<ds>_<model>_seed<n>_<split>/<id>.npy,
written by scripts/reeval_main_seeds.py as np.save(uint8 argmax map)). If the local
checkpoints hold the same weights, regenerating those arrays reproduces the manifest hashes
(up to rare argmax flips from CPU vs GPU arithmetic); a different training would reproduce
almost none. Also reports tensor-level state hashes and checkpoint metadata.
Output: revision/validation/checkpoint_identity.json
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

from analysis.revision.evaluate_extra_cohorts import load

ROOT = Path(__file__).resolve().parents[2]
SPLIT = {"isic2018": "val", "acdc": "test", "synapse": "test"}


def main(manifest_path: str) -> None:
    manifest = {r["relative_path"]: r for r in csv.DictReader(open(manifest_path))}
    torch.set_num_threads(4)
    report = []
    for dataset in ("isic2018", "acdc", "synapse"):
        for arm, cohort in (("unet", "famO_unet"), ("bds_lite_full", "famO_full")):
            for seed in (1, 2, 3):
                ckpt = ROOT / f"outputs/runs/{dataset}_{arm}_seed{seed}/best.pt"
                entry = manifest[f"outputs/runs/{dataset}_{arm}_seed{seed}/best.pt"]
                payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                tensor_hash = hashlib.sha256()
                for k in sorted(payload["model"]):
                    tensor_hash.update(k.encode())
                    tensor_hash.update(payload["model"][k].numpy().tobytes())
                model = load(cohort, dataset, seed)
                folder = (
                    f"outputs/evaluations/predictions/{dataset}_{arm}_seed{seed}_{SPLIT[dataset]}"
                )
                expected = {
                    Path(k).stem: v["sha256"]
                    for k, v in manifest.items()
                    if k.startswith(folder + "/")
                }
                match = 0
                for sid, digest in sorted(expected.items()):
                    data = np.load(ROOT / f"data/processed/{dataset}/{SPLIT[dataset]}/{sid}.npz")
                    x = torch.from_numpy(data["image"][None].astype(np.float32))
                    with torch.inference_mode():
                        out = model(x) if arm == "unet" else model(x, return_boundary=False)
                    pred = out["seg_logits"].argmax(1).numpy()[0].astype(np.uint8)
                    buffer = io.BytesIO()
                    np.save(buffer, pred)
                    match += hashlib.sha256(buffer.getvalue()).hexdigest() == digest
                row = dict(
                    run=f"{dataset}_{arm}_seed{seed}",
                    file_sha256_matches_manifest=hashlib.sha256(ckpt.read_bytes()).hexdigest()
                    == entry["sha256"],
                    file_bytes_match=ckpt.stat().st_size == int(entry["bytes"]),
                    tensor_state_sha256=tensor_hash.hexdigest()[:16],
                    epoch=payload.get("epoch"),
                    predictions_in_manifest=len(expected),
                    regenerated_hash_matches=match,
                    match_fraction=match / len(expected) if expected else None,
                )
                report.append(row)
                print(json.dumps(row), flush=True)
    out = ROOT / "revision/validation/checkpoint_identity.json"
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
