# ruff: noqa: E501
"""What the trained gate does in BDS-Lite. Inference only, exploratory and post hoc.

Per dataset, seeds 1-3, first 96 test images of the probe subset: mean |factor - 1|, its constant
per-channel part, the spatial SD per channel, and the share of pixel labels changed by removing the
gate. Output: results/tables/families/gate_structure.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis/revision"))
import boundary_transfer_probe as probe  # noqa: E402

OUT = ROOT / "results/tables/families/gate_structure.csv"


def main() -> None:
    rows = []
    for dataset in probe.DATASETS:
        items = probe.subset(dataset, "test")[:96]
        for seed in (1, 2, 3):
            model = probe.load("M6", dataset, seed)
            factors, gated, plain = [], [], []
            with torch.inference_mode():
                for start in range(0, len(items), 8):
                    x = torch.stack([d[0] for d in items[start : start + 8]]).to(probe.DEVICE)
                    logits, feature = model.seg_decoder(model.encoder(x))
                    q = model.seg_to_boundary_feature(feature)
                    gate = torch.sigmoid(model.boundary_gate.to_gate(q))
                    factor = 1 + model.boundary_gate.scale * (gate - 0.5)
                    factors.append(factor.cpu())
                    gated.append(model.seg_decoder.out(feature * factor).argmax(1).cpu())
                    plain.append(logits.argmax(1).cpu())
            f = torch.cat(factors).numpy()
            rows.append(
                dict(
                    dataset=dataset,
                    seed=seed,
                    mean_abs=float(np.abs(f - 1).mean()),
                    constant=float(np.abs(f.mean((0, 2, 3)) - 1).mean()),
                    spatial_sd=float(f.std((0, 2, 3)).mean()),
                    labels_changed=float((torch.cat(gated) != torch.cat(plain)).float().mean()),
                )
            )
            print(rows[-1], flush=True)
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
