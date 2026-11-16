import csv
import hashlib
from pathlib import Path

import torch

r = Path("/home/jc/bds-lite")
rows = []
for base in [r / "results", r / "outputs", r / "recovered/followup/results"]:
    for p in sorted(base.rglob("best.pt")):
        x = torch.load(p, map_location="cpu", weights_only=False)
        c = x.get("config", {})
        t = c.get("trainer", {})
        loss_cfg = c.get("loss", {})
        rows.append(
            dict(
                path=str(p.relative_to(r)),
                epoch=x.get("epoch"),
                dataset=c.get("dataset", {}).get("name"),
                seed=c.get("project", {}).get("seed"),
                model=c.get("model", {}).get("name"),
                planned_epochs=t.get("epochs"),
                batch=t.get("batch_size"),
                lr=t.get("lr"),
                surface=loss_cfg.get("surface_weight"),
                boundary=loss_cfg.get("boundary_weight"),
                distill=loss_cfg.get("distill_weight"),
                gsl=loss_cfg.get("use_gsl"),
                prediction_npy_count=len(list((p.parent / "predictions").glob("*.npy"))),
                same_folder_config=(p.parent / "config.json").exists(),
                last_pt=(p.parent / "last.pt").exists(),
                sha256=hashlib.file_digest(p.open("rb"), "sha256").hexdigest(),
                verification="loadable; no training-reproduction claim",
            )
        )
with (r / "results/audit/checkpoints.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
print("loadable checkpoints", len(rows))
