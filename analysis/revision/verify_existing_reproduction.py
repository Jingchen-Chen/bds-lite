"Reuse a logged 150-epoch rerun only after trajectory/state/metric checks."

import csv
import io
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    old_log = ROOT / "results/runs/isic2018_p1b_unet_seed1/metrics.jsonl"
    new_run = ROOT / "recovered/followup/results/followup_20261018_v2/isic2018_U_seed1"
    old = [json.loads(x) for x in old_log.read_text().splitlines()]
    new = [json.loads(x) for x in (new_run / "metrics.jsonl").read_text().splitlines()]
    if [r["epoch"] for r in old] != list(range(1, 151)) or [r["epoch"] for r in new] != list(
        range(1, 151)
    ):
        raise ValueError("incomplete original or rerun history")
    trajectory_delta = max(
        abs(a["val"]["val_dsc"] - b["val_dsc"]) for a, b in zip(old, new, strict=True)
    )
    old_ckpt = torch.load(
        ROOT / "results/checkpoints/isic2018_p1b_unet_seed1/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    new_ckpt = torch.load(new_run / "best.pt", map_location="cpu", weights_only=False)
    state_equal = old_ckpt["model"].keys() == new_ckpt["model"].keys() and all(
        torch.equal(v, new_ckpt["model"][k]) for k, v in old_ckpt["model"].items()
    )
    source = ROOT / "results/checkpoints/isic2018_p1b_unet_seed1/eval_val.csv"
    raw = source.read_text()
    # Preserve original: this historical export literally encoded its line endings.
    normalized = raw.replace("\\r\\n", "\n")
    old_metrics = next(r for r in csv.DictReader(io.StringIO(normalized)) if r["label"] == "mean")
    summary = json.loads((new_run / "summary.json").read_text())
    dsc_delta = abs(float(old_metrics["dsc"]) - summary["dsc"])
    hd_delta = abs(float(old_metrics["hd95"]) - summary["hd95"])
    passed = trajectory_delta < 1e-12 and state_equal and dsc_delta <= 0.005 and hd_delta <= 1
    result = dict(
        status="COMPLETE — VERIFIED" if passed else "BLOCKED",
        scope=("Verified logged May/October rerun; no additional seed or fresh November run"),
        epochs_compared=150,
        max_validation_trajectory_delta=trajectory_delta,
        selected_state_equal=state_equal,
        dsc_delta=dsc_delta,
        hd95_delta_px=hd_delta,
        original_eval_encoding="literal escaped CRLF decoded in memory; source unchanged",
        original_best_epoch=old_ckpt["epoch"],
        rerun_best_epoch=new_ckpt["epoch"],
        evidence=[
            "results/validation/reused_reproduction.json",
            str(old_log.relative_to(ROOT)),
            str((new_run / "metrics.jsonl").relative_to(ROOT)),
            str((new_run / "status.json").relative_to(ROOT)),
            "results/validation/followup_overlap_recomputed.csv",
            "results/validation/followup_boundary_distance_recomputed.csv",
        ],
        note=(
            "U seed1 tolerance passed; three-dataset gradient diagnostics"
            " and trainer resume pending"
        ),
    )
    (ROOT / "results/validation/reused_reproduction.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (ROOT / "results/gates/reproduction_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise ValueError("existing rerun fails verification")


if __name__ == "__main__":
    main()
