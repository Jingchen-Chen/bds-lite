"""Read atomic diagnostic checkpoints; never label partial runs complete."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    reports = []
    for dataset in ["isic2018", "acdc", "synapse"]:
        run = ROOT / f"results/revision_runs/{dataset}_wp5_full_seed1"
        last = run / "last.pt"
        if not last.exists():
            reports.append(dict(dataset=dataset, status="NOT STARTED or no safe checkpoint yet"))
            continue
        content = last.read_bytes()
        state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
        history, probes, epoch, step = (
            state["history"],
            state["probes"],
            state["epoch"],
            state["step"],
        )
        if [r["epoch"] for r in history] != list(range(1, epoch + 1)):
            raise AssertionError("noncontiguous checkpoint history")
        if [r["step"] for r in probes] != list(range(50, step + 1, 50)):
            raise AssertionError("missing or duplicated every50-step probes")
        sum_residuals = []
        for probe in probes:
            total_dot = 0.0
            for name in ["sdf", "boundary", "distill", "aux"]:
                norm, cosine = probe[name + "_norm"], probe[name + "_cosine"]
                if cosine is not None and (not np.isfinite(cosine) or abs(cosine) > 1 + 1e-10):
                    raise AssertionError("invalid cosine")
                if (cosine is None) != (norm == 0 or probe["seg_norm"] == 0):
                    raise AssertionError("undefined-gradient policy mismatch")
                if name != "aux":
                    total_dot += norm * (cosine or 0)
            aux_dot = probe["aux_norm"] * (probe["aux_cosine"] or 0)
            sum_residuals.append(abs(total_dot - aux_dot))
            # Independently accumulated FP32 backward passes need not satisfy a
            # tight scalar identity. Preserve the residual; do not call it passed.
        prior = ROOT / f"results/runs/{dataset}_p1b_bdslite_full_current_seed1/metrics.jsonl"
        old = [json.loads(line) for line in prior.read_text().splitlines()]
        differences = [
            abs(row["val"]["val_dsc"] - old[i]["val"]["val_dsc"]) for i, row in enumerate(history)
        ]
        complete = (run / "final_metrics.json").exists()
        if complete and epoch != 150:
            raise AssertionError("premature final marker")
        reports.append(
            dict(
                dataset=dataset,
                status="FINISHED — needs final validation"
                if complete
                else "RUNNING/RESUMABLE — not a completed experiment",
                checkpoint_epoch=epoch,
                global_step=step,
                probes=len(probes),
                auxiliary_dot_max_absolute_residual=max(sum_residuals, default=0),
                auxiliary_dot_validation="pending independent FP32 accumulation audit",
                null_sdf_cosines=sum(r["sdf_cosine"] is None for r in probes),
                max_historical_prefix_val_dsc_delta=max(differences),
                prefix_trajectory_matches_1e12=max(differences) <= 1e-12,
                epoch_seconds_mean=float(np.mean([r["seconds"] for r in history])),
                last_sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    result = dict(
        scope="point-in-time checkpoint audit, not final scientific validation", reports=reports
    )
    (ROOT / "results/validation/diagnostic_progress.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
