"""Verify matched inference settings, exact checkpoint aliases and gate operators."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def signature(config):
    model = config["model"]
    dataset = config["dataset"]
    return dict(
        model=model["name"],
        base=model.get("base_channels", 32),
        channels=model.get("in_channels", dataset.get("in_channels")),
        classes=model.get("num_classes", dataset.get("num_classes")),
        boundary_classes=model.get("boundary_classes", 1),
        gate=model.get("boundary_gate", False),
        scale=model.get("boundary_gate_scale", 0.5),
    )


def main():
    entries = json.loads((ROOT / "protocols/gate_identity/gate_identity_queue.json").read_text())[
        "runs"
    ]
    aliases = {
        (row["arm"], row["seed"]): row["identical"]
        for row in json.loads((ROOT / "results/validation/isic_main_aliases.json").read_text())
    }
    rows = []
    for entry in entries:
        folder = ROOT / entry["output"]
        after = json.loads((folder / "provenance.json").read_text())
        ds, split, seed = entry["dataset"], entry["split"], entry["seed"]
        if ds != "isic2018":
            base = (
                ROOT
                / "results/prediction_recovery/main_p1b_full"
                / f"{ds}_seed{seed}/{split}/provenance.json"
            )
            before = json.loads(base.read_text())
            if before["entry"]["checkpoint_sha256"] != entry["checkpoint_sha256"]:
                raise ValueError("baseline/counterfactual checkpoint differs")
            before_config = before["embedded_config"]
        else:
            if not aliases["F", seed]:
                raise ValueError("ISIC full alias state differs")
            if split == "val":
                base = (
                    ROOT
                    / "recovered/followup/results/followup_20261018_v2"
                    / f"isic2018_F_seed{seed}/config.json"
                )
                before_config = json.loads(base.read_text())
            else:
                base = (
                    ROOT
                    / "results/official_isic"
                    / split.removeprefix("official_")
                    / f"F_seed{seed}/provenance.json"
                )
                before_config = json.loads(base.read_text())["config"]
        if signature(before_config) != signature(after["embedded_config"]):
            raise ValueError("inference architecture/scalar settings differ before intervention")
        if not after["gate_identity_operator_parity"]["exact_logits"]:
            raise ValueError("operator equivalence failed")
        rows.append(
            dict(
                run_id=entry["run_id"],
                matched_model_and_gate_settings=True,
                exact_state_or_alias_verified=True,
                identity_operator_exact_logits=True,
            )
        )
    with (ROOT / "results/validation/gate_identity_lineage.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Verified all", len(rows), "matched gate interventions")


if __name__ == "__main__":
    main()
