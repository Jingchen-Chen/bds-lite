"""Build a conservative task/reviewer ledger; no scientific result synthesis."""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# ID, state, evidence, missing/next action, training requirement
TASKS = [
    (
        "WP0.1a",
        "PARTIAL",
        "revision/audit/inventory.csv; revision/REPO_MAP.md",
        ("Finish checkpoint/prediction lineage and native patient/test-list compatibility"),
        "no",
    ),
    (
        "WP0.1b",
        "COMPLETE — VERIFIED",
        "revision/audit/migration.csv",
        "Migration verified; missing fourth directory remains explicitly unresolved",
        "no",
    ),
    (
        "WP0.2",
        "COMPLETE — VERIFIED",
        "revision/env_freeze.txt",
        "Use recorded interpreter; no upgrades",
        "no",
    ),
    (
        "WP0.3",
        "PARTIAL",
        "scripts/revision/run_queue.py; revision/queue.yaml",
        ("Dispatcher implemented; resume adapter and populated reviewed run matrix still needed"),
        "no",
    ),
    (
        "WP0.4a",
        "PARTIAL",
        "recovered/followup/scripts/followup/run.py",
        (
            "Prediction saving exists in followup, absent required new-run "
            "archive contract in primary trainer"
        ),
        "no",
    ),
    (
        "WP0.4b",
        "PARTIAL",
        "recovered/followup/scripts/followup/run.py",
        ("Gradient probes at 0/25/75/150 differ from required every 50 steps; add adapter"),
        "yes: three seed-1 runs",
    ),
    (
        "WP0.5",
        "COMPLETE — VERIFIED",
        "revision/validation/reused_reproduction.json",
        (
            "Existing May/October 150-epoch logs and metrics verified; "
            "no redundant November rerun needed"
        ),
        "no: reuse completed rerun",
    ),
    (
        "WP0.6",
        "PARTIAL",
        "revision/validation/followup_overlap_recomputed.csv",
        ("ISIC durations measured from recovered runs; other run types still need duration audit"),
        "no",
    ),
    (
        "WP1a",
        "PARTIAL",
        "recovered/followup/research/boundary_attribution_20261017/PROTOCOL.md",
        ("Acquire official val/test, lock checkpoint family and analysis before evaluation"),
        "inference only",
    ),
    (
        "WP1b",
        "PARTIAL",
        "recovered/followup/results/acdc_phase_reduction_audit_20261020",
        (
            "Development stacked masks exist; main matched cases and standard "
            "evaluator parity missing; physical geometry unresolved"
        ),
        "inference/evaluation",
    ),
    (
        "WP1c",
        "PARTIAL",
        "recovered/followup/results/complete_miss_audit_20261018_v2",
        "Extend exact area bins and classwise analysis to all main methods/datasets",
        "evaluation only",
    ),
    (
        "WP1d",
        "BLOCKED",
        (
            "recovered/followup/research/boundary_attribution_20261017/ACDC_GE"
            "OMETRY_AMENDMENT_20261018.md"
        ),
        "Resolve trustworthy voxel units and paired ED/ES test geometry before EF/mass",
        "no",
    ),
    (
        "WP1e",
        "PARTIAL",
        (
            "analysis/revision/structural_descriptors.py; "
            "revision/structural_descriptors/summary.csv"
        ),
        ("Training-mask descriptors generated; needs validation and generated manuscript table"),
        "no",
    ),
    (
        "WP1f",
        "PARTIAL",
        "recovered/followup/research/boundary_attribution_20261017/final_analysis",
        (
            "ISIC remote-FP diagnostics exist; exact 2px FP/FN decomposition "
            "and all-method CDFs missing"
        ),
        "evaluation only",
    ),
    (
        "WP1g",
        "PARTIAL",
        "results/checkpoints; recovered/followup/results/gsl",
        (
            "Map actual GSL checkpoints to saved per-case outputs; regenerate "
            "missing predictions before considering retraining"
        ),
        "inference first",
    ),
    (
        "WP1h",
        "PARTIAL",
        "results/submission_critical_evidence_v1/training_cost_profiling_results.md",
        ("Historical local profiling exists; batch1/8/16 CPU and kernel analysis still required"),
        "profiling only",
    ),
    (
        "WP1i",
        "PARTIAL",
        "results/final_manuscript_audit_v2; scripts/eval_p1b_gate_comparison.py",
        ("Gate artifacts exist; validate matched family, metric tolerance and full seed coverage"),
        "inference/evaluation",
    ),
    (
        "WP2.M0",
        "COMPLETE — NEEDS VALIDATION",
        "results/checkpoints; recovered/followup/results/followup_20261018_v2",
        ("ISIC 3-seed overlap verified; main all-dataset protocol and distance validation pending"),
        "reuse; no automatic rerun",
    ),
    (
        "WP2.M1",
        "PARTIAL",
        "recovered/followup/results/followup_20261018_v2",
        "S arm ISIC 3 seeds exists; ACDC/Synapse locked matched runs not verified",
        "conditional missing datasets",
    ),
    (
        "WP2.M2",
        "PARTIAL",
        "recovered/followup/results/pathway_controls_20261018",
        (
            "Inspect exact C arm and historical boundary-only variants; cannot"
            " equate names with configuration"
        ),
        "conditional",
    ),
    (
        "WP2.M3",
        "PARTIAL",
        "recovered/followup/results/followup_20261018_v2",
        "B arm ISIC 3 seeds exists; ACDC/Synapse missing compatible evidence",
        "conditional missing datasets",
    ),
    (
        "WP2.M4",
        "PARTIAL",
        "recovered/followup/results/boundary_specificity_20261018",
        "D arm/control configs need exact mapping; other datasets not verified",
        "conditional",
    ),
    (
        "WP2.M5",
        "PARTIAL",
        "results/checkpoints; recovered/followup/results/pathway_controls_20261018",
        "Separate gate-identity-at-inference from alpha=0 training; verify all seeds",
        "conditional",
    ),
    (
        "WP2.M6",
        "COMPLETE — NEEDS VALIDATION",
        "results/checkpoints; recovered/followup/results/followup_20261018_v2",
        "ISIC overlap verified; complete primary protocol lineage still open",
        "reuse",
    ),
    (
        "WP2.analysis",
        "NOT STARTED",
        "revision/plan.md",
        "All-dataset ablation table and two Holm families need complete inputs",
        "no",
    ),
    (
        "WP3.alpha",
        "NOT STARTED",
        "revision/plan.md",
        "Create locked sensitivity configs after Tier A",
        "yes",
    ),
    (
        "WP3.lambda",
        "NOT STARTED",
        "revision/plan.md",
        "Create locked sensitivity configs after Tier A",
        "yes",
    ),
    (
        "WP4.C1",
        "NOT STARTED",
        "revision/plan.md",
        "Verify official Boundary Loss implementation and schedule",
        "yes",
    ),
    (
        "WP4.C2",
        "COMPLETE — NEEDS VALIDATION",
        "results/checkpoints; src/bds_lite/losses/boundary_dou.py",
        (
            "Historical BoundaryDoU checkpoints exist; check protocol/sanity "
            "and regenerate predictions"
        ),
        "inference first",
    ),
    (
        "WP4.GSL",
        "COMPLETE — NEEDS VALIDATION",
        "results/checkpoints; recovered/followup/results/gsl",
        ("Reported nine matched runs not fully present at cited Phase16 paths; resolve artifacts"),
        "inference first",
    ),
    (
        "WP4.C3",
        "PARTIAL",
        "revision/plan.md",
        "Existing compact Attention U-Net code/configs; pin source and verify matched runs",
        "yes",
    ),
    (
        "WP4.C4",
        "NOT STARTED",
        "revision/plan.md",
        "Inspect official LB-UNet license/multiclass adaptation",
        "yes",
    ),
    (
        "WP4.C5",
        "NOT STARTED",
        "revision/plan.md",
        "Pin Swin-Unet/pretraining and document reference protocol",
        "yes",
    ),
    (
        "WP4.C6_C7",
        "BLOCKED",
        "revision/plan.md",
        "Tier C requires HUMAN decision; old EGE-UNet file does not approve new Tier C",
        "human decision",
    ),
    (
        "WP4.literature",
        "NOT STARTED",
        "revision/plan.md",
        ("Verify each original source/table; create lit_numbers.csv without invented rows"),
        "no",
    ),
    (
        "WP5a",
        "PARTIAL",
        "recovered/followup/research/boundary_attribution_20261017/final_analysis",
        (
            "ISIC fixed-eight-image gradient probes exist; do not equate to "
            "50-step three-dataset hook"
        ),
        "yes: seed-1 diagnostics",
    ),
    (
        "WP5b",
        "PARTIAL",
        "revision/structural_descriptors; revision/methods_verified.md",
        ("Derivation drafted; validate training L/A estimator and small-displacement assumptions"),
        "no",
    ),
    (
        "WP5c",
        "PARTIAL",
        "revision/methods_verified.md",
        (
            "Bounded-gate/stopgrad/source formulas drafted; citations and "
            "manuscript integration pending"
        ),
        "no",
    ),
    (
        "WP6.prereg",
        "COMPLETE — VERIFIED",
        "analysis/revision/prereg_revision.md",
        "Commit a6dc26c predates official model evaluations; historical findings declared",
        "no",
    ),
    (
        "WP6.families",
        "PARTIAL",
        "recovered/followup/analysis; results/final_manuscript_audit",
        (
            "Legacy cluster statistics exist; locked revision P/A/C and seed-"
            "averaged units not yet revalidated"
        ),
        "no",
    ),
    (
        "WP6.sesoi_power",
        "NOT STARTED",
        "analysis/revision/prereg_revision.md",
        ("Generate CIs, SESOI judgements and cluster-level detectable effect after prereg"),
        "no",
    ),
    (
        "WP7a",
        "BLOCKED",
        "revision/plan.md",
        ("ISIC official page verified; ACDC/Synapse HUMAN final license approval outstanding"),
        "human decision",
    ),
    (
        "WP7b",
        "NOT STARTED",
        "revision/plan.md",
        "Deterministic examples and image/GT/boundary/SDF panels with license gating",
        "no",
    ),
    (
        "WP7c",
        "PARTIAL",
        "scripts/generate_manuscript_qualitative_figure.py; recovered/followup/figures",
        (
            "Legacy figures not full required GSL/M1/comparator panel; "
            "deterministic selection manifest needed"
        ),
        "inference first",
    ),
    (
        "WP7d",
        "NOT STARTED",
        "revision/plan.md",
        ("Required revision heatmap/sensitivity/gradient/clinical figures depend on results"),
        "no",
    ),
    (
        "WP8",
        "BLOCKED",
        "revision/plan.md",
        ("HUMAN decides Tier C; closed followup 2.5D routes not a matched revision success"),
        "human decision",
    ),
    (
        "WP9",
        "PARTIAL",
        "recovered/peerj-submission/main.tex; revision/methods_verified.md",
        (
            "Submitted LaTeX recovered; result-independent draft only, verify "
            "and integrate F1-F10 after results"
        ),
        "no",
    ),
    (
        "WP10",
        "PARTIAL",
        "revision/reviewer_requirements.csv",
        (
            "Original review wording/PDF unavailable; prepare ID-based "
            "scaffold without invented quotations/line numbers"
        ),
        "no",
    ),
    (
        "WP11",
        "PARTIAL",
        "revision/audit; revision/STATUS.md",
        (
            "Final tables/manuscript/response/licensing/author review and "
            "release criteria remain open"
        ),
        "no",
    ),
]


def main():
    directory = ROOT / "revision"
    updates = json.loads((directory / "task_updates.json").read_text())
    tasks = [tuple(updates.get(row[0], row)) for row in TASKS]
    with (directory / "tasks.csv").open("w") as stream:
        w = csv.writer(stream, lineterminator="\n")
        w.writerow(
            [
                "task",
                "status",
                "evidence",
                "missing_and_next_action",
                "additional_training_or_evaluation",
            ]
        )
        w.writerows(tasks)
    status = [
        "# Major revision status — 2026-11-02",
        "",
        ("Statuses concern revision acceptance, not whether scripts or prose reports exist."),
        (
            "Dataset development results are not official-test findings. "
            "Manuscript integration is not yet complete."
        ),
        "",
        "| Task | Status | Actual evidence | Missing / next action | Compute |",
        "|---|---|---|---|---|",
    ]
    overview = ["## Work-package checklist", ""]
    for number in range(12):
        wp = f"WP{number}"
        subset = [row[1] for row in tasks if row[0].split(".")[0] == wp]
        done = all(value == "COMPLETE — VERIFIED" for value in subset)
        state = (
            "COMPLETE — VERIFIED"
            if done
            else "BLOCKED"
            if all(value == "BLOCKED" for value in subset)
            else "PARTIAL"
            if any(value != "NOT STARTED" for value in subset)
            else "NOT STARTED"
        )
        overview.append(f"- [{'x' if done else ' '}] {wp}: {state}")
    overview.extend(["", "Detailed task evidence and reviewer IDs are also in tasks.csv.", ""])
    status[5:5] = overview
    status += ["| " + " | ".join(row) + " |" for row in tasks]
    (directory / "STATUS.md").write_text("\n".join(status) + "\n")
    matrix = []
    for line in (directory / "plan.md").read_text().splitlines():
        if re.match(r"^\| (E-|R[123]-|S-)", line):
            fields = [p.strip() for p in line.strip("|").split("|")]
            if len(fields) == 5:
                matrix.append(
                    dict(
                        reviewer_id=fields[0],
                        requirement=fields[1],
                        required_action=fields[2],
                        wp=fields[3],
                        priority=fields[4],
                        status="PARTIAL"
                        if fields[3] in ["WP9", "WP10", "WP10/11"]
                        else "NOT STARTED",
                        evidence="revision/tasks.csv; revision/STATUS.md",
                        missing=(
                            "Full reviewer closure requires validated result + manuscript "
                            "location + response; none certified closed"
                        ),
                        validation=("Mapped from roadmap; original review PDF absent"),
                        additional_training_or_evaluation="See linked WP task",
                        next_action=fields[2],
                    )
                )
    with (directory / "reviewer_requirements.csv").open("w") as stream:
        w = csv.DictWriter(stream, fieldnames=list(matrix[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(matrix)
    ledger = []
    for row in tasks:
        item = dict(
            zip(
                [
                    "task",
                    "status",
                    "evidence",
                    "missing_and_next_action",
                    "additional_training_or_evaluation",
                ],
                row,
                strict=True,
            )
        )
        wp = row[0].split(".")[0]
        ids = []
        for requirement in matrix:
            tokens = requirement["wp"].replace("WP", "").split("/")
            if wp.removeprefix("WP") in tokens:
                ids.append(requirement["reviewer_id"])
        item.update(
            reviewer_ids=";".join(ids),
            requirement_reference="revision/plan.md " + row[0],
            validation_performed=(
                "Verified output, see evidence"
                if row[1] == "COMPLETE — VERIFIED"
                else "Partial checks; acceptance evidence incomplete"
            ),
            manuscript_integration="Not yet integrated into final revised manuscript",
        )
        ledger.append(item)
    with (directory / "tasks.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ledger)
    print(len(tasks), "tasks;", len(matrix), "reviewer requirements")


if __name__ == "__main__":
    main()
