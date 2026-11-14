"""Refresh task-end audit reports without losing historical report versions."""

import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    directory = ROOT / "results/final_manuscript_audit_v2"
    prior = ROOT / "revision/audit/prior_final_reports"
    prior.mkdir(parents=True, exist_ok=True)
    names = ["final_completion_report.md", "final_audit_report.md", "final_artifact_manifest.csv"]
    for name in names:
        source = directory / name
        if source.exists() and not (prior / name).exists():
            shutil.copy2(source, prior / name)
    directory.mkdir(parents=True, exist_ok=True)
    completion = """# Revision execution milestone — 2026-11-02

Overall readiness: **not-ready-major-blockers**. Research recovery, independent-image ISIC
evaluation and the available
stacked-volume/gate evaluations are complete; the major revision is not complete.

Active working repository: /home/jc/bds-lite. Authoritative plan: revision/plan.md.
Existing source directories were preserved; a live monitor changed two operational
JSONs during the audit and both versions are retained. No original dataset,
checkpoint or prediction was deleted or overwritten.

Verified work: SHA256 inventory and controlled migration, 275 loadable checkpoint
paths cataloged, 12 historical development runs' five metrics recomputed,
retrospectively verified 150-epoch U seed1 rerun, registered prospective comparison
plan, official U/S/F three-seed validation/test inference, independent checks of
9,900 saved maps, primary uncertainty/Holm/SESOI analysis, training structural
summaries, generated LaTeX tables and deterministic qualitative draft.
Further completed work: 73 recovery cohorts (45,042 maps, 245,125 image-class
rows), selected-epoch replay for all 12 main ACDC/Synapse checkpoints, 20 volume
cohorts (520 volumes/2,160 class rows), four real-volume EDT spot-checks,
patient/case volume statistics, 202,444 error-decomposition rows and 995 area
groups, and 21 matched gate-identity cohorts with qualified exploratory statistics.

Actual findings and limitations: revision/OFFICIAL_ISIC_FINDINGS.md. The official
test does not establish the proposed HD95 benefit. Original study identities do
not establish patient independence. Negative outcomes are retained.
VOLUME_FINDINGS.md separates positive mean
volume Dice from conditional/worse surface outcomes. GATE_IDENTITY_FINDINGS.md
reports measurable mask/rank changes within engineering SESOI intervals, without
claiming exact invariance, clinical equivalence or a training-time M5 result.

Remaining: resume-capable training adapter and missing matched experiments;
Phase16 GSL checkpoint recovery; official evaluation of all remaining methods;
remaining volume comparators; physical geometry and clinical metrics; other
statistical families; complete figures, manuscript and response integration.
Attention seed2026 is not seeds1/2/3. Legacy GSL is not the Phase16 family.

Human-only dependencies: editorial decision/deadline, final licensing and
narrative approval, optional Tier C, author/coauthor metadata and manuscript
review, final submission. No external messages, Git push or submission performed.

Files and per-task statuses: revision/tasks.csv, STATUS.md, MIGRATION_MANIFEST.md,
COMPLETED_WORK.md and REMAINING_WORK.md. No new training run has been launched.
Prior task-end reports are preserved in revision/audit/prior_final_reports/.
"""
    audit = """# Consolidation and validation audit

See revision/REPOSITORY_AUDIT.md, CONFLICTS.md, GSL_LINEAGE_AUDIT.md,
OFFICIAL_ISIC_FINDINGS.md, DEVIATIONS.md and audit/ logs for evidence.

The full initial inventory covers research files, with environments/caches excluded
explicitly. All 32,926 migrated paths rehash correctly. Original-file recheck has
two documented live-monitor exceptions among 46,422 files; it is not reported as
a zero-exception source-preservation pass. No scientific artifact drift was found.

Raw whole-tree Ruff failed on preserved third-party/snapshot sources. Active-code
checks explicitly exclude recovered/ via ruff.toml. Original snapshot bytes and
failed logs remain intact. Initial baseline whitespace warnings in historical
files were preserved rather than silently rewriting their bytes. Initial commit
failed for missing author identity; the recorded source repository used agent
identity Codex <codex@localhost>, used command-locally for consolidation commits.

The first checkpoint inference parity check failed by 59 pixels on eight images;
contiguous NCHW inputs restored exact parity across all nine fixed checkpoints.
Raw preprocessing and all 9,900 official saved DSC values passed independent
checks. All five metrics on the 6,228 historical development predictions matched.
All 73 recovery cohorts passed independent overlap/ID/shape/dtype/hash validation;
all 21 gate cohorts passed state/configuration/operator-lineage checks. All 20
volume completion hashes and four real-volume EDT metric checks passed. Source
preservation final verification found zero unexpected failures; explicit original
report backups and two live monitor exceptions remain documented.

Final active-code format/lint/test evidence: revision/audit/final_checks/.
No failed scientific check is treated as completed acceptance. WP0 queue remains
partial until a validated resume-capable training adapter exists. Missing GSL
Phase16 checkpoints and unverified physical geometry remain explicit blockers.

This is an execution milestone report, not journal-submission clearance.
"""
    (directory / names[0]).write_text(completion)
    (directory / names[1]).write_text(audit)
    paths = []
    for parent in ["revision", "scripts/revision", "analysis/revision"]:
        paths.extend(
            p
            for p in (ROOT / parent).rglob("*")
            if p.is_file() and p.suffix != ".bundle" and "__pycache__" not in p.parts
        )
    paths.extend([directory / names[0], directory / names[1]])
    with (directory / names[2]).open("w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "bytes", "sha256"])
        for path in sorted(set(paths)):
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            writer.writerow([str(path.relative_to(ROOT)), path.stat().st_size, digest])
    print(
        "Updated milestone reports; preserved prior reports; hashed", len(set(paths)), "artifacts"
    )


if __name__ == "__main__":
    main()
