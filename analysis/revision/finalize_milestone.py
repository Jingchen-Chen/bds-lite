"""Write current milestone reports with frozen snapshots of changing runtime logs."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    now = datetime.now(timezone.utc)
    snapshot = ROOT / "revision/audit/runtime_snapshots" / now.strftime("%Y%m%dT%H%M%SZ")
    snapshot.mkdir(parents=True)
    source_rows, progress = [], []
    sources = list((ROOT / "revision/training_supervision").glob("*.jsonl"))
    sources += list((ROOT / "revision/training_supervision").glob("*.log"))
    for dataset in ["isic2018", "acdc", "synapse"]:
        path = ROOT / f"results/revision_runs/{dataset}_wp5_full_seed1/progress.json"
        if path.exists():
            sources.append(path)
            state = json.loads(path.read_text())
            progress.append(
                f"{dataset}: observed epoch{state['epoch']}/150; final acceptance pending"
            )
        else:
            progress.append(f"{dataset}: registered; no saved progress observed")
    for source in sources:
        destination = snapshot / str(source.relative_to(ROOT)).replace("/", "__")
        destination.write_bytes(source.read_bytes())
        source_rows.append(
            dict(
                source=str(source.relative_to(ROOT)),
                snapshot=str(destination.relative_to(ROOT)),
                sha256=sha(destination),
            )
        )
    (snapshot / "provenance.json").write_text(
        json.dumps(
            dict(
                utc=now.isoformat(),
                files=source_rows,
                interpretation="point-in-time runtime snapshots",
            ),
            indent=2,
        )
        + "\n"
    )
    recovery = list(csv.DictReader((ROOT / "revision/validation/prediction_recovery.csv").open()))
    cdf = json.loads((ROOT / "revision/validation/surface_cdf.json").read_text())
    volumes = [
        json.loads(p.read_text())
        for p in (ROOT / "revision/volume_results").rglob("volume_complete.json")
    ]
    directory = ROOT / "results/final_manuscript_audit_v2"
    directory.mkdir(exist_ok=True)
    completion = f"""# Major-revision execution milestone

UTC: {now.isoformat()}. Active repository: /home/jc/bds-lite.
Readiness: **not-ready-major-blockers**. This is not a completed major revision.

Recovery/consolidation:46,422 original research files inventoried;32,926 migrated
paths verified. Original data, checkpoints and directories preserved. Two live
monitor exceptions and verified backups of3 refreshed reports remain explicit.
Followup historical scientific STOP decisions remain in force. Missing separate
fourth historical repository, author anchor and original reviewer PDF are not
reconstructed or invented.

Verified evaluation:12 historical development runs recomputed; existing150-epoch
U seed1 rerun passes WP0 tolerances;9,900 official U/S/F maps validated. Further
{len(recovery)} recovery cohorts/{sum(int(r["images"]) for r in recovery)} maps/
{sum(int(r["class_rows"]) for r in recovery)} image-class rows independently checked.
{len(volumes)} volume cohorts/{sum(r["volumes"] for r in volumes)} volumes/
{sum(r["class_rows"] for r in volumes)} volume-class rows; four real EDT checks.
Matched gate identity:21 cohorts, qualified exploratory statistics, not M5.
CDF:{cdf["cdf_cohorts"]} cohorts/{cdf["valid_surfaces"]} valid surfaces; all saved
HD95/missingness joins and{len(cdf["independent_kdtree_checks"])} independent KDTree checks.
Full error decomposition, figures and generated manuscript fragments are retained.

Actual findings: OFFICIAL_ISIC_FINDINGS.md does not establish the claimed HD95
benefit. VOLUME_FINDINGS.md preserves positive mean Dice and conditional/worse
surface outcomes. GATE_IDENTITY_FINDINGS.md does not assert exact mask invariance.
SURFACE_ERROR_FINDINGS.md describes mixed near/tail changes, not causal mechanism.
No verified millimetre/EF/mass result or Synapse anatomical relabeling is claimed.

Training tools:5,163 stored targets match definitions; CPU exact resume and
three-dataset actual-batch GPU probe-neutrality accepted.65 tests passed in the
last complete suite. Real ISIC epoch2-to3 resume verified;85C stop at epoch24
preserved. Bounded thermal supervisor resumes unchanged runs after cooling;
unknown failures and explicit STOP are not retried. Current observed progress:

{chr(10).join("- " + r for r in progress)}

These diagnostics are not new independent seeds or completed mechanism evidence.
A separate/combined FP32 accumulation residual remains an open numerical audit.
Runtime logs change; frozen snapshot provenance:{snapshot.relative_to(ROOT)}.

Remaining: missing matched experiments/comparators and sensitivity; Phase16 GSL
weights; physical geometry/clinical measures; remaining statistics and resource
profiles; full manuscript/response integration and compilation; author approval.
Human-only:decision/deadline, licensing/narrative/metadata/final manuscript review,
optional TierC and submission. No journal messages, push or submission performed.

See revision/STATUS.md, tasks.csv, COMPLETED_WORK.md, REMAINING_WORK.md,
RESULTS_LOG.md and DEVIATIONS.md. Reports before this task remain under
revision/audit/prior_final_reports/. Use analysis/revision/finalize_milestone.py
while the training-source snapshot (including the legacy reporting script) is frozen.
"""
    audit = """# Integrity and validation audit

All original/migrated preservation results, live-monitor exceptions and report
backups are documented. No scientific original drift, deletion or checkpoint
substitution is accepted. Legacy GSL/Attention and historical development studies
remain separate from matched primary evidence.

Known failures/corrections remain visible:initial contiguous-input parity; whole
snapshot lint scope; empty-SDF probe autograd handling; raw CSV join schemas;
report generator letter-suffix grouping; initial linear CDF rendering. Active
checks exclude immutable recovered/ and do not modify its bytes. Corrected CDFs
use right-continuous steps and retain original linear renders separately.
The gradient scalar-identity check remains pending, not relabeled passed.

The real queue exit130 at epoch2 was an intentional resume check; later exit130
at epoch24 followed the recorded85C guard. Neither is a completed scientific run.
Cooling changes elapsed time, not optimization/data. All restarts are logged.
Training code/config/data hashes stay frozen; actual new-run final archive
acceptance remains pending until150 epochs and independently verified outputs.

Format/lint/test records:revision/audit/final_checks/. Reports are milestone
snapshots, not submission clearance. The artifact manifest excludes mutable live
supervisor files and includes their timestamped audit snapshots instead.
No fully addressed reviewer comment is inferred solely from implemented scripts.
"""
    (directory / "final_completion_report.md").write_text(completion)
    (directory / "final_audit_report.md").write_text(audit)
    paths = []
    for folder in [
        "revision",
        "scripts/revision",
        "analysis/revision",
        "configs/revision",
        "tests/revision",
    ]:
        for path in (ROOT / folder).rglob("*"):
            rel = str(path.relative_to(ROOT))
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".bundle"
                and not rel.startswith("revision/training_supervision/")
                and rel not in ["revision/.gpu.lock", "revision/audit/thermal_supervisor.log"]
            ):
                paths.append(path)
    paths += [directory / "final_completion_report.md", directory / "final_audit_report.md"]
    with (directory / "final_artifact_manifest.csv").open("w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "bytes", "sha256"])
        for path in sorted(set(paths)):
            writer.writerow([str(path.relative_to(ROOT)), path.stat().st_size, sha(path)])
    print(
        "Updated reports and",
        len(set(paths)),
        "frozen-artifact hashes; runtime snapshots preserved",
    )


if __name__ == "__main__":
    main()
