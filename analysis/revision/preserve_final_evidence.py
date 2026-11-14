"""Preserve final statistical evidence as an additive archive; never rewrite experiments.

The historical 1,108-file archive is retained. This adds missing structured inputs,
E3 records and source snapshots in original relative paths. Large data/model/cache
bytes stay local and are listed separately. Experiment C outcomes are excluded.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision/evidence_freeze"
BASE = OUT / "primary-evidence-20261110.sha256"
DELTA = OUT / "final-supplement-20261110.tar.gz"
SUMS = OUT / "final-supplement-20261110.sha256"
RECORD = OUT / "final_preservation.json"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest(path: Path) -> dict[str, str]:
    return {
        name: digest
        for digest, name in (line.split("  ", 1) for line in path.read_text().splitlines())
    }


def selected() -> set[Path]:
    paths = set()
    for folder in ("src", "scripts", "analysis", "configs", "splits", "preregistration"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix in (
                ".py",
                ".sh",
                ".yaml",
                ".yml",
                ".json",
                ".md",
                ".ots",
            ):
                paths.add(path)
    for folder in (
        "revision/tables",
        "revision/validation",
        "revision/kd_unified",
        "revision/kd_posthoc_balanced",
        "revision/restructure_posthoc",
        "revision/followup/results",
        "revision/acdc_confirm/results",
    ):
        paths.update(
            p for p in (ROOT / folder).rglob("*") if p.suffix in (".csv", ".json") and p.is_file()
        )
    # All paths here are structured result tables already published by the audit,
    # not arbitrary training directories (which could expose blinded C outcomes).
    for path in (OUT / "final_statistics_audit.json", OUT / "final_e3_analysis.json"):
        record = json.loads(path.read_text())
        for name in record.get("source_sha256", {}):
            p = ROOT / name
            if p.suffix in (".csv", ".json", ".jsonl", ".yaml", ".py"):
                paths.add(p)
    for name in (
        "results/followup_eval/evaluation_record_e3.json",
        "results/followup_eval/evaluation_record_e1e2.json",
        "results/acdc_confirm_eval/evaluation_record.json",
        "results/acdc_confirm_eval_posthoc/teacher_cache_timing.json",
        "revision/followup_registration.md",
        "revision/followup/AMENDMENT_1.md",
        "revision/followup/PROTOCOL_LOG.md",
        "revision/followup/supervision/events.jsonl",
        "revision/acdc_confirm/official_test_preprocessing.json",
        "revision/acdc_confirm/official_test_manifest.json",
        "revision/acdc_confirm/preprocessing_dev_gate.json",
        "revision/acdc_confirm/preprocessing_impact.json",
        "revision/queue_followup_e1e2.yaml",
        "revision/queue_followup_e3.yaml",
        "revision/COMPARATOR_PROTOCOL.md",
        "revision/EXPANSION_PROTOCOL.md",
        "revision/SYNAPSE_LABEL_SEMANTICS.md",
        "pyproject.toml",
        "revision/env_freeze.txt",
    ):
        paths.add(ROOT / name)
    # Preserve structured histories/configurations for every inventoried record,
    # including negative/partial historical runs; never traverse blinded C outputs.
    inventory = OUT / "final_experiment_inventory.csv"
    for row in csv.DictReader(inventory.open()):
        if row["row_type"] != "training_record" or "Experiment C" in row["family"]:
            continue
        directory = ROOT / row["path"]
        for name in (
            "config.json",
            "history.json",
            "metrics.jsonl",
            "final_metrics.json",
            "provenance.json",
            "kd_probes.json",
            "per_case_test.csv",
            "per_case_val.csv",
        ):
            candidate = directory / name
            if candidate.is_file():
                paths.add(candidate)
    paths.add(ROOT / "revision/audit/project_audit_20261107/experiment_ledger.csv")
    for path in OUT.glob("final_*"):
        if path.suffix in (".json", ".csv", ".md") and path.name not in (
            "final_preservation.json",
            "final_consistency_audit.json",
            "final_checksums.csv",
        ):
            paths.add(path)
    for seed in range(201, 211):
        for arm in ("decoder_aug", "kd_band_aug"):
            run = ROOT / f"results/revision_runs/acdc_followup_{arm}_seed{seed}"
            paths.update(
                run / name
                for name in (
                    "config.json",
                    "history.json",
                    "final_metrics.json",
                    "provenance.json",
                    "kd_probes.json",
                )
            )
            paths.add(
                ROOT
                / f"results/followup_eval/acdc_followup_{arm}_seed{seed}/per_case_official_test.csv"
            )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def verify_archive(archive: Path, expected: dict[str, str]) -> None:
    seen = set()
    with tarfile.open(archive, "r:gz") as stream:
        for item in stream:
            if not item.isfile() or item.name not in expected or item.name in seen:
                raise ValueError(f"Unexpected archive member: {item.name}")
            data = stream.extractfile(item).read()
            if hashlib.sha256(data).hexdigest() != expected[item.name]:
                raise ValueError(f"Archive hash mismatch: {item.name}")
            seen.add(item.name)
    if seen != set(expected):
        raise ValueError("Archive membership mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    base = manifest(BASE)
    for name, digest in base.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Original selected evidence changed: {name}")
    if args.build:
        if DELTA.exists() or SUMS.exists() or RECORD.exists():
            raise FileExistsError("Final preservation outputs already exist; preserve them")
        chosen = {
            str(p.relative_to(ROOT)): p for p in selected() if str(p.relative_to(ROOT)) not in base
        }
        digests = {name: sha(chosen[name]) for name in sorted(chosen)}
        with (
            DELTA.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w|") as archive,
        ):
            for name in sorted(chosen):
                data = chosen[name].read_bytes()
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode, entry.mtime = len(data), 0o644, 0
                archive.addfile(entry, io.BytesIO(data))
        SUMS.write_text("".join(f"{digest}  {name}\n" for name, digest in digests.items()))
        record = {
            "scope": (
                "Final layered statistical-evidence preservation; "
                "not complete training/inference reproduction"
            ),
            "base_members": len(base),
            "supplement_members": len(digests),
            "total_unique_members": len(base) + len(digests),
            "archives": {
                p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                for p in (OUT / "primary-evidence-20261110.tar.gz", DELTA)
            },
            "content_manifests": {p.name: sha(p) for p in (BASE, SUMS)},
            "excluded_large_dependencies": (
                "final_dependency_manifest.csv; local-only, "
                "not independently backed up by this Git freeze"
            ),
            "blinded_exclusion": "No Experiment C outcomes selected",
        }
        RECORD.write_text(json.dumps(record, indent=2) + "\n")
    delta = manifest(SUMS)
    for name, digest in delta.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Final selected evidence changed: {name}")
    verify_archive(OUT / "primary-evidence-20261110.tar.gz", base)
    verify_archive(DELTA, delta)
    print(json.dumps(json.loads(RECORD.read_text()), indent=2))
    print("Both archives verified member-by-member; all selected live bytes match.")


if __name__ == "__main__":
    main()
