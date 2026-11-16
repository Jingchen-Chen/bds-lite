# ruff: noqa: E501
"""Phase 1 analysis of the KD comparator: training and validation data only (protocol 6, OPEN-2).

Reads the teacher run, the cache/gate record, D+KD-band seeds 1-3, and the recorded decoder-only
validation files (seeds 1-10). Never opens a test file; it refuses to run if any new-arm folder
contains a test output. Writes results/kd_isic2018/phase1/phase1_validation.{json,md}.

Feasibility rule (Option A, frozen): proceed to Phase 2 if and only if
  F1 the teacher passed the quality gate (OPEN-1);
  F2 every Phase 1 student completed 150 epochs, every logged training and validation loss is
     finite, and L_kd decreased (mean over epochs 141-150 below the epoch-1 mean);
  F3 all Phase 1 runs completed through the queue at batch size 8 (the pre-specified fallback,
     batch 4 x 2 accumulation, would also satisfy F3 but is labelled). Supervisor halts and thermal
     resumes are reported; a halt that was followed by a completed run does not fail F3.
The validation contrasts are exploratory and are not part of the decision.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT))
import expansion_statistics as ex  # noqa: E402

RUNS = ROOT / "results/revision_runs"
TEACHER = RUNS / "isic2018_kd_teacher_seed100"
CACHE = RUNS / "isic2018_kd_teacher_cache"
STUDENTS = [RUNS / f"isic2018_kd_band_seed{s}" for s in (1, 2, 3)]
DECODER = [RUNS / f"isic2018_m2_aux_boundary_only_seed{s}" for s in range(1, 11)]
EVENTS = ROOT / "results/kd_isic2018/supervision/events.jsonl"
OUT = ROOT / "results/kd_isic2018/phase1"
TEST_OUTPUTS = ["preds_test.npz", "per_case_test.csv", "test_metrics.json"]
METRICS = ["boundary_f1", "dsc", "hd95", "iou", "assd"]


def no_test_outputs(folders) -> None:
    for folder in folders:
        for name in TEST_OUTPUTS:
            if (folder / name).exists():
                raise RuntimeError(f"test output exists in a new-arm folder: {folder / name}")


def finite_history(history) -> bool:
    for row in history:
        for part in ("train", "val"):
            for value in row[part].values():
                if isinstance(value, float) and not math.isfinite(value):
                    return False
    return True


def telemetry_max(folder: Path) -> float | None:
    path = folder / "gpu_telemetry.csv"
    if not path.exists():
        return None
    temps = []
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) > 1 and parts[1].strip().isdigit():
            temps.append(int(parts[1]))
    return max(temps) if temps else None


def run_summary(folder: Path, student: bool) -> dict:
    final = json.loads((folder / "final_metrics.json").read_text())
    history = json.loads((folder / "history.json").read_text())
    summary = dict(
        run=folder.name,
        epochs=len(history),
        selected_epoch=final["selected_epoch"],
        best_val_dsc=final["best_val_dsc"],
        val=final["metrics"]["val"],
        finite=finite_history(history),
        median_epoch_seconds=statistics.median(h["seconds"] for h in history),
        peak_allocated_mib=max(h["peak_allocated_mib"] for h in history),
        peak_reserved_mib=max(h["peak_reserved_mib"] for h in history),
        max_temperature_c=telemetry_max(folder),
        amp_skipped_steps=sum(h["train"]["amp_skipped_steps"] for h in history),
    )
    if student:
        kd = [h["train"]["kd"] for h in history]
        summary["kd_epoch1"] = kd[0]
        summary["kd_last10_mean"] = float(np.mean(kd[-10:]))
        summary["kd_decreased"] = summary["kd_last10_mean"] < kd[0]
        summary["probes"] = json.loads((folder / "kd_probes.json").read_text())
    return summary


def events() -> list[dict]:
    if not EVENTS.exists():
        return []
    return [json.loads(line) for line in EVENTS.read_text().splitlines() if line.strip()]


def per_case(path: Path) -> dict:
    return ex.unit_values(path, "isic2018")


def main() -> None:
    no_test_outputs([TEACHER, CACHE, *STUDENTS])
    for folder in [TEACHER, *STUDENTS]:
        if not (folder / "final_metrics.json").exists():
            raise FileNotFoundError(f"Phase 1 run incomplete: {folder.name}")
    gate = json.loads((CACHE / "cache_complete.json").read_text())
    diagnostics = json.loads((CACHE / "diagnostics.json").read_text())
    teacher = run_summary(TEACHER, student=False)
    students = [run_summary(f, student=True) for f in STUDENTS]
    log = events()
    halts = [e for e in log if e["event"].startswith("halt")]
    thermal = [e for e in log if e["event"] == "guarded_resume_pending_cooling"]
    batch_sizes = {
        json.loads((f / "config.json").read_text())["trainer"]["batch_size"]
        for f in [TEACHER, *STUDENTS]
    }
    checks = dict(
        F1_teacher_gate=gate["gate"]["status"] == "PASSED",
        F2_students_stable=all(
            s["epochs"] == 150 and s["finite"] and s["kd_decreased"] for s in students
        ),
        F3_resources=batch_sizes <= {8, 4},
    )
    contrasts = {}
    kd_values = [per_case(f / "per_case_val.csv") for f in STUDENTS]
    decoder_values = [per_case(f / "per_case_val.csv") for f in DECODER]
    for metric in METRICS:
        contrasts[metric] = ex.contrast(kd_values, decoder_values, metric)
    result = dict(
        status="exploratory; validation images only; no new-arm test metric exists",
        decision_rule="Option A (protocol OPEN-2): feasibility only, not the direction of any estimate",
        feasibility=checks,
        proceed_to_phase2=all(checks.values()),
        gate=gate["gate"],
        cache_check=gate["check"],
        teacher=teacher,
        teacher_diagnostics=diagnostics,
        students=students,
        validation_contrasts_kd_band_minus_decoder=contrasts,
        supervision=dict(thermal_resumes=len(thermal), halts=halts),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase1_validation.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    lines = [
        "# KD comparator Phase 1 (validation only, exploratory)",
        "",
        f"Teacher gate: {gate['gate']['status']} (teacher val Boundary F1 "
        f"{gate['gate']['teacher']['boundary_f1']:.4f} vs {gate['gate']['thresholds']['boundary_f1_min']:.4f}; "
        f"DSC {gate['gate']['teacher']['dsc']:.4f} vs {gate['gate']['thresholds']['dsc_min']:.4f}).",
        f"Feasibility: {checks}. Proceed to Phase 2 under Option A: {result['proceed_to_phase2']}.",
        "",
        "| Metric | D+KD-band (3) | Decoder-only (10) | Delta [95% CI] | label |",
        "|---|---|---|---|---|",
    ]
    for metric, c in contrasts.items():
        lines.append(
            f"| {metric} | {c['arm_a_mean']:.4f} | {c['arm_b_mean']:.4f} | "
            f"{c['delta']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}] | {c['classification']} |"
        )
    lines += [
        "",
        "Positive favours D+KD-band; HD95/ASSD signs reversed. Three runs against ten: not a test.",
    ]
    (OUT / "phase1_validation.md").write_text("\n".join(lines) + "\n")
    with (OUT / "phase1_runs.csv").open("w") as stream:
        fields = [
            "run",
            "epochs",
            "selected_epoch",
            "best_val_dsc",
            "median_epoch_seconds",
            "peak_allocated_mib",
            "max_temperature_c",
            "amp_skipped_steps",
        ]
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows([teacher, *students])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
