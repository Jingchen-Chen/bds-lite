# ruff: noqa: E501
"""Unified three-dataset analysis of external-teacher D+KD-band (exploratory; frozen before results).

Plan: revision/KD_COMPARATOR_AMENDMENT_1.md item 5 (ISIC2018) and revision/KD_MULTICLASS_PROTOCOL.md 7
(ACDC, Synapse). Estimator: analysis/revision/expansion_statistics.py::contrast and unit_values,
unchanged (per-image values; ACDC/Synapse label-macro per slice, then the patient/case unit mean;
delta of arm means of run means; SE = Welch run SE combined with the unit SE; Welch df).

Arms: D+KD-band seeds 1-3 (new). References, recorded and not retrained:
  ISIC2018  decoder-only, U-Net, BDS-Lite seeds 1-10 (primary) and seeds 1-3 (seed-matched sensitivity),
            paths from expansion_statistics.isic_path;
  ACDC, Synapse  decoder-only (M2; ACDC unweighted), U-Net (M0), BDS-Lite (M6), seeds 1-3, paths
            from expansion_statistics.cell_path (the manuscript's retrained references).
Primary (exploratory): D+KD-band - decoder-only, test Boundary F1, per dataset; unadjusted p and Holm
over the three datasets. Everything else is descriptive. Reading of an interval (no "confirmed"):
  beyond +/-0.01 (2 px HD95, 0.5 px ASSD): exploratory relevant gain/harm; excludes zero only:
  exploratory gain/harm of unclear size; inside +/-SESOI: no relevant effect; otherwise not established.
Also: per-seed values, seed consistency, per-class deltas (ACDC/Synapse), missed-structure and
false-alarm rates, validation contrasts against decoder-only, inverse-variance pooled KD - decoder
per metric with Cochran's Q (exploratory), teacher test metrics, and training/inference cost.
Boundary specificity is not assessed (no D+KD-full control).

Outputs: revision/kd_unified/{unified_results.json, contrasts.csv, per_seed.csv, per_class.csv,
costs.csv, summary.md}. Datasets without a completed test evaluation are reported as missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT))
import expansion_statistics as ex  # noqa: E402
import family_statistics as fs  # noqa: E402

from analysis.kd3.integrity import check  # noqa: E402
from analysis.revision import make_audit_assets as ma  # noqa: E402

RUNS = ROOT / "results/revision_runs"
OUT = ROOT / "revision/kd_unified"
DATASETS = ["isic2018", "acdc", "synapse"]
METRICS = ex.METRICS
SESOI = ex.SESOI
REFERENCES = {"decoder": "decoder-only", "unet": "U-Net", "full": "BDS-Lite"}
CELL = {"decoder": "M2", "unet": "M0", "full": "M6"}
CLASSES = {"isic2018": 2, "acdc": 4, "synapse": 9}


def kd_path(ds, seed, split="test"):
    return RUNS / f"{ds}_kd_band_seed{seed}/per_case_{split}.csv"


def ref_path(ds, arm, seed, split="test"):
    if ds == "isic2018":
        return ex.isic_path(arm, seed) if split == "test" else ma.validation_path(arm, seed)
    if split == "val":
        if arm != "decoder":
            return None
        suffix = "_unweighted" if ds == "acdc" else ""
        return RUNS / f"{ds}_m2_aux_boundary_only{suffix}_seed{seed}/per_case_val.csv"
    return ex.cell_path(CELL[arm], ds, seed, False)


def ref_seeds(ds, matched=False):
    return [1, 2, 3] if ds != "isic2018" or matched else list(range(1, 11))


def reading(low, high, sesoi):
    if low > sesoi:
        return "exploratory relevant gain"
    if high < -sesoi:
        return "exploratory relevant harm"
    if -sesoi < low < high < sesoi:
        return "no relevant effect"
    if low > 0:
        return "exploratory gain, size unclear"
    if high < 0:
        return "exploratory harm, size unclear"
    return "not established"


def run_means(values, metric):
    return [float(np.mean([v for v in r[metric].values() if math.isfinite(v)])) for r in values]


def label_values(path: Path, ds: str, label: int) -> dict:
    """unit_values restricted to one label: metric -> unit -> mean over that label's rows."""
    units = defaultdict(lambda: defaultdict(list))
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if int(row.get("label", 1)) != label:
                continue
            for metric in METRICS:
                value = fs._value(row, metric)
                if math.isfinite(value):
                    units[metric][fs.cluster_of(ds, row["id"])].append(value)
    return {m: {u: float(np.mean(v)) for u, v in units[m].items()} for m in METRICS}


def rates(paths) -> dict:
    out = {"missed_structure": [], "false_alarm": []}
    for path in paths:
        rows = list(csv.DictReader(path.open()))
        present = [r for r in rows if r["empty_target"] == "False"]
        absent = [r for r in rows if r["empty_target"] == "True"]
        out["missed_structure"].append(
            sum(r["empty_prediction"] == "True" for r in present) / len(present)
            if present
            else math.nan
        )
        out["false_alarm"].append(
            sum(r["empty_prediction"] == "False" for r in absent) / len(absent)
            if absent
            else math.nan
        )
    return {
        k: (float(np.mean(v)) if not all(math.isnan(x) for x in v) else None)
        for k, v in out.items()
    }


def history_cost(folder: Path) -> dict:
    history = json.loads((folder / "history.json").read_text())
    seconds = [h["seconds"] for h in history]
    out = dict(
        run=folder.name,
        epochs=len(history),
        median_epoch_seconds=statistics.median(seconds),
        sum_epoch_hours=sum(seconds) / 3600,
    )
    if "peak_allocated_mib" in history[0]:
        out["peak_allocated_mib"] = max(h["peak_allocated_mib"] for h in history)
    telemetry = folder / "gpu_telemetry.csv"
    if telemetry.exists():
        temps = [
            int(x.split(",")[1])
            for x in telemetry.read_text().splitlines()
            if len(x.split(",")) > 1 and x.split(",")[1].strip().isdigit()
        ]
        out["max_temperature_c"] = max(temps) if temps else None
        out["telemetry_hours"] = len(temps) * 10 / 3600
    return out


def deployed_parameters(folder: Path) -> int:
    state = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)["model"]
    return int(
        sum(
            v.numel()
            for k, v in state.items()
            if not k.startswith("boundary_decoder")
            and v.dtype.is_floating_point
            and "running" not in k
        )
    )


def thermal_resumes(dataset: str) -> dict:
    files = [
        ROOT / "revision/kd_comparator/supervision/events.jsonl",
        ROOT / "revision/kd_multiclass/supervision/events.jsonl",
    ]
    counts = defaultdict(int)
    for f in files:
        if f.exists():
            for line in f.read_text().splitlines():
                event = json.loads(line)
                evidence = event.get("evidence")
                if event["event"] == "guarded_resume_pending_cooling" and isinstance(
                    evidence, dict
                ):
                    if evidence["run"].startswith(dataset):
                        counts[evidence["run"]] += 1
    return dict(counts)


def pooled(rows):
    w = np.array([1 / r["se"] ** 2 for r in rows])
    d = np.array([r["delta"] for r in rows])
    est = float((w * d).sum() / w.sum())
    se = float(1 / math.sqrt(w.sum()))
    q = float((w * (d - est) ** 2).sum())
    return dict(
        delta=est,
        se=se,
        low=est - 1.96 * se,
        high=est + 1.96 * se,
        cochran_q=q,
        q_p=float(stats.chi2.sf(q, len(rows) - 1)) if len(rows) > 1 else None,
        k=len(rows),
    )


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def analyse(ds: str) -> dict:
    kd = {s: ex.unit_values(kd_path(ds, s), ds) for s in (1, 2, 3)}
    out = dict(dataset=ds, contrasts=[], per_seed=[], per_class=[], validation=[])
    refs = {}
    for arm in REFERENCES:
        for matched in [False, True] if ds == "isic2018" else [False]:
            seeds = ref_seeds(ds, matched)
            values = [ex.unit_values(ref_path(ds, arm, s), ds) for s in seeds]
            refs[(arm, matched)] = (seeds, values)
            for metric in METRICS:
                c = ex.contrast(list(kd.values()), values, metric)
                out["contrasts"].append(
                    dict(
                        dataset=ds,
                        comparison=f"KD-band - {REFERENCES[arm]}",
                        reference_seeds=f"{seeds[0]}-{seeds[-1]}",
                        seed_matched=matched or ds != "isic2018",
                        metric=metric,
                        primary=arm == "decoder" and metric == "boundary_f1" and not matched,
                        reading=reading(c["low"], c["high"], SESOI[metric]),
                        **c,
                    )
                )
    for metric in METRICS:
        for s, v in kd.items():
            out["per_seed"].append(
                dict(
                    dataset=ds,
                    arm="KD-band",
                    seed=s,
                    metric=metric,
                    value=run_means([v], metric)[0],
                )
            )
        for arm in REFERENCES:
            seeds, values = refs[(arm, False)]
            for s, v in zip(seeds, values, strict=True):
                out["per_seed"].append(
                    dict(
                        dataset=ds,
                        arm=REFERENCES[arm],
                        seed=s,
                        metric=metric,
                        value=run_means([v], metric)[0],
                    )
                )
    # Seed consistency against decoder-only, Boundary F1 and DSC
    out["consistency"] = {}
    for metric in ["boundary_f1", "dsc"]:
        k = run_means(list(kd.values()), metric)
        d = run_means(refs[("decoder", False)][1], metric)
        out["consistency"][metric] = dict(
            kd_runs=k,
            decoder_runs=d,
            kd_above_decoder_mean=sum(x > np.mean(d) for x in k),
            kd_min_minus_decoder_max=min(k) - max(d),
            kd_max_minus_decoder_min=max(k) - min(d),
        )
    if ds != "isic2018":
        for label in range(1, CLASSES[ds]):
            kd_l = [label_values(kd_path(ds, s), ds, label) for s in (1, 2, 3)]
            for arm in REFERENCES:
                ref_l = [label_values(ref_path(ds, arm, s), ds, label) for s in (1, 2, 3)]
                for metric in ["dsc", "boundary_f1", "hd95"]:
                    c = ex.contrast(kd_l, ref_l, metric)
                    out["per_class"].append(
                        dict(
                            dataset=ds,
                            label=label,
                            comparison=f"KD-band - {REFERENCES[arm]}",
                            metric=metric,
                            delta=c["delta"],
                            low=c["low"],
                            high=c["high"],
                            arm_a_mean=c["arm_a_mean"],
                            arm_b_mean=c["arm_b_mean"],
                        )
                    )
    val_kd = [ex.unit_values(kd_path(ds, s, "val"), ds) for s in (1, 2, 3)]
    val_ref = [ex.unit_values(ref_path(ds, "decoder", s, "val"), ds) for s in ref_seeds(ds)]
    for metric in METRICS:
        c = ex.contrast(val_kd, val_ref, metric)
        out["validation"].append(
            dict(
                dataset=ds,
                comparison="KD-band - decoder-only (validation)",
                metric=metric,
                reading=reading(c["low"], c["high"], SESOI[metric]),
                **c,
            )
        )
    out["rates"] = {"KD-band": rates([kd_path(ds, s) for s in (1, 2, 3)])}
    for arm in REFERENCES:
        out["rates"][REFERENCES[arm]] = rates([ref_path(ds, arm, s) for s in refs[(arm, False)][0]])
    teacher = RUNS / f"{ds}_kd_teacher_seed100"
    out["teacher_test"] = {
        m: run_means([ex.unit_values(teacher / "per_case_test.csv", ds)], m)[0] for m in METRICS
    }
    decoder_folder = (
        (RUNS / "isic2018_m2_aux_boundary_only_seed1")
        if ds == "isic2018"
        else ref_path(ds, "decoder", 1).parent
    )
    costs = [history_cost(teacher)] + [
        history_cost(RUNS / f"{ds}_kd_band_seed{s}") for s in (1, 2, 3)
    ]
    costs.append(
        dict(
            history_cost(decoder_folder),
            run=decoder_folder.name + " (existing decoder-only reference)",
        )
    )
    out["costs"] = costs
    out["deployed_parameters"] = dict(
        kd_band=deployed_parameters(RUNS / f"{ds}_kd_band_seed1"),
        decoder_only=deployed_parameters(decoder_folder),
    )
    out["thermal_resumes"] = thermal_resumes(ds)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()
    results, missing = {}, {}
    for ds in args.datasets:
        teacher = RUNS / f"{ds}_kd_teacher_seed100/test_metrics.json"
        students = [RUNS / f"{ds}_kd_band_seed{s}/test_metrics.json" for s in (1, 2, 3)]
        if not teacher.exists() or not all(p.exists() for p in students):
            missing[ds] = "test evaluation not completed"
            continue
        integrity = check(ds, allow_test=True)
        if not integrity["passed"]:
            missing[ds] = f"post-test integrity failed: {integrity['problems']}"
            continue
        results[ds] = analyse(ds)
    primary = [r for ds in results for r in results[ds]["contrasts"] if r["primary"]]
    for row, p in zip(primary, holm([r["p"] for r in primary]), strict=True):
        row["p_holm_over_datasets"] = p
    pooled_rows = []
    for metric in METRICS:
        rows = [
            r
            for ds in results
            for r in results[ds]["contrasts"]
            if r["comparison"] == "KD-band - decoder-only"
            and r["metric"] == metric
            and not (r["dataset"] == "isic2018" and r["seed_matched"])
        ]
        if len(rows) >= 2:
            pooled_rows.append(
                dict(
                    metric=metric,
                    datasets=[r["dataset"] for r in rows],
                    **pooled(
                        [
                            dict(delta=r["delta"], se=math.hypot(r["se_run"], r["se_unit"]))
                            for r in rows
                        ]
                    ),
                )
            )
    OUT.mkdir(parents=True, exist_ok=True)
    payload = dict(
        status="exploratory; three D+KD-band seeds per dataset; no confirmed effects",
        missing=missing,
        results=results,
        pooled_kd_minus_decoder=pooled_rows,
    )
    (OUT / "unified_results.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    for name, key in [
        ("contrasts.csv", "contrasts"),
        ("per_seed.csv", "per_seed"),
        ("per_class.csv", "per_class"),
    ]:
        rows = [r for ds in results for r in results[ds][key]]
        if rows:
            with (OUT / name).open("w") as stream:
                fields = list(dict.fromkeys(k for r in rows for k in r))
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    cost_rows = [dict(dataset=ds, **c) for ds in results for c in results[ds]["costs"]]
    if cost_rows:
        with (OUT / "costs.csv").open("w") as stream:
            fields = list(dict.fromkeys(k for r in cost_rows for k in r))
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(cost_rows)
    lines = [
        "# External-teacher D+KD-band: unified results (exploratory)",
        "",
        f"Missing: {missing or 'none'}",
        "",
        "| Dataset | Comparison | Ref seeds | Metric | KD-band | Reference | Delta [95% CI] | p | Holm (3 datasets) | Reading |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ds in results:
        for r in results[ds]["contrasts"]:
            lines.append(
                f"| {ds} | {r['comparison']} | {r['reference_seeds']} | {r['metric']} | {r['arm_a_mean']:.4f} | {r['arm_b_mean']:.4f} | "
                f"{r['delta']:+.4f} [{r['low']:+.4f}, {r['high']:+.4f}] | {r['p']:.3f} | {r.get('p_holm_over_datasets', float('nan')):.3f} | {r['reading']} |"
            )
    lines += [
        "",
        "Positive delta favours D+KD-band; HD95/ASSD signs reversed (positive = smaller distance).",
        "",
        "## Pooled KD-band - decoder-only (inverse variance, exploratory)",
        "",
        "| Metric | Datasets | Delta [95% CI] | Cochran Q (p) |",
        "|---|---|---|---|",
    ]
    for r in pooled_rows:
        lines.append(
            f"| {r['metric']} | {', '.join(r['datasets'])} | {r['delta']:+.4f} [{r['low']:+.4f}, {r['high']:+.4f}] | {r['cochran_q']:.2f} ({r['q_p']:.3f}) |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
