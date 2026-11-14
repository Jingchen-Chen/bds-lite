"""Independently verify completed scientific evidence without changing experiment outputs.

CPU-only case-table arithmetic; no model execution, training, E3 or Experiment C reads.
Only path resolution is imported from the registered analysis. Statistical calculations
are implemented here. Final audit receipts are written under revision/evidence_freeze/.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
import family_statistics as path_rules  # noqa: E402

METRICS = ("dsc", "iou", "boundary_f1", "hd95", "assd")
CHECKS: list[dict] = []
SOURCES: set[Path] = set()
TOLERANCE = 1e-10
COUNTS: dict[str, int] = {}


def table(path: str | Path) -> pd.DataFrame:
    p = ROOT / path
    SOURCES.add(p)
    return pd.read_csv(p)


def document(path: str | Path) -> dict | list:
    p = ROOT / path
    SOURCES.add(p)
    return json.loads(p.read_text())


@cache
def values(path: str | Path, dataset: str, label: int | None = None) -> pd.DataFrame:
    f = table(path)
    if "label" not in f:
        f["label"] = 1
    if f.duplicated(["id", "label"]).any():
        raise ValueError(f"duplicate case/label rows: {path}")
    if label is not None:
        f = f[f.label == label].copy()
    both = f.empty_prediction.astype(str).str.lower().eq("true") & f.empty_target.astype(
        str
    ).str.lower().eq("true")
    f.loc[both, list(METRICS)] = np.nan
    sample = f.groupby("id", sort=True)[list(METRICS)].mean()
    if dataset != "isic2018":
        unit = sample.index.str.extract(r"^(patient\d+|case\d+)", expand=False)
        if unit.isna().any():
            raise ValueError(f"unknown case identities: {path}")
        sample = sample.groupby(unit, sort=True).mean()
    return sample


def matrices(a: list, b: list, metric: str) -> tuple[np.ndarray, np.ndarray]:
    common = set.intersection(*(set(f.index[f[metric].notna()]) for f in a + b))
    units = sorted(common)
    sign = -1 if metric in ("hd95", "assd") else 1
    return tuple(sign * np.array([f.loc[units, metric] for f in arm]) for arm in (a, b))


def contrast(a: list, b: list, metric: str) -> dict:
    x, y = matrices(a, b, metric)
    nx, ny = len(x), len(y)
    rx, ry = x.mean(1), y.mean(1)
    vx, vy = rx.var(ddof=1) / nx, ry.var(ddof=1) / ny
    unit_variance = (x.mean(0) - y.mean(0)).var(ddof=1) / x.shape[1]
    se = math.sqrt(vx + vy + unit_variance)
    df = (vx + vy) ** 2 / (vx**2 / (nx - 1) + vy**2 / (ny - 1))
    delta = float(rx.mean() - ry.mean())
    half = float(stats.t.ppf(0.975, df) * se)
    sign = -1 if metric in ("hd95", "assd") else 1
    return dict(
        delta=delta,
        low=delta - half,
        high=delta + half,
        p=float(2 * stats.t.sf(abs(delta / se), df)),
        se=se,
        df=df,
        arm_a_mean=float(sign * rx.mean()),
        arm_b_mean=float(sign * ry.mean()),
        arm_a_sd=float(rx.std(ddof=1)),
        arm_b_sd=float(ry.std(ddof=1)),
    )


def compare(group: str, claim: str, actual: dict, expected: dict, fields: tuple) -> None:
    for field in fields:
        measured, recorded = float(actual[field]), float(expected[field])
        error = abs(measured - recorded)
        CHECKS.append(
            dict(
                group=group,
                claim=claim,
                quantity=field,
                recomputed=measured,
                recorded=recorded,
                absolute_error=error,
                tolerance=TOLERANCE,
                passed=bool(math.isfinite(error) and error <= TOLERANCE),
            )
        )


def check_holm(group: str, family: list, expected_key: str) -> None:
    running = 0.0
    for rank, (name, got, expected) in enumerate(sorted(family, key=lambda item: item[1]["p"])):
        running = max(running, min(1.0, (len(family) - rank) * got["p"]))
        compare(group, name, {"holm": running}, {"holm": expected[expected_key]}, ("holm",))


def isic_path(arm: str, seed: int) -> Path:
    if arm in ("unet", "full") and seed <= 3:
        short = "U" if arm == "unet" else "F"
        return ROOT / f"revision/official_isic/test/{short}_seed{seed}/per_case.csv"
    name = dict(unet="m0_unet", full="m6_full", decoder="m2_aux_boundary_only", gsl="gsl")[arm]
    return ROOT / f"results/revision_runs/isic2018_{name}_seed{seed}/per_case_test.csv"


def cell_path(cell: str, dataset: str, seed: int) -> Path:
    if cell == "M7":
        return (
            ROOT
            / f"results/revision_runs/{dataset}_m7_decoder_surface_seed{seed}/per_case_test.csv"
        )
    p = path_rules.per_case_path(cell, dataset, seed)
    if p is None:
        raise FileNotFoundError(f"completed cell missing: {cell} {dataset} {seed}")
    return p


def acdc_arms() -> dict:
    arms = {}
    for arm in ("decoder", "kd_band", "kd_full", "kd_band_peer"):
        base = (
            "results/acdc_confirm_eval/acdc_confirm"
            if arm in ("decoder", "kd_band")
            else "results/followup_eval/acdc_followup"
        )
        arms[arm] = [
            values(f"{base}_{arm}_seed{s}/per_case_official_test.csv", "acdc")
            for s in range(201, 218)
        ]
    return arms


def confirmation_followup(arms: dict) -> None:
    frame = table("revision/followup/results/contrasts.csv")
    pairs = dict(
        S3=("kd_band", "decoder"),
        C1=("kd_band", "kd_full"),
        C2=("kd_full", "decoder"),
        C3=("kd_band_peer", "decoder"),
        C4=("kd_band", "kd_band_peer"),
    )
    for metric, block in frame.groupby("metric", sort=False):
        family = []
        for row in block.to_dict("records"):
            first, second = pairs[row["id"]]
            got = contrast(arms[first], arms[second], metric)
            compare("ACDC", f"{row['id']} {metric}", got, row, ("delta", "low", "high", "p"))
            if row["id"] != "S3":
                family.append((f"{row['id']} {metric}", got, row))
        check_holm("ACDC Holm", family, "p_holm")
    COUNTS["acdc_contrasts"] = len(frame)
    COUNTS["acdc_holm"] = 20
    primary = document("revision/acdc_confirm/results/results.json")["primary"]["contrast"]
    compare(
        "ACDC primary JSON",
        "Boundary F1",
        contrast(arms["kd_band"], arms["decoder"], "boundary_f1"),
        primary,
        ("delta", "low", "high", "p"),
    )
    perclass = table("revision/acdc_confirm/results/per_class.csv")
    for row in perclass.to_dict("records"):
        label = {"RV": 1, "myocardium": 2, "LV": 3}[row["structure"]]
        a, b = [
            [
                values(
                    f"results/acdc_confirm_eval/acdc_confirm_{arm}_seed{s}/per_case_official_test.csv",
                    "acdc",
                    label,
                )
                for s in range(201, 218)
            ]
            for arm in ("kd_band", "decoder")
        ]
        compare(
            "ACDC per-class",
            f"{row['structure']} {row['metric']}",
            contrast(a, b, row["metric"]),
            row,
            ("delta", "low", "high", "p"),
        )
    COUNTS["acdc_per_class_contrasts"] = len(perclass)


def isic_extension() -> None:
    frame = table("revision/tables/families/expansion_contrasts.csv")
    for metric, block in frame.groupby("metric", sort=False):
        for prefix, seeds in (("all", range(1, 11)), ("fresh", range(4, 11))):
            family = []
            for row in block.to_dict("records"):
                a, b = [
                    [values(isic_path(row[arm], s), "isic2018") for s in seeds]
                    for arm in ("first", "second")
                ]
                got = contrast(a, b, metric)
                expected = {k: row[f"{prefix}_{k}"] for k in ("delta", "low", "high", "p")}
                name = f"{prefix} {row['first']}-{row['second']} {metric}"
                compare("ISIC extension", name, got, expected, ("delta", "low", "high", "p"))
                family.append((name, got, row))
            check_holm("ISIC Holm", family, f"{prefix}_p_holm")
    COUNTS["isic_contrasts"] = 2 * len(frame)
    COUNTS["isic_holm"] = 2 * len(frame)


def factorial() -> None:
    weights = {
        "interaction": dict(M7=1, M2=-1, M1=-1, M0=1),
        "surface without decoder": dict(M1=1, M0=-1),
        "surface with decoder": dict(M7=1, M2=-1),
        "decoder without surface": dict(M2=1, M0=-1),
        "decoder with surface": dict(M7=1, M1=-1),
        "distillation+gate without surface": dict(M3=1, M2=-1),
        "distillation+gate with surface": dict(M6=1, M7=-1),
        "distillation+gate interaction with surface": dict(M6=1, M7=-1, M3=-1, M2=1),
    }
    frame = table("revision/tables/families/expansion_interaction.csv")
    frame = frame[frame.seeds == "1-3"]
    for (effect, metric), block in frame.groupby(["effect", "metric"], sort=False):
        w = weights[effect]
        parts = []
        for row in block[~block.dataset.str.startswith("pooled")].to_dict("records"):
            cells = {
                c: [values(cell_path(c, row["dataset"], s), row["dataset"]) for s in (1, 2, 3)]
                for c in w
            }
            ids = sorted(
                set.intersection(
                    *(set(f.index[f[metric].notna()]) for a in cells.values() for f in a)
                )
            )
            a = {c: np.array([f.loc[ids, metric] for f in arm]) for c, arm in cells.items()}
            sign = -1 if metric in ("hd95", "assd") else 1
            estimate = float(sign * sum(w[c] * x.mean() for c, x in a.items()))
            df = sum(len(x) - 1 for x in a.values())
            pooled = sum(np.sum((x.mean(1) - x.mean()) ** 2) for x in a.values()) / df
            unit = sum(w[c] * x.mean(0) for c, x in a.items())
            variance = pooled * sum(v * v / 3 for v in w.values()) + unit.var(ddof=1) / len(ids)
            se = math.sqrt(variance)
            half = stats.t.ppf(0.975, df) * se
            got = dict(estimate=estimate, se=se, df=df, low=estimate - half, high=estimate + half)
            compare("stage 1", f"{effect} {row['dataset']} {metric}", got, row, tuple(got))
            parts.append(got)
        iw = np.array([1 / p["se"] ** 2 for p in parts])
        est = float(np.dot(iw, [p["estimate"] for p in parts]) / iw.sum())
        se = math.sqrt(1 / iw.sum())
        expected = block[block.dataset.str.startswith("pooled")].iloc[0].to_dict()
        compare(
            "stage 1 pooled",
            f"{effect} {metric}",
            dict(estimate=est, low=est - 1.96 * se, high=est + 1.96 * se),
            expected,
            ("estimate", "low", "high"),
        )
    COUNTS["stage1_dataset_contrasts"] = len(frame[~frame.dataset.str.startswith("pooled")])
    COUNTS["stage1_pooled_contrasts"] = len(frame[frame.dataset.str.startswith("pooled")])


def exploratory_kd() -> None:
    frame = table("revision/kd_unified/contrasts.csv")
    frame = frame[
        (frame.comparison == "KD-band - decoder-only")
        & ~((frame.dataset == "isic2018") & frame.seed_matched)
    ]
    family = []
    for row in frame.to_dict("records"):
        ds = row["dataset"]
        a = [
            values(f"results/revision_runs/{ds}_kd_band_seed{s}/per_case_test.csv", ds)
            for s in (1, 2, 3)
        ]
        seeds = range(1, 11) if ds == "isic2018" else (1, 2, 3)
        b = [
            values(isic_path("decoder", s) if ds == "isic2018" else cell_path("M2", ds, s), ds)
            for s in seeds
        ]
        got = contrast(a, b, row["metric"])
        compare("exploratory KD", f"{ds} {row['metric']}", got, row, ("delta", "low", "high", "p"))
        if row["metric"] == "boundary_f1":
            family.append((ds, got, row))
    check_holm("exploratory KD Holm", family, "p_holm_over_datasets")
    balanced = table("revision/kd_posthoc_balanced/contrasts.csv")
    for analysis, block in balanced.groupby("analysis", sort=False):
        family = []
        for row in block.to_dict("records"):
            ds = row["dataset"]
            a = [
                values(f"results/revision_runs/{ds}_kd_band_seed{s}/per_case_test.csv", ds)
                for s in (1, 2, 3)
            ]
            first_seed, last_seed = map(int, row["reference_seeds"].split("-"))
            b = [
                values(isic_path("decoder", s) if ds == "isic2018" else cell_path("M2", ds, s), ds)
                for s in range(first_seed, last_seed + 1)
            ]
            got = contrast(a, b, "boundary_f1")
            compare(analysis, ds, got, row, ("delta", "low", "high", "p"))
            family.append((ds, got, row))
        check_holm(f"{analysis} Holm", family, "p_holm_over_datasets")
    COUNTS["original_exploratory_kd_contrasts"] = len(frame)
    COUNTS["balanced_kd_contrasts"] = int(
        (balanced.analysis == "post hoc balanced presentation").sum()
    )
    COUNTS["retained_original_kd_display_contrasts"] = (
        len(balanced) - COUNTS["balanced_kd_contrasts"]
    )


def absolute_scores() -> None:
    methods = {"M0", "M1", "M2", "M3", "M4", "M5", "M6", "GSL", "C1", "C3", "C4"}
    frame = table("revision/tables/families/method_summary.csv")
    n = 0
    for row in frame[frame.method.isin(methods)].to_dict("records"):
        per_seed = []
        for seed in (1, 2, 3):
            p = cell_path(row["method"], row["dataset"], seed)
            f = table(p)
            if "empty_prediction" in f:
                both = f.empty_prediction.astype(str).str.lower().eq(
                    "true"
                ) & f.empty_target.astype(str).str.lower().eq("true")
                f.loc[both, list(METRICS)] = np.nan
            per_seed.append(f.groupby("id")[list(METRICS)].mean().mean())
        a = pd.DataFrame(per_seed)
        for metric in METRICS:
            compare(
                "absolute mean/SD",
                f"{row['dataset']} {row['method']} {metric}",
                dict(mean=a[metric].mean(), sd=a[metric].std(ddof=1)),
                dict(mean=row[f"{metric}_mean"], sd=row[f"{metric}_sd"]),
                ("mean", "sd"),
            )
            n += 1
    COUNTS["absolute_metric_cells"] = n


def reference_estimators(arms: dict) -> dict:
    frame = table("revision/followup/results/posthoc_reference_estimators.csv")
    rng = np.random.default_rng(20261109)
    fractions = {}
    for row in frame.to_dict("records"):
        x, y = matrices(arms[row["first"]], arms[row["second"]], row["metric"])
        nx, u = x.shape
        ny = len(y)

        def terms(z):
            n, units = z.shape
            residual = z - z.mean(1)[:, None] - z.mean(0)[None, :] + z.mean()
            return units * z.mean(1).var(ddof=1), np.sum(residual**2) / ((n - 1) * (units - 1))

        rx, ex = terms(x)
        ry, ey = terms(y)
        eps = (ex * (nx - 1) + ey * (ny - 1)) / (nx + ny - 2)
        interaction = (x.mean(0) - y.mean(0)).var(ddof=1) - eps * (1 / nx + 1 / ny)
        v = np.array([rx / (nx * u), ry / (ny * u), max(0, interaction) / u])
        se = math.sqrt(v.sum())
        df = v.sum() ** 2 / np.sum(v * v / np.array([nx - 1, ny - 1, u - 1]))
        delta = float(x.mean() - y.mean())
        half = stats.t.ppf(0.975, df) * se
        bs = np.empty(10000)
        for i in range(10000):
            cols = rng.integers(0, u, u)
            aa, bb = rng.integers(0, nx, nx), rng.integers(0, ny, ny)
            bs[i] = x[np.ix_(aa, cols)].mean() - y[np.ix_(bb, cols)].mean()
        low, high = np.percentile(bs, [2.5, 97.5])
        got = dict(
            mixed_low=delta - half,
            mixed_high=delta + half,
            mixed_p=2 * stats.t.sf(abs(delta / se), df),
            bootstrap_low=low,
            bootstrap_high=high,
        )
        compare("post hoc references", f"{row['id']} {row['metric']}", got, row, tuple(got))
        if row["metric"] == "boundary_f1":
            fractions[row["id"]] = float(eps * (1 / nx + 1 / ny) / u / row["prespecified_se"] ** 2)
    COUNTS["posthoc_reference_rows"] = len(frame)
    return fractions


def historical_and_selection() -> dict:
    split = document("splits/isic2018_split.json")
    f = table("results/midterm_q2_rescue/analysis/per_case_metrics.csv")
    ids = set(f.loc[f.dataset == "isic2018", "sample_id"])
    overlap = {
        part: len(ids & {Path(x).name for x in split[part]}) for part in ("train", "val", "test")
    }
    expected = dict(train=358, val=84, test=77)
    compare(
        "historical contamination", "519 evaluated identities", overlap, expected, tuple(expected)
    )
    selections = {}
    for name in ("synapse_m2_aux_boundary_only_seed1", "acdc_confirm_decoder_seed201"):
        base = Path("results/revision_runs") / name
        f = table(base / "per_case_val.csv")
        both = f.empty_prediction.astype(str).str.lower().eq("true") & f.empty_target.astype(
            str
        ).str.lower().eq("true")
        record = document(base / "final_metrics.json")
        compare(
            "selection rule",
            name,
            dict(all_row_dsc=f.dsc.mean()),
            dict(all_row_dsc=record["best_val_dsc"]),
            ("all_row_dsc",),
        )
        selections[name] = dict(
            rows=len(f),
            both_empty=int(both.sum()),
            all_row_dsc=float(f.dsc.mean()),
            non_both_empty_row_dsc=float(f.loc[~both, "dsc"].mean()),
        )
    for arm in (
        "acdc_confirm_decoder",
        "acdc_confirm_kd_band",
        "acdc_followup_kd_full",
        "acdc_followup_kd_band_peer",
    ):
        for seed in range(201, 218):
            base = Path("results/revision_runs") / f"{arm}_seed{seed}"
            history = document(base / "history.json")
            final = document(base / "final_metrics.json")
            if [e["epoch"] for e in history] != list(range(1, 151)):
                raise ValueError(f"epoch coverage mismatch: {base}")
            v = np.array([e["val"]["val_dsc"] for e in history])
            if not np.isfinite(v).all():
                raise ValueError(f"nonfinite selection metric: {base}")
            compare(
                "history selection",
                base.name,
                dict(epoch=int(v.argmax()) + 1, best=float(v.max())),
                dict(epoch=final["selected_epoch"], best=final["best_val_dsc"]),
                ("epoch", "best"),
            )
    COUNTS["student_history_records"] = 68
    return dict(
        historical_evaluated_ids=len(ids),
        historical_overlap=overlap,
        validation_selection=selections,
    )


def artifact_hashes() -> dict:
    checked = 0
    for base, name in (
        ("results/acdc_confirm_eval", "evaluation_record.json"),
        ("results/followup_eval", "evaluation_record_e1e2.json"),
    ):
        record = document(Path(base) / name)
        for run in record["runs"]:
            for artifact, digest in run["artifacts"].items():
                p = ROOT / base / run["run"] / artifact
                SOURCES.add(p)
                if hashlib.file_digest(p.open("rb"), "sha256").hexdigest() != digest:
                    raise ValueError(f"artifact hash mismatch: {p}")
                checked += 1
            p = ROOT / "results/revision_runs" / run["run"] / "best.pt"
            SOURCES.add(p)
            if hashlib.file_digest(p.open("rb"), "sha256").hexdigest() != run["best_sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {p}")
    return dict(evaluation_artifacts=checked, mismatches=0)


def supplemental_evidence() -> dict:
    """Recompute saved diagnostic aggregations; never rerun profiling or model probes."""
    costs = table("revision/kd_unified/costs.csv")
    for row in costs.to_dict("records"):
        name = row["run"].split(" (")[0]
        history = document(f"results/revision_runs/{name}/history.json")
        seconds = [h["seconds"] for h in history]
        compare(
            "training cost",
            name,
            {
                "epochs": len(history),
                "median_epoch_seconds": np.median(seconds),
                "sum_epoch_hours": sum(seconds) / 3600,
            },
            row,
            ("epochs", "median_epoch_seconds", "sum_epoch_hours"),
        )
    COUNTS["training_cost_runs"] = len(costs)
    probe_rows = table("revision/followup/results/probes.csv")
    for row in probe_rows.to_dict("records"):
        prefix = "acdc_confirm" if row["arm"] in ("decoder", "kd_band") else "acdc_followup"
        name = f"{prefix}_{row['arm']}_seed{row['seed']}"
        raw = document(f"results/revision_runs/{name}/kd_probes.json")
        hit = [r for r in raw if r["epoch"] == row["epoch"]]
        if len(hit) != 1:
            raise ValueError(f"probe epoch identity mismatch: {name} {row['epoch']}")
        for key in ("loss_kd", "kd_norm_ratio", "kd_cosine", "boundary_norm_ratio"):
            expected, got = row[key], hit[0][key]
            if pd.isna(expected):
                if got is not None:
                    raise ValueError(f"undefined probe mismatch: {name} {key}")
                continue
            compare("followup probe", f"{name} epoch {row['epoch']}", {key: got}, row, (key,))
    COUNTS["followup_probe_records"] = len(probe_rows)
    gradients = table("revision/tables/gradient_summary.csv")
    windows = {"all": (1, 150), "early": (1, 10), "late": (101, 150)}
    for row in gradients.to_dict("records"):
        raw = document(
            f"results/revision_runs/{row['dataset']}_wp5_full_seed1/gradient_probes.json"
        )
        low, high = windows[row["window"]]
        picked = [r for r in raw if low <= r["epoch"] <= high]
        cosines = np.array(
            [r[row["term"] + "_cosine"] for r in picked if r[row["term"] + "_cosine"] is not None]
        )
        ratios = np.array(
            [
                r[row["term"] + "_norm_ratio"]
                for r in picked
                if r[row["term"] + "_norm_ratio"] is not None
            ]
        )
        got = dict(
            probes=len(picked),
            null_cosines=len(picked) - len(cosines),
            cosine_mean=cosines.mean(),
            cosine_median=np.median(cosines),
            conflict_fraction=(cosines < 0).mean(),
            norm_ratio_median=np.median(ratios),
        )
        compare(
            "gradient diagnostic",
            f"{row['dataset']} {row['term']} {row['window']}",
            got,
            row,
            tuple(got),
        )
    COUNTS["gradient_summary_rows"] = len(gradients)
    profile = table("revision/tables/resource_profile.csv")
    base = profile[(profile.dataset == "isic2018") & (profile.model == "U-Net")].iloc[0]
    full = profile[(profile.dataset == "isic2018") & (profile.model == "BDS-Lite (deployed)")].iloc[
        0
    ]
    mechanism = table("revision/tables/families/boundary_transfer_probe.csv")
    mechanism_means = {}
    for dataset, block in mechanism.groupby("dataset"):
        distilled = block[block.arm.isin(["M3", "M5", "M6"])]
        plain = block[block.arm.isin(["M2", "M4"])]
        mechanism_means[dataset] = {
            "distilled_head_q_s_mean": float(distilled.head_q_s.mean()),
            "undistilled_head_q_s_mean": float(plain.head_q_s.mean()),
            "distilled_probe_F_s_mean": float(distilled.probe_F_s.mean()),
            "undistilled_probe_F_s_mean": float(plain.probe_F_s.mean()),
            "scope": "Table aggregation only; original feature/pixel outputs not reconstructed.",
        }
    return {
        "resource_overhead_from_saved_profile": {
            "gpu_fp32_batch1_pct": float(
                100 * (full.gpu_fp32_b1_ms_median / base.gpu_fp32_b1_ms_median - 1)
            ),
            "gpu_fp32_batch16_pct": float(
                100 * (full.gpu_fp32_b16_ms_median / base.gpu_fp32_b16_ms_median - 1)
            ),
            "additional_deployed_parameters": int(full.params_deployed - base.params_deployed),
            "cuda_kernels_baseline": int(base.cuda_kernels_b1),
            "cuda_kernels_full": int(full.cuda_kernels_b1),
            "scope": "Recorded profile, not independently remeasured latency.",
        },
        "mechanistic_table_aggregations": mechanism_means,
        "verification_limits": [
            "No inference/profile rerun; feature, gate and pixel probes not re-created.",
            "Teacher cache generation, quality and weights not re-created here.",
            "Other sensitivities and literature remain source-reported unless separately audited.",
        ],
    }


def main() -> None:
    arms = acdc_arms()
    confirmation_followup(arms)
    isic_extension()
    factorial()
    exploratory_kd()
    absolute_scores()
    fractions = reference_estimators(arms)
    findings = historical_and_selection()
    hashes = artifact_hashes()
    supplemental = supplemental_evidence()
    environments = {
        name: importlib.util.find_spec(name).origin
        for name in ("numpy", "scipy", "torch", "segmentation_models_pytorch", "nibabel")
    }
    failures = [r for r in CHECKS if not r["passed"]]
    result = dict(
        status="passed" if not failures else "failed",
        checked_utc=datetime.now(timezone.utc).isoformat(),
        scope="Completed evidence only; no model execution, E3 or Experiment C outcomes.",
        counts=COUNTS,
        scalar_checks=len(CHECKS),
        mismatches=len(failures),
        max_absolute_error=max(r["absolute_error"] for r in CHECKS),
        tolerance=TOLERANCE,
        findings=findings,
        supplemental_evidence=supplemental,
        duplicated_residual_fraction_of_registered_se2=fractions,
        recorded_artifact_hashes=hashes,
        environment_resolution=environments,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
            for p in sorted(SOURCES)
        },
        coverage_limitations=[
            "Reproducing registered arithmetic is not a proof of statistical assumptions.",
            "Does not certify raw-data provenance, all historical runs or literature values.",
            "Local dependency resolution does not establish environment portability.",
        ],
    )
    out = ROOT / "revision/evidence_freeze"
    (out / "final_statistics_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "final_statistics_audit.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CHECKS[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(CHECKS)
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
