# ruff: noqa: E501
"""Frozen analysis for revision/EXPANSION_PROTOCOL.md. Committed before any expansion result exists.

Experiment A (ISIC2018, official test, four arms, ten seeds)
  Six contrasts on all five metrics. Boundary F1 is primary. Unit of replication: the training run.
  Delta = difference of arm means of run means, oriented so positive favors the first-named arm.
  se_run  = Welch standard error of the run means (test set held fixed).
  se_unit = SD over test images of the per-image difference of arm means, / sqrt(n images).
  se      = sqrt(se_run^2 + se_unit^2); t with Welch-Satterthwaite df; two-sided p from Delta/se.
  Computed for fresh seeds 4-10 (confirmation; Holm over the six contrasts within each metric) and
  for all seeds 1-10 (best estimate). Mann-Whitney on run means is a sensitivity check.

Experiment B (all datasets, seeds 1-3, 2x2 of decoder x surface loss)
  Interaction I = (decoder+surface - decoder) - (surface - U-Net), with simple effects and the
  effect of distillation+gate with and without the surface loss.
  se_run from the pooled within-cell variance; se_unit from the per-unit contrast. Pooled across
  datasets by inverse variance, with Cochran's Q.

Component model: additive main effects over the eight ablation cells, as in
narrative_component_effects.py, with the new cell and the ISIC2018 ten-seed arms.

Outputs in revision/tables/families/: expansion_contrasts.csv, expansion_interaction.csv,
expansion_components.csv. --dry-run uses seeds 1-3 only, substitutes BDS-Lite for the missing cell,
writes to the scratch path given, and exists only to test this code on data already known.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
import family_statistics as fs  # noqa: E402

RUNS = ROOT / "results/revision_runs"
OFFICIAL = ROOT / "revision/official_isic/test"
METRICS = fs.METRICS
SESOI = fs.SESOI
DATASETS = ["isic2018", "acdc", "synapse"]
FRESH, ALL = list(range(4, 11)), list(range(1, 11))
PAIRS = [
    ("decoder", "unet"),
    ("full", "unet"),
    ("full", "decoder"),
    ("gsl", "unet"),
    ("gsl", "full"),
    ("gsl", "decoder"),
]
NAMES = {
    "unet": "U-Net",
    "decoder": "U-Net + boundary decoder",
    "full": "BDS-Lite",
    "gsl": "U-Net + GSL",
}
COMPONENTS = ["decoder", "surface", "distill", "gate"]
CELLS = {
    "M0": (0, 0, 0, 0),
    "M1": (0, 1, 0, 0),
    "M2": (1, 0, 0, 0),
    "M7": (1, 1, 0, 0),
    "M3": (1, 0, 1, 1),
    "M4": (1, 1, 0, 1),
    "M5": (1, 1, 1, 0),
    "M6": (1, 1, 1, 1),
}


def isic_path(arm: str, seed: int) -> Path:
    if arm == "unet":
        if seed <= 3:
            return OFFICIAL / f"U_seed{seed}/per_case.csv"
        return RUNS / f"isic2018_m0_unet_seed{seed}/per_case_test.csv"
    if arm == "full":
        if seed <= 3:
            return OFFICIAL / f"F_seed{seed}/per_case.csv"
        return RUNS / f"isic2018_m6_full_seed{seed}/per_case_test.csv"
    if arm == "decoder":
        return RUNS / f"isic2018_m2_aux_boundary_only_seed{seed}/per_case_test.csv"
    return RUNS / f"isic2018_gsl_seed{seed}/per_case_test.csv"


def cell_path(cell: str, dataset: str, seed: int, dry: bool) -> Path:
    if cell == "M7":
        if dry:
            return fs.per_case_path("M6", dataset, seed)
        return RUNS / f"{dataset}_m7_decoder_surface_seed{seed}/per_case_test.csv"
    if cell == "M2" and dataset == "acdc" and not dry:
        return RUNS / f"acdc_m2_aux_boundary_only_unweighted_seed{seed}/per_case_test.csv"
    return fs.per_case_path(cell, dataset, seed)


def unit_values(path: Path, dataset: str) -> dict[str, dict[str, float]]:
    """metric -> unit -> value for one run: label macro per sample, then unit mean."""
    samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as stream:
        for row in csv.DictReader(stream):
            for metric in METRICS:
                samples[row["id"]][metric].append(fs._value(row, metric))
    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        units: dict[str, list[float]] = defaultdict(list)
        for sample, values in samples.items():
            arr = np.array(values[metric], float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                units[fs.cluster_of(dataset, sample)].append(float(arr.mean()))
        out[metric] = {u: float(np.mean(v)) for u, v in units.items()}
    return out


def classify(low: float, high: float, sesoi: float) -> str:
    if low > sesoi:
        return "relevant gain"
    if high < -sesoi:
        return "relevant harm"
    if -sesoi < low < high < sesoi:
        return "no relevant effect"
    if low > 0:
        return "gain, size unclear"
    if high < 0:
        return "harm, size unclear"
    return "undetermined"


def contrast(a: list[dict], b: list[dict], metric: str) -> dict:
    """a and b are lists of per-run metric->unit->value dictionaries."""
    sign = -1.0 if metric in fs.DISTANCE else 1.0
    units = sorted(set.intersection(*[set(r[metric]) for r in a + b]))
    ya = sign * np.array([[r[metric][u] for u in units] for r in a])
    yb = sign * np.array([[r[metric][u] for u in units] for r in b])
    ra, rb = ya.mean(1), yb.mean(1)
    delta = float(ra.mean() - rb.mean())
    va, vb = ra.var(ddof=1) / ra.size, rb.var(ddof=1) / rb.size
    se_run = math.sqrt(va + vb)
    per_unit = ya.mean(0) - yb.mean(0)
    se_unit = float(per_unit.std(ddof=1) / math.sqrt(len(units)))
    se = math.hypot(se_run, se_unit)
    df = (va + vb) ** 2 / (va**2 / (ra.size - 1) + vb**2 / (rb.size - 1))
    t = stats.t.ppf(0.975, df)
    low, high = delta - t * se, delta + t * se
    return dict(
        delta=delta,
        se_run=se_run,
        se_unit=se_unit,
        df=df,
        low=low,
        high=high,
        p=float(2 * stats.t.sf(abs(delta / se), df)),
        mann_whitney_p=float(stats.mannwhitneyu(ra, rb).pvalue),
        n_a=ra.size,
        n_b=rb.size,
        arm_a_mean=float(sign * ra.mean()),
        arm_b_mean=float(sign * rb.mean()),
        arm_a_sd=float(ra.std(ddof=1)),
        arm_b_sd=float(rb.std(ddof=1)),
        classification=classify(low, high, SESOI[metric]),
    )


def holm(rows: list[dict]) -> None:
    order = sorted(rows, key=lambda r: r["p"])
    running = 0.0
    for rank, row in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * row["p"]))
        row["p_holm"] = running


def status(fresh: dict, overall: dict) -> str:
    """Pre-registered reading of one contrast. Fresh seeds confirm; all seeds give the estimate."""
    excludes_zero = fresh["low"] > 0 or fresh["high"] < 0
    same_sign = fresh["delta"] * overall["delta"] > 0
    if fresh["p_holm"] < 0.05 and excludes_zero and same_sign:
        return "confirmed"
    if overall["low"] > 0 or overall["high"] < 0:
        return "supported by all seeds, not confirmed by fresh seeds"
    return "not established"


def experiment_a(seeds_fresh: list[int], seeds_all: list[int]) -> list[dict]:
    values = {
        (arm, seed): unit_values(isic_path(arm, seed), "isic2018")
        for arm in NAMES
        for seed in seeds_all
    }
    rows = []
    for metric in METRICS:
        block = []
        for first, second in PAIRS:
            pick = lambda arm, seeds: [values[(arm, s)] for s in seeds]  # noqa: E731
            fresh = contrast(pick(first, seeds_fresh), pick(second, seeds_fresh), metric)
            overall = contrast(pick(first, seeds_all), pick(second, seeds_all), metric)
            block.append((first, second, fresh, overall))
        holm([item[2] for item in block])
        holm([item[3] for item in block])
        for first, second, fresh, overall in block:
            row = dict(metric=metric, first=first, second=second, status=status(fresh, overall))
            row |= {f"fresh_{k}": v for k, v in fresh.items()}
            row |= {f"all_{k}": v for k, v in overall.items()}
            rows.append(row)
    return rows


def linear_contrast(
    cells: dict[str, list[dict]], weights: dict[str, float], metric: str
) -> tuple[float, float, float, int]:
    """Estimate, se_run (pooled within-cell), se_unit, residual df for sum_c w_c mean_c."""
    sign = -1.0 if metric in fs.DISTANCE else 1.0
    used = list(cells)
    units = sorted(set.intersection(*[set(r[metric]) for c in used for r in cells[c]]))
    run_means = {
        c: sign * np.array([[r[metric][u] for u in units] for r in cells[c]]) for c in used
    }
    estimate = sum(w * run_means[c].mean(1).mean() for c, w in weights.items())
    ss = sum(((m.mean(1) - m.mean(1).mean()) ** 2).sum() for m in run_means.values())
    df = sum(m.shape[0] - 1 for m in run_means.values())
    se_run = math.sqrt(ss / df * sum(w**2 / run_means[c].shape[0] for c, w in weights.items()))
    per_unit = sum(w * run_means[c].mean(0) for c, w in weights.items())
    se_unit = float(per_unit.std(ddof=1) / math.sqrt(len(units)))
    return float(estimate), se_run, se_unit, df


EFFECTS = {
    "interaction": {"M7": 1, "M2": -1, "M1": -1, "M0": 1},
    "surface without decoder": {"M1": 1, "M0": -1},
    "surface with decoder": {"M7": 1, "M2": -1},
    "decoder without surface": {"M2": 1, "M0": -1},
    "decoder with surface": {"M7": 1, "M1": -1},
}
TRANSFER = {
    "distillation+gate without surface": {"M3": 1, "M2": -1},
    "distillation+gate with surface": {"M6": 1, "M7": -1},
    "distillation+gate interaction with surface": {"M6": 1, "M7": -1, "M3": -1, "M2": 1},
}


def experiment_b(dry: bool, extra: dict[tuple[str, int], dict] | None) -> list[dict]:
    rows: list[dict] = []
    values = {}
    for dataset in DATASETS:
        for cell in CELLS:
            values[(dataset, cell)] = [
                unit_values(cell_path(cell, dataset, s, dry), dataset) for s in fs.SEEDS
            ]
    for metric in METRICS:
        for effects, label in ((EFFECTS, "2x2"), (TRANSFER, "transfer")):
            for name, weights in effects.items():
                part = []
                for dataset in DATASETS:
                    cells = {c: values[(dataset, c)] for c in weights}
                    estimate, se_run, se_unit, df = linear_contrast(cells, weights, metric)
                    se = math.hypot(se_run, se_unit)
                    t = stats.t.ppf(0.975, df)
                    low, high = estimate - t * se, estimate + t * se
                    part.append(
                        dict(
                            set=label,
                            effect=name,
                            metric=metric,
                            dataset=dataset,
                            seeds="1-3",
                            estimate=estimate,
                            se=se,
                            df=df,
                            low=low,
                            high=high,
                            classification=classify(low, high, SESOI[metric]),
                        )
                    )
                est = np.array([r["estimate"] for r in part])
                w = 1 / np.array([r["se"] ** 2 for r in part])
                mean, se = float((w * est).sum() / w.sum()), math.sqrt(1 / w.sum())
                q = float((w * (est - mean) ** 2).sum())
                low, high = mean - 1.96 * se, mean + 1.96 * se
                rows += part
                rows.append(
                    dict(
                        set=label,
                        effect=name,
                        metric=metric,
                        dataset=f"pooled (Q p={stats.chi2.sf(q, 2):.2f})",
                        seeds="1-3",
                        estimate=mean,
                        se=se,
                        df=float("nan"),
                        low=low,
                        high=high,
                        classification=classify(low, high, SESOI[metric]),
                    )
                )
    if extra:  # ISIC2018 sensitivity: U-Net and decoder cells use all ten seeds.
        cells = {c: values[("isic2018", c)] for c in ("M1", "M7")}
        cells["M0"] = [extra[("unet", s)] for s in ALL]
        cells["M2"] = [extra[("decoder", s)] for s in ALL]
        for metric in METRICS:
            for name, weights in EFFECTS.items():
                part = {c: cells[c] for c in weights}
                estimate, se_run, se_unit, df = linear_contrast(part, weights, metric)
                se = math.hypot(se_run, se_unit)
                t = stats.t.ppf(0.975, df)
                low, high = estimate - t * se, estimate + t * se
                rows.append(
                    dict(
                        set="2x2",
                        effect=name,
                        metric=metric,
                        dataset="isic2018",
                        seeds="U-Net and decoder 1-10",
                        estimate=estimate,
                        se=se,
                        df=df,
                        low=low,
                        high=high,
                        classification=classify(low, high, SESOI[metric]),
                    )
                )
    return rows


def components(dry: bool, extra: dict[tuple[str, int], dict] | None) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        runs: list[tuple[str, dict]] = []
        for cell in CELLS:
            runs += [
                (cell, unit_values(cell_path(cell, dataset, s, dry), dataset)) for s in fs.SEEDS
            ]
        if extra and dataset == "isic2018":
            short = {"unet": "M0", "decoder": "M2", "full": "M6"}
            runs += [(short[arm], extra[(arm, s)]) for arm in short for s in FRESH]
        x = np.array([(1, *CELLS[cell]) for cell, _ in runs], float)
        pinv = np.linalg.inv(x.T @ x) @ x.T
        df = len(runs) - x.shape[1]
        t = stats.t.ppf(0.975, df)
        for metric in METRICS:
            sign = -1.0 if metric in fs.DISTANCE else 1.0
            units = sorted(set.intersection(*[set(r[metric]) for _, r in runs]))
            y = sign * np.array([[r[metric][u] for u in units] for _, r in runs])
            run_means = y.mean(1)
            beta = pinv @ run_means
            resid = run_means - x @ beta
            cov = (resid @ resid / df) * np.linalg.inv(x.T @ x)
            per_unit = pinv @ y
            for i, name in enumerate(COMPONENTS, start=1):
                se = math.hypot(
                    math.sqrt(cov[i, i]), per_unit[i].std(ddof=1) / math.sqrt(len(units))
                )
                low, high = beta[i] - t * se, beta[i] + t * se
                rows.append(
                    dict(
                        dataset=dataset,
                        metric=metric,
                        component=name,
                        estimate=beta[i],
                        se=se,
                        low=low,
                        high=high,
                        classification=classify(low, high, SESOI[metric]),
                    )
                )
    pooled = []
    for metric in METRICS:
        for name in COMPONENTS:
            part = [r for r in rows if r["metric"] == metric and r["component"] == name]
            est = np.array([r["estimate"] for r in part])
            w = 1 / np.array([r["se"] ** 2 for r in part])
            mean, se = float((w * est).sum() / w.sum()), math.sqrt(1 / w.sum())
            q = float((w * (est - mean) ** 2).sum())
            low, high = mean - 1.96 * se, mean + 1.96 * se
            pooled.append(
                dict(
                    dataset=f"pooled (Q p={stats.chi2.sf(q, 2):.2f})",
                    metric=metric,
                    component=name,
                    estimate=mean,
                    se=se,
                    low=low,
                    high=high,
                    classification=classify(low, high, SESOI[metric]),
                )
            )
    return rows + pooled


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=Path, default=None)
    args = parser.parse_args()
    dry = args.dry_run is not None
    folder = args.dry_run if dry else ROOT / "revision/tables/families"
    if dry:
        a = experiment_a([1, 2, 3], [1, 2, 3])
        extra = None
    else:
        a = experiment_a(FRESH, ALL)
        extra = {(arm, s): unit_values(isic_path(arm, s), "isic2018") for arm in NAMES for s in ALL}
    write(folder / "expansion_contrasts.csv", a)
    write(folder / "expansion_interaction.csv", experiment_b(dry, extra))
    write(folder / "expansion_components.csv", components(dry, extra))
    print(f"wrote {folder}")


if __name__ == "__main__":
    main()
