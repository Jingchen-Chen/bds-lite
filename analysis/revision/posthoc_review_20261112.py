# ruff: noqa: E501
"""POST HOC analyses added in the review round of 2026-11-12 (no training, no model execution).

Authorization and scope: evidence/EVIDENCE_ADDENDUM_20261112_review_round.md. Reads only saved
per-case evaluation outputs (hash-checked through analysis/followup/analyze.arm_runs for the
official ACDC arms) and results/tables/families/expansion_interaction.csv. No registered estimate,
decision or reading is changed; every output of this script is post hoc and exploratory.

1. Augmentation-by-distillation interaction on the official ACDC patients 101-150:
       I = (D+KD-band - D) - (D+KD-band_aug - D_aug).
   The four arms were scored on the same 50 patients, so the two simple contrasts are not
   independent: the test-unit term is computed from the per-patient double difference, not as the
   sum of the unit terms of the two contrasts. Three standard errors are set side by side:
   (a) the registered form extended to a four-arm linear contrast (between-run Welch terms of the
       four arms plus Var_u(dd_u)/U, Welch-Satterthwaite df from the between-run terms);
   (b) a crossed random-effects model by the method of moments, which removes the run-by-patient
       residual that (a) counts in both components (Supplementary S20);
   (c) a hierarchical bootstrap (patients resampled once and shared by the four arms, runs
       resampled within arm; 10,000 draws), expected to be conservative.
   Run counts: 17/17 without and 10/10 with augmentation (all runs), and a seed-matched
   sensitivity analysis on seeds 201-210 in all four arms.
2. Sensitivity of the pooled three-dataset summaries of study 1 (three runs per arm) to the pooling
   rule: registered inverse-variance fixed effect with a normal interval; the same with a
   t interval (Satterthwaite df of the weighted mean); equal weights (normal and t); DerSimonian-
   Laird random effects; and leave-one-dataset-out fixed-effect summaries.
3. Between-run SD of Boundary F1 of every arm with ten or more runs.
4. Missed structures and false alarms (per run, over slice-class or image-class pairs).

    PYTHONPATH=src:. .venv/bin/python analysis/revision/posthoc_review_20261112.py
Outputs: results/posthoc_review_20261112/{interaction,pooling_sensitivity,run_sd,
missed_false_alarm}.csv and summary.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT / "analysis/followup"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import expansion_statistics as ex  # noqa: E402
from analyze import arm_runs  # noqa: E402
from posthoc_reference_estimators import anova_arm  # noqa: E402

OUT = ROOT / "results/posthoc_review_20261112"
DRAWS = 10_000
SEED = 20261112
WEIGHTS = {"kd_band": 1.0, "decoder": -1.0, "kd_band_aug": -1.0, "decoder_aug": 1.0}
MATCHED = list(range(201, 211))
POOLED_EFFECTS = (
    "decoder without surface",
    "distillation+gate without surface",
    "distillation+gate with surface",
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- 1. interaction
def arm_matrices(values: dict[str, list[dict]], metric: str) -> dict[str, np.ndarray]:
    sign = -1.0 if metric in ("hd95", "assd") else 1.0
    units = sorted(set.intersection(*[set(r[metric]) for runs in values.values() for r in runs]))
    if len(units) != 50:
        raise ValueError(f"expected 50 patients defined in every run, got {len(units)}")
    return {
        arm: sign * np.array([[r[metric][u] for u in units] for r in runs])
        for arm, runs in values.items()
    }


def interaction_registered(y: dict[str, np.ndarray]) -> dict:
    u = next(iter(y.values())).shape[1]
    estimate = sum(w * y[a].mean() for a, w in WEIGHTS.items())
    run_terms = {a: y[a].mean(1).var(ddof=1) / y[a].shape[0] for a in WEIGHTS}
    dd = sum(w * y[a].mean(0) for a, w in WEIGHTS.items())
    unit_term = float(dd.var(ddof=1)) / u
    var_run = sum(run_terms.values())
    df = var_run**2 / sum(v**2 / (y[a].shape[0] - 1) for a, v in run_terms.items())
    se = math.sqrt(var_run + unit_term)
    t = stats.t.ppf(0.975, df)
    return dict(
        estimate=float(estimate),
        se=se,
        se_run=math.sqrt(var_run),
        se_unit=math.sqrt(unit_term),
        df=float(df),
        low=float(estimate - t * se),
        high=float(estimate + t * se),
        p=float(2 * stats.t.sf(abs(estimate / se), df)),
    )


def interaction_mixed(y: dict[str, np.ndarray]) -> dict:
    u = next(iter(y.values())).shape[1]
    estimate = sum(w * y[a].mean() for a, w in WEIGHTS.items())
    parts = {a: anova_arm(y[a]) for a in WEIGHTS}
    s_e2 = sum(p[1] * p[3] for p in parts.values()) / sum(p[3] for p in parts.values())
    dd = sum(w * y[a].mean(0) for a, w in WEIGHTS.items())
    arm_patient = float(dd.var(ddof=1)) - s_e2 * sum(1 / y[a].shape[0] for a in WEIGHTS)
    terms = [parts[a][0] / (y[a].shape[0] * u) for a in WEIGHTS] + [max(0.0, arm_patient) / u]
    dfs = [parts[a][2] for a in WEIGHTS] + [u - 1]
    var = sum(terms)
    df = var**2 / sum(t**2 / f for t, f in zip(terms, dfs, strict=True) if t > 0)
    se = math.sqrt(var)
    t = stats.t.ppf(0.975, df)
    return dict(
        estimate=float(estimate),
        se=se,
        df=float(df),
        low=float(estimate - t * se),
        high=float(estimate + t * se),
        p=float(2 * stats.t.sf(abs(estimate / se), df)),
        arm_patient_truncated=arm_patient < 0,
    )


def interaction_bootstrap(y: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    u = next(iter(y.values())).shape[1]
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        cols = rng.integers(0, u, u)
        total = 0.0
        for arm, w in WEIGHTS.items():
            n = y[arm].shape[0]
            total += w * y[arm][np.ix_(rng.integers(0, n, n), cols)].mean()
        out[i] = total
    low, high = np.percentile(out, [2.5, 97.5])
    return dict(
        low=float(low),
        high=float(high),
        se=float(out.std(ddof=1)),
        share_le_zero=float((out <= 0).mean()),
    )


def interaction(paths: dict[str, dict[int, Path]]) -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for label, seeds in (("all runs (17/17/10/10)", None), ("seeds 201-210 in all arms", MATCHED)):
        values = {
            arm: [
                ex.unit_values(p, "acdc")
                for s, p in paths[arm].items()
                if seeds is None or s in seeds
            ]
            for arm in WEIGHTS
        }
        for metric in ex.METRICS:
            y = arm_matrices(values, metric)
            reg, mix, boot = (
                interaction_registered(y),
                interaction_mixed(y),
                interaction_bootstrap(y, rng),
            )
            simple = ex.contrast(values["kd_band"], values["decoder"], metric)
            simple_aug = ex.contrast(values["kd_band_aug"], values["decoder_aug"], metric)
            if abs(reg["estimate"] - (simple["delta"] - simple_aug["delta"])) > 1e-12:
                raise ValueError("interaction differs from the difference of simple contrasts")
            naive = math.sqrt(
                simple["se_run"] ** 2
                + simple["se_unit"] ** 2
                + simple_aug["se_run"] ** 2
                + simple_aug["se_unit"] ** 2
            )
            rows.append(
                dict(
                    runs=label,
                    metric=metric,
                    n_kd_band=y["kd_band"].shape[0],
                    n_decoder=y["decoder"].shape[0],
                    n_kd_band_aug=y["kd_band_aug"].shape[0],
                    n_decoder_aug=y["decoder_aug"].shape[0],
                    patients=50,
                    simple_noaug=simple["delta"],
                    simple_aug=simple_aug["delta"],
                    interaction=reg["estimate"],
                    registered_form_se=reg["se"],
                    registered_form_se_run=reg["se_run"],
                    registered_form_se_unit=reg["se_unit"],
                    registered_form_df=reg["df"],
                    registered_form_low=reg["low"],
                    registered_form_high=reg["high"],
                    registered_form_p=reg["p"],
                    independent_contrasts_se=naive,
                    mixed_se=mix["se"],
                    mixed_df=mix["df"],
                    mixed_low=mix["low"],
                    mixed_high=mix["high"],
                    mixed_p=mix["p"],
                    mixed_arm_patient_truncated=mix["arm_patient_truncated"],
                    bootstrap_se=boot["se"],
                    bootstrap_low=boot["low"],
                    bootstrap_high=boot["high"],
                    bootstrap_share_le_zero=boot["share_le_zero"],
                )
            )
    return rows


# ---------------------------------------------------------------- 2. pooling
def summary_row(effect: str, metric: str, rule: str, est: float, se: float, df: float) -> dict:
    crit = 1.959963984540054 if math.isinf(df) else float(stats.t.ppf(0.975, df))
    low, high = est - crit * se, est + crit * se
    return dict(
        effect=effect,
        metric=metric,
        rule=rule,
        estimate=est,
        se=se,
        df=df,
        low=low,
        high=high,
        classification=ex.classify(low, high, ex.SESOI[metric]),
    )


def pooling() -> list[dict]:
    with (ROOT / "results/tables/families/expansion_interaction.csv").open() as stream:
        source = [r for r in csv.DictReader(stream) if r["seeds"] == "1-3"]
    rows = []
    for effect in POOLED_EFFECTS:
        for metric in ("boundary_f1", "dsc"):
            part = {
                r["dataset"]: r
                for r in source
                if r["effect"] == effect and r["metric"] == metric and r["dataset"] in ex.DATASETS
            }
            est = np.array([float(part[d]["estimate"]) for d in ex.DATASETS])
            se = np.array([float(part[d]["se"]) for d in ex.DATASETS])
            dfs = np.array([float(part[d]["df"]) for d in ex.DATASETS])
            w = 1 / se**2
            fixed, fixed_se = float((w * est).sum() / w.sum()), math.sqrt(1 / w.sum())
            share = w / w.sum()
            # Satterthwaite df of a weighted mean with fixed weights c_i: Var = sum c_i^2 se_i^2.
            fixed_df = float((share**2 * se**2).sum() ** 2 / ((share**2 * se**2) ** 2 / dfs).sum())
            equal, equal_se = float(est.mean()), math.sqrt(float((se**2).sum())) / 3
            equal_df = float((se**2).sum() ** 2 / ((se**2) ** 2 / dfs).sum())
            q = float((w * (est - fixed) ** 2).sum())
            tau2 = max(0.0, (q - 2) / (w.sum() - (w**2).sum() / w.sum()))
            wr = 1 / (se**2 + tau2)
            rows += [
                summary_row(
                    effect,
                    metric,
                    "inverse variance, normal (registered)",
                    fixed,
                    fixed_se,
                    math.inf,
                ),
                summary_row(
                    effect, metric, "inverse variance, t (Satterthwaite)", fixed, fixed_se, fixed_df
                ),
                summary_row(effect, metric, "equal weights, normal", equal, equal_se, math.inf),
                summary_row(
                    effect, metric, "equal weights, t (Satterthwaite)", equal, equal_se, equal_df
                ),
                summary_row(
                    effect,
                    metric,
                    f"DerSimonian-Laird random effects (tau2={tau2:.3g}), normal",
                    float((wr * est).sum() / wr.sum()),
                    math.sqrt(1 / wr.sum()),
                    math.inf,
                ),
            ]
            for i, dataset in enumerate(ex.DATASETS):
                keep = [j for j in range(3) if j != i]
                wk = w[keep]
                rows.append(
                    summary_row(
                        effect,
                        metric,
                        f"inverse variance without {dataset}, normal",
                        float((wk * est[keep]).sum() / wk.sum()),
                        math.sqrt(1 / wk.sum()),
                        math.inf,
                    )
                )
                rows[-1]["weight_share"] = float(share[i])
            for row in rows[-8:]:
                row.setdefault("weight_share", "")
    return rows


# ---------------------------------------------------------------- 3. run SD
def run_sd(paths: dict[str, dict[int, Path]]) -> list[dict]:
    rows = []
    for arm in ex.NAMES:
        runs = [ex.unit_values(ex.isic_path(arm, s), "isic2018") for s in ex.ALL]
        means = np.array([np.mean(list(r["boundary_f1"].values())) for r in runs])
        rows.append(
            dict(
                dataset="isic2018",
                test_set="official test images",
                arm=arm,
                runs=means.size,
                mean=float(means.mean()),
                sd=float(means.std(ddof=1)),
            )
        )
    for arm, by_seed in paths.items():
        runs = [ex.unit_values(p, "acdc") for p in by_seed.values()]
        common = sorted(set.intersection(*[set(r["boundary_f1"]) for r in runs]))
        means = np.array([np.mean([r["boundary_f1"][u] for u in common]) for r in runs])
        rows.append(
            dict(
                dataset="acdc",
                test_set="official patients 101-150",
                arm=arm,
                runs=means.size,
                mean=float(means.mean()),
                sd=float(means.std(ddof=1)),
            )
        )
    return rows


# ---------------------------------------------------------------- 4. missed / false alarms
def pair_counts(path: Path) -> dict[str, float]:
    present = missed = absent = alarm = 0
    with path.open() as stream:
        for row in csv.DictReader(stream):
            target_empty = row["empty_target"] == "True"
            prediction_empty = row["empty_prediction"] == "True"
            if target_empty:
                absent += 1
                alarm += not prediction_empty
            else:
                present += 1
                missed += prediction_empty
    return dict(
        present=present,
        missed=missed,
        absent=absent,
        false_alarm=alarm,
        missed_rate=missed / present,
        false_alarm_rate=alarm / absent if absent else float("nan"),
    )


def missed(paths: dict[str, dict[int, Path]]) -> list[dict]:
    groups: list[tuple[str, str, str, list[Path]]] = []
    for dataset in ex.DATASETS:
        for cell, name in (("M0", "U"), ("M2", "D"), ("M3", "D+F+G"), ("M6", "BDS-Lite")):
            groups.append(
                (
                    dataset,
                    "matched protocol, seeds 1-3",
                    name,
                    [ex.cell_path(cell, dataset, s, False) for s in (1, 2, 3)],
                )
            )
    for arm, by_seed in paths.items():
        groups.append(("acdc", "official patients 101-150", arm, list(by_seed.values())))
    rows = []
    for dataset, test_set, arm, files in groups:
        counts = [pair_counts(p) for p in files]
        rows.append(
            dict(
                dataset=dataset,
                test_set=test_set,
                arm=arm,
                runs=len(counts),
                present_pairs=counts[0]["present"],
                absent_pairs=counts[0]["absent"],
                missed_rate_mean=float(np.mean([c["missed_rate"] for c in counts])),
                missed_rate_sd=float(np.std([c["missed_rate"] for c in counts], ddof=1)),
                false_alarm_rate_mean=float(np.mean([c["false_alarm_rate"] for c in counts])),
                false_alarm_rate_sd=float(np.std([c["false_alarm_rate"] for c in counts], ddof=1)),
            )
        )
    return rows


def main() -> None:
    paths = arm_runs()
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "interaction": interaction(paths),
        "pooling_sensitivity": pooling(),
        "run_sd": run_sd(paths),
        "missed_false_alarm": missed(paths),
    }
    for name, rows in tables.items():
        write(OUT / f"{name}.csv", rows)
    inputs = {
        str(p.relative_to(ROOT)): sha(p) for by_seed in paths.values() for p in by_seed.values()
    }
    source = ROOT / "results/tables/families/expansion_interaction.csv"
    inputs[str(source.relative_to(ROOT))] = sha(source)
    (OUT / "summary.json").write_text(
        json.dumps(
            dict(
                label="post hoc, exploratory; see EVIDENCE_ADDENDUM_20261112_review_round.md",
                draws=DRAWS,
                seed=SEED,
                inputs=inputs,
                outputs={f"{n}.csv": sha(OUT / f"{n}.csv") for n in tables},
            ),
            indent=2,
        )
        + "\n"
    )
    for r in tables["interaction"]:
        print(
            f"{r['runs']:28s} {r['metric']:12s} I={r['interaction']:+.4f} "
            f"reg [{r['registered_form_low']:+.4f}, {r['registered_form_high']:+.4f}] "
            f"p={r['registered_form_p']:.2g} mixed [{r['mixed_low']:+.4f}, {r['mixed_high']:+.4f}] "
            f"boot [{r['bootstrap_low']:+.4f}, {r['bootstrap_high']:+.4f}]"
        )
    for r in tables["pooling_sensitivity"]:
        if r["metric"] == "boundary_f1":
            print(
                f"{r['effect'][:34]:34s} {r['rule'][:52]:52s} {r['estimate']:+.4f} "
                f"[{r['low']:+.4f}, {r['high']:+.4f}] {r['weight_share']}"
            )
    for r in tables["run_sd"]:
        print(f"{r['dataset']:9s} {r['arm']:14s} n={r['runs']:2d} sd={r['sd']:.4f}")
    for r in tables["missed_false_alarm"]:
        print(
            f"{r['dataset']:9s} {r['test_set'][:16]:16s} {r['arm']:13s} n={r['runs']:2d} "
            f"missed={r['missed_rate_mean']:.4f} false alarm={r['false_alarm_rate_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
