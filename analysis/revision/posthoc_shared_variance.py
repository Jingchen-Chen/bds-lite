# ruff: noqa: E501
"""Post hoc sensitivity analysis: the run-by-unit residual term shared by both SE components (Supplementary S20).

Authorized by the author on 2026-11-09 (restructure revision, decision 8 and Option B). This is
a disclosure-only sensitivity analysis. Every reading, label and stage-3 decision stays with the
pre-specified estimator. The alternative estimator below was chosen post hoc, after all results
were known.

**Recipe-level SE (expansion_statistics.contrast / linear_contrast).** The SE combines two
components:
- SE_run^2 = s_A^2/n_A + s_B^2/n_B (or the pooled within-cell form);
- SE_unit^2 = Var_u(d_u)/U.

**The shared term.** Assume y[a,r,u] = mu_a + rho_ar + upsilon_u + gamma_au + eps_aru, with the
test units common to both arms. Then the term
    SHARED = sum_a sigma_eps,a^2 / (n_a U)
enters both SE_run^2 and SE_unit^2, so SE^2 overestimates Var(Delta) by SHARED.

**Estimator.** sigma_eps,a^2 is the residual mean square of the additive run + unit fit within
arm a. For linear contrasts it is pooled over cells, and SHARED = sigma_eps^2 * sum w_c^2/n_c / U.

**Adjusted variance (k = 0.5, 1, 1.5 times SHARED).** Each method-of-moments component is
truncated at zero:
    V_k = max(SE_run^2 - k SHARED, 0) + max(SE_unit^2 - k SHARED, 0) + k SHARED.
The untruncated SE^2 - SHARED is also reported. Intervals keep the pre-specified reference
distribution:
- t with the pre-specified df (Welch from the run terms, or pooled residual df);
- normal for inverse-variance pooling, recomputed with the adjusted per-dataset variances.

**Why the 0.5x / 1.5x range.** Under the model, the residual mean square has (n-1)(U-1) df per
arm (at least 22 here). Its relative sampling SD is therefore at most sqrt(2/22), about 0.30. The
0.5x-1.5x range covers more than +-1.6 such SDs. It also gives some room for model
misspecification (heteroscedastic or non-additive residuals), which the sampling SD does not
cover.

**Coverage.** A small plug-in simulation estimates the coverage of the pre-specified and
adjusted (k = 1) 95% intervals. It uses the variance components estimated for selected
comparisons, under normal, homoscedastic errors. It is not a validation of coverage beyond that
model.

**Reading rule.** The rule was fixed before the first run (commit bbf2d9e): "negligible" only if
removing SHARED lowers the SE by less than 5% in every Boundary F1 contrast.

Each recomputed Delta, SE and pre-specified interval must reproduce the recorded value
(tolerance 1e-9), or the script fails.

Outputs in results/posthoc_restructure/:
- shared_component.csv (every comparison, five metrics);
- shared_component.json (summary);
- coverage.csv.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT / "src"))
import expansion_statistics as ex  # noqa: E402
import family_statistics as fs  # noqa: E402

from scripts.acdc_confirm.analyze import ARMS, new_path  # noqa: E402
from scripts.acdc_confirm.common import SEEDS  # noqa: E402

OUT = ROOT / "results/posthoc_restructure"
RUNS = ROOT / "results/revision_runs"
THRESHOLD = 0.05
FACTORS = (0.5, 1.0, 1.5)
COVERAGE_REPS, COVERAGE_SEED = 4000, 20261109


def matrix(runs: list[dict], metric: str, units: list[str]) -> np.ndarray:
    sign = -1.0 if metric in fs.DISTANCE else 1.0
    return sign * np.array([[r[metric][u] for u in units] for r in runs])


def residual_ms(y: np.ndarray) -> tuple[float, int]:
    resid = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
    df = (y.shape[0] - 1) * (y.shape[1] - 1)
    return float((resid**2).sum() / df), df


def adjusted(se_run2: float, se_unit2: float, shared: float, k: float) -> tuple[float, bool]:
    a, b = se_run2 - k * shared, se_unit2 - k * shared
    return max(a, 0.0) + max(b, 0.0) + k * shared, (a < 0 or b < 0)


def describe(label, metric, delta, se_run2, se_unit2, shared, df, n_a, n_b, units, extra=None):
    se2 = se_run2 + se_unit2
    q = stats.t.ppf(0.975, df)
    row = dict(
        **label,
        metric=metric,
        n_a=n_a,
        n_b=n_b,
        units=units,
        df=df,
        delta=delta,
        se_run=math.sqrt(se_run2),
        se_unit=math.sqrt(se_unit2),
        se=math.sqrt(se2),
        low=delta - q * math.sqrt(se2),
        high=delta + q * math.sqrt(se2),
        shared_variance=shared,
        shared_fraction_of_se2=shared / se2,
        untruncated_se_without_shared=math.sqrt(se2 - shared) if se2 > shared else float("nan"),
    )
    for k in FACTORS:
        v, truncated = adjusted(se_run2, se_unit2, shared, k)
        tag = f"k{k:g}".replace(".", "")
        row[f"se_{tag}"] = math.sqrt(v)
        row[f"low_{tag}"] = delta - q * math.sqrt(v)
        row[f"high_{tag}"] = delta + q * math.sqrt(v)
        row[f"truncated_{tag}"] = truncated
    row["se_reduction"] = 1 - row["se_k1"] / row["se"]
    row["sesoi"] = ex.SESOI[metric]
    row["pre_excludes_zero"] = row["low"] > 0 or row["high"] < 0
    for k in FACTORS:
        tag = f"k{k:g}".replace(".", "")
        row[f"excludes_zero_{tag}"] = row[f"low_{tag}"] > 0 or row[f"high_{tag}"] < 0
    if extra:
        row |= extra
    return row


def check(value: float, expected: float, what: str) -> None:
    if abs(value - expected) > 1e-9:
        raise ValueError(f"does not reproduce {what}: {value} vs {expected}")


def two_arm(a: list[dict], b: list[dict], metric: str, **label) -> dict:
    c = ex.contrast(a, b, metric)
    units = sorted(set.intersection(*[set(r[metric]) for r in a + b]))
    ya, yb = matrix(a, metric, units), matrix(b, metric, units)
    ms_a, _ = residual_ms(ya)
    ms_b, _ = residual_ms(yb)
    shared = ms_a / (len(a) * len(units)) + ms_b / (len(b) * len(units))
    row = describe(
        label,
        metric,
        c["delta"],
        c["se_run"] ** 2,
        c["se_unit"] ** 2,
        shared,
        c["df"],
        len(a),
        len(b),
        len(units),
        extra=dict(
            ms_eps_a=ms_a,
            ms_eps_b=ms_b,
            s2_a=float(ya.mean(1).var(ddof=1)),
            s2_b=float(yb.mean(1).var(ddof=1)),
        ),
    )
    check(row["low"], c["low"], f"low {label} {metric}")
    check(row["high"], c["high"], f"high {label} {metric}")
    return row


def stage3() -> list[dict]:
    values = {
        arm: [ex.unit_values(new_path(arm, s, "official_test"), "acdc") for s in SEEDS]
        for arm in ARMS
    }
    recorded = json.loads((ROOT / "results/acdc_confirm/results/results.json").read_text())
    rows = []
    for metric in ex.METRICS:
        r = two_arm(
            values["kd_band"],
            values["decoder"],
            metric,
            family="stage 3",
            dataset="acdc",
            contrast="D+KD-band - D",
        )
        if metric == "boundary_f1":
            for key in ("delta", "low", "high"):
                check(r[key], recorded["primary"]["contrast"][key], f"stage-3 {key}")
        rows.append(r)
    return rows


def stage2() -> list[dict]:
    frozen = list(csv.DictReader((ROOT / "results/kd_posthoc_balanced/contrasts.csv").open()))
    rows = []
    for ds in ex.DATASETS:
        kd = [
            ex.unit_values(RUNS / f"{ds}_kd_band_seed{s}/per_case_test.csv", ds) for s in (1, 2, 3)
        ]
        refs = {"1-3": (1, 2, 3)} | ({"1-10": tuple(range(1, 11))} if ds == "isic2018" else {})
        for seeds_label, seeds in refs.items():
            ref = [
                ex.unit_values(
                    ex.isic_path("decoder", s)
                    if ds == "isic2018"
                    else ex.cell_path("M2", ds, s, False),
                    ds,
                )
                for s in seeds
            ]
            family = (
                "stage 2 balanced display" if seeds_label == "1-3" else "stage 2 original frozen"
            )
            for metric in ex.METRICS:
                r = two_arm(kd, ref, metric, family=family, dataset=ds, contrast="D+KD-band - D")
                if metric == "boundary_f1":
                    match = [
                        f
                        for f in frozen
                        if f["dataset"] == ds and f["reference_seeds"] == seeds_label
                    ]
                    for key in ("delta", "low", "high"):
                        check(r[key], float(match[0][key]), f"stage-2 {key} {ds} {seeds_label}")
                rows.append(r)
    return rows


def extension() -> list[dict]:
    values = {
        (arm, s): ex.unit_values(ex.isic_path(arm, s), "isic2018")
        for arm in ex.NAMES
        for s in ex.ALL
    }
    recorded = list(
        csv.DictReader((ROOT / "results/tables/families/expansion_contrasts.csv").open())
    )
    rows = []
    for metric in ex.METRICS:
        for first, second in ex.PAIRS:
            r = two_arm(
                [values[(first, s)] for s in ex.ALL],
                [values[(second, s)] for s in ex.ALL],
                metric,
                family="ISIC2018 ten-run extension (all seeds)",
                dataset="isic2018",
                contrast=f"{first} - {second}",
            )
            rec = next(
                x
                for x in recorded
                if x["metric"] == metric and x["first"] == first and x["second"] == second
            )
            for key in ("delta", "low", "high"):
                check(
                    r[key], float(rec[f"all_{key}"]), f"extension {key} {first}-{second} {metric}"
                )
            rows.append(r)
    return rows


STAGE1 = {
    "decoder without surface": ex.EFFECTS["decoder without surface"],
    "distillation+gate without surface": ex.TRANSFER["distillation+gate without surface"],
    "distillation+gate with surface": ex.TRANSFER["distillation+gate with surface"],
}


def stage1() -> tuple[list[dict], dict]:
    recorded = list(
        csv.DictReader((ROOT / "results/tables/families/expansion_interaction.csv").open())
    )
    cache = {
        (ds, c): [ex.unit_values(ex.cell_path(c, ds, s, False), ds) for s in fs.SEEDS]
        for ds in ex.DATASETS
        for c in ("M0", "M2", "M3", "M6", "M7")
    }
    rows, components = [], {}
    for metric in ex.METRICS:
        for name, weights in STAGE1.items():
            part = []
            for ds in ex.DATASETS:
                cells = {c: cache[(ds, c)] for c in weights}
                estimate, se_run, se_unit, df = ex.linear_contrast(cells, weights, metric)
                units = sorted(set.intersection(*[set(r[metric]) for c in cells for r in cells[c]]))
                ss = dfs = 0.0
                for c in cells:
                    ms, d = residual_ms(matrix(cells[c], metric, units))
                    ss += ms * d
                    dfs += d
                n = len(next(iter(cells.values())))
                shared = ss / dfs * sum(w**2 / n for w in weights.values()) / len(units)
                rec = next(
                    x
                    for x in recorded
                    if x["effect"] == name
                    and x["metric"] == metric
                    and x["dataset"] == ds
                    and x["seeds"] == "1-3"
                )
                r = describe(
                    dict(family="stage 1", dataset=ds, contrast=name),
                    metric,
                    estimate,
                    se_run**2,
                    se_unit**2,
                    shared,
                    df,
                    n,
                    n,
                    len(units),
                    extra=dict(ms_eps_pooled=ss / dfs),
                )
                for key, rk in (("delta", "estimate"), ("low", "low"), ("high", "high")):
                    check(r[key], float(rec[rk]), f"stage-1 {key} {name} {ds} {metric}")
                components[(name, metric, ds)] = (se_run**2, se_unit**2, shared, estimate)
                part.append(r)
            rows += part
            rows.append(pooled_row(name, metric, part, recorded))
    return rows, components


def pooled_row(name, metric, part, recorded) -> dict:
    est = np.array([r["delta"] for r in part])
    row = dict(family="stage 1", dataset="pooled", contrast=name, metric=metric, df=float("inf"))
    for tag, var in [("", [r["se"] ** 2 for r in part])] + [
        (f"_k{k:g}".replace(".", ""), [r[f"se_k{k:g}".replace(".", "")] ** 2 for r in part])
        for k in FACTORS
    ]:
        w = 1 / np.array(var)
        mean, se = float((w * est).sum() / w.sum()), math.sqrt(1 / w.sum())
        if tag == "":
            row |= dict(delta=mean, se=se)
        row[f"delta{tag}"] = mean
        row[f"se{tag}"] = se
        row[f"low{tag}"] = mean - 1.96 * se
        row[f"high{tag}"] = mean + 1.96 * se
        row[f"excludes_zero{tag}" if tag else "pre_excludes_zero"] = (
            row[f"low{tag}"] > 0 or row[f"high{tag}"] < 0
        )
    row["truncated_k1"] = any(r["truncated_k1"] for r in part)
    row["truncated_k05"] = any(r["truncated_k05"] for r in part)
    row["truncated_k15"] = any(r["truncated_k15"] for r in part)
    row["se_reduction"] = 1 - row["se_k1"] / row["se"]
    row["sesoi"] = ex.SESOI[metric]
    rec = next(
        x
        for x in recorded
        if x["effect"] == name and x["metric"] == metric and x["dataset"].startswith("pooled")
    )
    for key, rk in (("delta", "estimate"), ("low", "low"), ("high", "high")):
        check(row[key], float(rec[rk]), f"stage-1 pooled {key} {name} {metric}")
    return row


# ------------------------------------------------------------------ coverage simulation
def simulate_two_arm(rng, n_a, n_b, units, s2rho, ms, vgamma, pooled_df):
    """Return coverage (pre-specified, adjusted k=1) of 95% intervals for Delta = 0."""
    hits = np.zeros(2)
    for _ in range(COVERAGE_REPS):
        g = rng.normal(0, math.sqrt(vgamma), units)
        ya = (
            rng.normal(0, math.sqrt(s2rho[0]), (n_a, 1))
            + g
            + rng.normal(0, math.sqrt(ms[0]), (n_a, units))
        )
        yb = rng.normal(0, math.sqrt(s2rho[1]), (n_b, 1)) + rng.normal(
            0, math.sqrt(ms[1]), (n_b, units)
        )
        ra, rb = ya.mean(1), yb.mean(1)
        delta = ra.mean() - rb.mean()
        if pooled_df:
            pv = (((ra - ra.mean()) ** 2).sum() + ((rb - rb.mean()) ** 2).sum()) / (n_a + n_b - 2)
            se_run2, df = pv * (1 / n_a + 1 / n_b), n_a + n_b - 2
            pooled_ms = (residual_ms(ya)[0] + residual_ms(yb)[0]) / 2
            shared = pooled_ms * (1 / n_a + 1 / n_b) / units
        else:
            va, vb = ra.var(ddof=1) / n_a, rb.var(ddof=1) / n_b
            se_run2 = va + vb
            df = (va + vb) ** 2 / (va**2 / (n_a - 1) + vb**2 / (n_b - 1))
            shared = residual_ms(ya)[0] / (n_a * units) + residual_ms(yb)[0] / (n_b * units)
        se_unit2 = np.var(ya.mean(0) - yb.mean(0), ddof=1) / units
        q = stats.t.ppf(0.975, df)
        v1, _ = adjusted(se_run2, se_unit2, shared, 1.0)
        hits += [abs(delta) <= q * math.sqrt(se_run2 + se_unit2), abs(delta) <= q * math.sqrt(v1)]
    return hits / COVERAGE_REPS


def plug_in(row) -> tuple[list[float], list[float], float]:
    if "ms_eps_a" in row:
        ms = [row["ms_eps_a"], row["ms_eps_b"]]
        s2rho = [
            max(row["s2_a"] - ms[0] / row["units"], 0.0),
            max(row["s2_b"] - ms[1] / row["units"], 0.0),
        ]
    else:  # pooled two-cell contrast: same components in both cells
        ms = [row["ms_eps_pooled"]] * 2
        run_var = row["se_run"] ** 2 / (1 / row["n_a"] + 1 / row["n_b"])
        s2rho = [max(run_var - ms[0] / row["units"], 0.0)] * 2
    vgamma = max(row["se_unit"] ** 2 * row["units"] - ms[0] / row["n_a"] - ms[1] / row["n_b"], 0.0)
    return s2rho, ms, vgamma


def coverage(rows: list[dict]) -> list[dict]:
    rng = np.random.default_rng(COVERAGE_SEED)
    picks = [
        ("stage 3", "acdc", "D+KD-band - D", False),
        ("stage 2 balanced display", "acdc", "D+KD-band - D", False),
        ("stage 2 balanced display", "synapse", "D+KD-band - D", False),
        ("stage 1", "acdc", "distillation+gate without surface", True),
        ("stage 1", "synapse", "distillation+gate without surface", True),
        ("stage 1", "acdc", "distillation+gate with surface", True),
        ("stage 1", "synapse", "distillation+gate with surface", True),
    ]
    out = []
    for family, ds, contrast, pooled in picks:
        row = next(
            r
            for r in rows
            if r["family"] == family
            and r["dataset"] == ds
            and r["contrast"] == contrast
            and r["metric"] == "boundary_f1"
        )
        s2rho, ms, vgamma = plug_in(row)
        pre, adj = simulate_two_arm(
            rng, row["n_a"], row["n_b"], row["units"], s2rho, ms, vgamma, pooled
        )
        out.append(
            dict(
                family=family,
                dataset=ds,
                contrast=contrast,
                metric="boundary_f1",
                n_a=row["n_a"],
                n_b=row["n_b"],
                units=row["units"],
                reps=COVERAGE_REPS,
                sigma2_run_a=s2rho[0],
                sigma2_run_b=s2rho[1],
                sigma2_eps_a=ms[0],
                sigma2_eps_b=ms[1],
                var_gamma_diff=vgamma,
                coverage_prespecified=pre,
                coverage_adjusted_k1=adj,
                mc_se=math.sqrt(0.95 * 0.05 / COVERAGE_REPS),
            )
        )
        print(out[-1], flush=True)
    return out


def write(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    s1, _ = stage1()
    rows = stage3() + stage2() + extension() + s1
    OUT.mkdir(parents=True, exist_ok=True)
    write("shared_component.csv", rows)
    cov = coverage(rows)
    write("coverage.csv", cov)
    comparisons = [r for r in rows if r["dataset"] != "pooled"]
    bf1 = [r for r in comparisons if r["metric"] == "boundary_f1"]
    worst = max(bf1, key=lambda r: r["se_reduction"])
    flips = [
        dict(family=r["family"], dataset=r["dataset"], contrast=r["contrast"], metric=r["metric"])
        for r in rows
        if r["pre_excludes_zero"] != r["excludes_zero_k1"]
    ]
    summary = dict(
        label="post hoc sensitivity analysis (author authorization 2026-11-09, Option B); readings unchanged",
        rule=f"negligible if every Boundary F1 SE reduction < {THRESHOLD:.0%} (fixed before the first run, commit bbf2d9e)",
        comparisons=len(comparisons),
        pooled_rows=len(rows) - len(comparisons),
        boundary_f1_comparisons=len(bf1),
        truncated_any_component={
            f"k{k:g}": sum(r[f"truncated_k{k:g}".replace(".", "")] for r in comparisons)
            for k in FACTORS
        },
        truncated_any_component_boundary_f1={
            f"k{k:g}": sum(r[f"truncated_k{k:g}".replace(".", "")] for r in bf1) for k in FACTORS
        },
        untruncated_negative=sum(
            not math.isfinite(r["untruncated_se_without_shared"]) for r in comparisons
        ),
        max_se_reduction_boundary_f1=worst["se_reduction"],
        max_se_reduction_at=dict(
            family=worst["family"], dataset=worst["dataset"], contrast=worst["contrast"]
        ),
        reading="negligible" if worst["se_reduction"] < THRESHOLD else "not negligible",
        zero_exclusion_changes_k1=flips,
        zero_exclusion_by_factor={
            f"k{k:g}": sum(r[f"excludes_zero_k{k:g}".replace(".", "")] for r in rows)
            for k in FACTORS
        }
        | dict(prespecified=sum(r["pre_excludes_zero"] for r in rows)),
    )
    (OUT / "shared_component.json").write_text(
        json.dumps(summary, indent=1, default=lambda o: o.item()) + "\n"
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
