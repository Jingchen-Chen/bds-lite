"""Post hoc balanced three-run KD display, explicitly authorized on 2026-11-09.

Recompute the existing Welch-run + test-unit estimator from saved class-level
outputs only. Keep the frozen 3-vs-10 analysis and the ACDC confirmation untouched.
Common seed numbers do not turn this into a paired test. Holm is applied to the
three Boundary F1 p-values within each displayed analysis, not across all past
analyses. These are exploratory, post hoc presentation results, never confirmation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.revision.expansion_statistics import cell_path, isic_path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/kd_posthoc_balanced"
RUNS = ROOT / "results/revision_runs"
DATASETS = ("isic2018", "acdc", "synapse")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def units(path: Path, dataset: str) -> pd.Series:
    frame = pd.read_csv(path)
    if frame.duplicated(["id", "label"]).any():
        raise ValueError(f"duplicate image/class: {path}")
    both = frame["empty_prediction"].astype(str).str.lower().eq("true") & frame[
        "empty_target"
    ].astype(str).str.lower().eq("true")
    frame.loc[both, "boundary_f1"] = np.nan
    sample = frame.groupby("id", sort=True)["boundary_f1"].mean()
    if dataset != "isic2018":
        clusters = sample.index.str.extract(r"^(patient\d+|case\d+)", expand=False)
        if clusters.isna().any():
            raise ValueError(f"unrecognized patient/case: {path}")
        sample = sample.groupby(clusters, sort=True).mean()
    return sample


def contrast(aa: list[pd.Series], bb: list[pd.Series]) -> dict:
    common = sorted(set.intersection(*(set(s.index[s.notna()]) for s in aa + bb)))
    a = np.array([s.loc[common].to_numpy() for s in aa])
    b = np.array([s.loc[common].to_numpy() for s in bb])
    ra, rb = a.mean(axis=1), b.mean(axis=1)
    na, nb = len(ra), len(rb)
    va, vb = ra.var(ddof=1) / na, rb.var(ddof=1) / nb
    df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    unit_var = np.var(a.mean(axis=0) - b.mean(axis=0), ddof=1) / len(common)
    se = math.sqrt(va + vb + unit_var)
    delta = ra.mean() - rb.mean()
    half = stats.t.ppf(0.975, df) * se
    return dict(
        delta=float(delta),
        low=float(delta - half),
        high=float(delta + half),
        p=float(2 * stats.t.sf(abs(delta / se), df)),
        se=se,
        df=float(df),
        n_a=na,
        n_b=nb,
        units=len(common),
        arm_a_mean=float(ra.mean()),
        arm_b_mean=float(rb.mean()),
    )


def holm(rows: list[dict]) -> None:
    running = 0.0
    for rank, row in enumerate(sorted(rows, key=lambda r: r["p"])):
        running = max(running, min(1.0, (len(rows) - rank) * row["p"]))
        row["p_holm_over_datasets"] = running


def main() -> None:
    frozen_file = ROOT / "results/kd_unified/contrasts.csv"
    frozen = list(csv.DictReader(frozen_file.open()))
    balanced, original, sources, errors = [], [], [], []
    for dataset in DATASETS:
        kd = [RUNS / f"{dataset}_kd_band_seed{s}/per_case_test.csv" for s in (1, 2, 3)]
        ref = [
            isic_path("decoder", s) if dataset == "isic2018" else cell_path("M2", dataset, s, False)
            for s in (1, 2, 3)
        ]
        aa, bb = [units(p, dataset) for p in kd], [units(p, dataset) for p in ref]
        row = dict(
            dataset=dataset,
            analysis="post hoc balanced presentation",
            reference_seeds="1-3",
            **contrast(aa, bb),
        )
        expected = next(
            r
            for r in frozen
            if r["dataset"] == dataset
            and r["comparison"] == "KD-band - decoder-only"
            and r["metric"] == "boundary_f1"
            and r["reference_seeds"] == "1-3"
        )
        errors.extend(abs(row[k] - float(expected[k])) for k in ("delta", "low", "high", "p"))
        balanced.append(row)
        for arm, paths in [("D+KD-band", kd), ("decoder-only", ref)]:
            for seed, path in zip((1, 2, 3), paths, strict=True):
                sources.append(
                    dict(
                        dataset=dataset,
                        arm=arm,
                        seed=seed,
                        path=path.relative_to(ROOT).as_posix(),
                        sha256=digest(path),
                    )
                )
        if dataset == "isic2018":
            ref10 = [isic_path("decoder", s) for s in range(1, 11)]
            orig = dict(
                dataset=dataset,
                analysis="original frozen reference (sensitivity in current presentation)",
                reference_seeds="1-10",
                **contrast(aa, [units(p, dataset) for p in ref10]),
            )
            for seed, path in zip(range(4, 11), ref10[3:], strict=True):
                sources.append(
                    dict(
                        dataset=dataset,
                        arm="decoder-only sensitivity",
                        seed=seed,
                        path=path.relative_to(ROOT).as_posix(),
                        sha256=digest(path),
                    )
                )
        else:
            orig = dict(
                row, analysis="original frozen reference (sensitivity in current presentation)"
            )
        original.append(orig)
    holm(balanced)
    holm(original)
    for row in original:
        expected = next(
            r
            for r in frozen
            if r["dataset"] == row["dataset"]
            and r["comparison"] == "KD-band - decoder-only"
            and r["metric"] == "boundary_f1"
            and r["primary"] == "True"
        )
        errors.extend(
            abs(row[k] - float(expected[k]))
            for k in ("delta", "low", "high", "p", "p_holm_over_datasets")
        )
    if max(errors) > 1e-10:
        raise ValueError(f"cannot reproduce original evidence: maximum error {max(errors)}")
    OUT.mkdir(exist_ok=True)
    with (OUT / "contrasts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(balanced[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(balanced + original)
    receipt = dict(
        status=(
            "post hoc main-display promotion and Holm family, author requested "
            "2026-11-09 after all results were known"
        ),
        preexisting_sensitivity=(
            "The frozen protocol already included seed-matched ISIC contrasts as "
            "descriptive sensitivities. Their promotion and new balanced-family "
            "Holm adjustment are post hoc."
        ),
        estimator=(
            "Unpaired Welch run variance plus test-unit SE; Welch df and t "
            "intervals; original estimator retained. Same seed numbers are not "
            "analyzed as paired observations."
        ),
        multiplicity=(
            "Separate three-dataset Holm families for Boundary F1. No new "
            "confirmatory decision; correction does not undo post hoc selection "
            "of this presentation."
        ),
        appropriateness=(
            "Balanced seed coverage is acceptable for exploratory presentation "
            "but is not required by Welch inference and discards seven available "
            "decoder runs. Three runs give unstable variance estimates and wide "
            "intervals. Retain the original frozen 3-vs-10 sensitivity rather "
            "than selecting the more favorable result."
        ),
        frozen_source_sha256=digest(frozen_file),
        max_reconciliation_error=max(errors),
        balanced=balanced,
        original_sensitivity=original,
        sources=sources,
        confirmation_hashes={
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                ROOT / "results/acdc_confirm/results/results.json",
                ROOT / "results/acdc_confirm/results/contrasts.csv",
            ]
        },
    )
    (OUT / "analysis.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for row in balanced:
        print(row)


if __name__ == "__main__":
    main()
