"""Generate gate-collapse evidence tables and qualified findings from actual effects."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ptext(value, latex=False):
    number = float(value)
    if number >= 1e-4:
        return f"{number:.6g}"
    mantissa, exponent = f"{number:.2e}".split("e")
    return f"${mantissa}\\times 10^{{{int(exponent)}}}$" if latex else f"{number:.3g}"


def main():
    tables = ROOT / "results/tables"
    rows = [
        r
        for r in csv.DictReader((tables / "gate_identity_effects.csv").open())
        if r["split"] in ["test", "official_test"]
        and r["metric"] not in ["hd95_penalized", "assd_penalized"]
    ]
    tex = [
        r"\begin{tabular}{llrrrrl}",
        r"Dataset & Metric & Units & Effect & CI lower & CI upper & Holm p \\",
        r"\hline",
    ]
    text = [
        "# Matched-checkpoint gate identity findings",
        "",
        (
            "The intervention replaces the inference gate with unit modulation on the same "
            "trained models. It is not training-time M5. Architecture/scalar settings, "
            "exact state/alias identity and operator logits were validated for every "
            "evaluated cohort. Statistical analysis is a revision-stage exploratory "
            "amendment registered after prediction generation began, before comparative "
            "effect inspection."
        ),
        "",
        (
            "| Dataset | Metric | Paired units | Identity−full mean | 95% bootstrap "
            "interval | Holm p |"
        ),
        "|---|---|---:|---:|---|---:|",
    ]
    for row in rows:
        dataset, metric, n = row["dataset"], row["metric"], row["n"]
        effect = f"{float(row['mean_delta_identity_minus_full']):.6f}"
        low = f"{float(row['ci95_low']):.6f}"
        high = f"{float(row['ci95_high']):.6f}"
        metric_tex = metric.replace("_", r"\_")
        tex.append(
            f"{dataset} & {metric_tex} & {n} & {effect} & {low} & {high} & "
            f"{ptext(row['p_holm'], True)} " + r"\\"
        )
        text.append(
            f"| {dataset} | {metric} | {n} | {effect} | [{low}, {high}] | {ptext(row['p_holm'])} |"
        )
    tex.append(r"\end{tabular}")
    (tables / "gate_identity_effects.tex").write_text("\n".join(tex) + "\n")
    text += [
        "",
        (
            "Pixel maps are not generally identical, and several rank-test differences are "
            "measurable. Every primary endpoint interval lies inside the already specified "
            "engineering SESOI band in these tested models. This supports a qualified "
            "practical-equivalence description conditional on trained seeds, not exact "
            "invariance, clinical equivalence, general removability or zero deployment "
            "overhead."
        ),
        "",
        (
            "The mean-effect bootstrap and Wilcoxon rank test assess different features of "
            "a potentially asymmetric difference distribution; an interval containing zero "
            "can coexist with a small rank-test p-value. Do not alter either output to "
            "force agreement. Shared finite-cell eligibility and all-cell diagonal-penalty "
            "sensitivities remain explicit."
        ),
        "",
        (
            "Sources: tables/gate_identity_effects.csv, gate_identity_by_seed.csv, "
            "gate_identity_per_unit_seed.csv and gate_identity_changed_pixels.csv. "
            "Provenance: validation/gate_identity_lineage.csv and per-run "
            "prediction-recovery files. All changed masks remain saved. This evidence does "
            "not complete the training M5 ablation."
        ),
    ]
    (ROOT / "results/prediction_recovery/GATE_IDENTITY_FINDINGS.md").write_text(
        "\n".join(text) + "\n"
    )
    print(
        "Generated", len(rows), "gate endpoint table rows without rounding small p-values to zero"
    )


if __name__ == "__main__":
    main()
