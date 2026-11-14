# ruff: noqa: E501
"""Keys and tables for the restructure revision (branch revision-restructure, 2026-11-09).

Reads only existing result files and writes revision/manuscript/generated/:
- numbers_restructure.tex: \\R{} keys, all prefixed RS.;
- tab_s20.tex, tab_s20_pooled.tex, tab_s20_coverage.tex: Supplementary S20.

It runs no analysis beyond formatting and simple arithmetic on recorded values. Each key's
source is named next to it.

Number format: Boundary F1, DSC and IoU differences at 4 decimals; HD95 and ASSD at 2 (px).

    PYTHONPATH=src:. .venv/bin/python analysis/revision/make_restructure_assets.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "revision/manuscript/generated"
FAM = ROOT / "revision/tables/families"
POST = ROOT / "revision/restructure_posthoc"
DIGITS = {"dsc": 4, "iou": 4, "boundary_f1": 4, "hd95": 2, "assd": 2}
KEYS: dict[str, str] = {}


def num(x: float, d: int, signed: bool = True) -> str:
    text = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    return text.replace("-", "$-$")


def ci(low: float, high: float, d: int) -> str:
    return f"[{num(low, d)}, {num(high, d)}]"


def put(key: str, value: str) -> None:
    if f"RS.{key}" in KEYS:
        raise KeyError(key)
    KEYS[f"RS.{key}"] = value


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


# ---------------------------------------------------------------- 4-decimal versions of 3-decimal keys
def four_decimal_keys() -> None:
    inter = rows(FAM / "expansion_interaction.csv")

    def pooled(effect: str, metric: str) -> dict:
        return next(
            r
            for r in inter
            if r["set"] in ("2x2", "transfer")
            and r["effect"] == effect
            and r["metric"] == metric
            and r["dataset"].startswith("pooled")
        )

    for effect, key in (("decoder without surface", "I.dec"), ("interaction", "I.inter")):
        r = pooled(effect, "boundary_f1")
        put(f"{key}.bf1.est", num(float(r["estimate"]), 4))
        put(f"{key}.bf1.ci", ci(float(r["low"]), float(r["high"]), 4))
    comp = rows(FAM / "expansion_components.csv")
    for component in ("decoder", "distill", "gate", "surface"):
        for metric in ("boundary_f1", "hd95"):
            r = next(
                x
                for x in comp
                if x["dataset"].startswith("pooled")
                and x["component"] == component
                and x["metric"] == metric
            )
            tag = "bf1" if metric == "boundary_f1" else metric
            put(f"K.{component}.{tag}.est", num(float(r["estimate"]), DIGITS[metric]))
            put(
                f"K.{component}.{tag}.ci",
                ci(float(r["low"]), float(r["high"]), DIGITS[metric]),
            )
    con = rows(FAM / "expansion_contrasts.csv")
    for first, second in (
        ("decoder", "unet"),
        ("gsl", "decoder"),
        ("gsl", "full"),
        ("full", "decoder"),
    ):
        r = next(
            x
            for x in con
            if x["metric"] == "boundary_f1" and x["first"] == first and x["second"] == second
        )
        put(f"X.{first}.{second}.bf1.all.delta", num(float(r["all_delta"]), 4))
        put(f"X.{first}.{second}.bf1.all.ci", ci(float(r["all_low"]), float(r["all_high"]), 4))
        put(f"X.{first}.{second}.bf1.fresh.delta", num(float(r["fresh_delta"]), 4))
    half = sorted(
        (float(r["all_high"]) - float(r["all_low"])) / 2
        for r in con
        if r["metric"] == "boundary_f1"
    )
    put("X.halfwidth", f"{half[len(half) // 2]:.4f}")
    dev = rows(FAM / "audit_development_controls.csv")
    r = next(
        x for x in dev if x["metric"] == "boundary_f1" and x["first"] == "UA" and x["second"] == "U"
    )
    put("DEV.UA.bf1.delta", num(float(r["delta"]), 4))
    put("DEV.UA.bf1.ci", ci(float(r["low"]), float(r["high"]), 4))
    screen = json.loads(
        (ROOT / "recovered/followup/results/acdc_context_audit_20261019/screen.json").read_text()
    )["contrasts"]
    dice = [c["dice_delta"] for c in screen]
    put("CTX.dsc.low", f"{min(dice):.4f}")
    put("CTX.dsc.high", f"{max(dice):.4f}")


# ---------------------------------------------------------------- teacher and costs
def teacher_keys() -> None:
    t = json.loads((POST / "teacher_metrics.json").read_text())
    off, internal = t["official_101_150"], t["internal_20_patients_recorded"]
    for metric in ("boundary_f1", "dsc"):
        tag = "bf1" if metric == "boundary_f1" else metric
        put(f"TCH.off.{tag}", f"{off[metric]['mean_all_units']:.4f}")
        put(f"TCH.int.{tag}", f"{internal[metric]:.4f}")
    put("TCH.off.units", str(off["boundary_f1"]["units_all"]))
    timing = json.loads(
        (ROOT / "results/acdc_confirm_eval_posthoc/teacher_cache_timing.json").read_text()
    )
    put("KDC.cache.sec", f"{timing['median_seconds']:.0f}")
    put("KDC.cache.reps", str(len(timing["runs"])))
    # deployed student = encoder + segmentation decoder of the stage-3 decoder-only network
    import torch  # noqa: F401  (build_model needs torch)

    from bds_lite.training import build_model

    config = json.loads(
        (ROOT / "results/revision_runs/acdc_confirm_decoder_seed201/config.json").read_text()
    )
    model = build_model(config)
    deployed = sum(
        p.numel() for name in ("encoder", "seg_decoder") for p in getattr(model, name).parameters()
    )
    bench = json.loads((ROOT / "revision/kd_multiclass/benchmark.json").read_text())["results"]
    teacher = next(r for r in bench if r["dataset"] == "acdc" and r["role"] == "teacher")[
        "parameters"
    ]
    student = next(r for r in bench if r["dataset"] == "acdc" and r["role"] == "student")[
        "parameters"
    ]
    if sum(p.numel() for p in model.parameters()) != student:
        raise ValueError("student parameter count differs from benchmark.json")
    put("KDC.student.deployed", f"{deployed:,}")
    put("KDC.ratio.deployed", f"{teacher / deployed:.1f}")
    put("KDC.ratio.training", f"{teacher / student:.1f}")


# ---------------------------------------------------------------- Supplementary S20
NAMES = {
    "decoder without surface": "D $-$ U",
    "distillation+gate without surface": "D+F+G $-$ D",
    "distillation+gate with surface": "D+F+G+S $-$ D+S",
    "decoder - unet": "D $-$ U",
    "full - unet": "D+F+G+S $-$ U",
    "full - decoder": "D+F+G+S $-$ D",
    "gsl - unet": "U-Net+GSL $-$ U",
    "gsl - full": "U-Net+GSL $-$ D+F+G+S",
    "gsl - decoder": "U-Net+GSL $-$ D",
    "D+KD-band - D": "D+KD-band $-$ D",
}
FAMILY = {
    "stage 1": "Stage 1",
    "ISIC2018 ten-run extension (all seeds)": "ISIC2018 ten-run",
    "stage 2 balanced display": "Stage 2 (3 vs 3)",
    "stage 2 original frozen": "Stage 2 (3 vs 10)",
    "stage 3": "Stage 3",
}
DS = {"isic2018": "ISIC2018", "acdc": "ACDC", "synapse": "Synapse", "pooled": "pooled"}
METRIC = {
    "boundary_f1": "Boundary F1",
    "dsc": "DSC",
    "iou": "IoU",
    "hd95": "HD95 (px)",
    "assd": "ASSD (px)",
}


def truthy(x: str) -> bool:
    return x == "True"


def s20() -> None:
    data = rows(POST / "shared_component.csv")
    summary = json.loads((POST / "shared_component.json").read_text())
    cov = rows(POST / "coverage.csv")
    for r in data:
        for k in r:
            if k not in ("family", "dataset", "contrast", "metric") and r[k] not in (
                "",
                "True",
                "False",
            ):
                r[k] = float(r[k])
    comparisons = [r for r in data if r["dataset"] != "pooled"]
    bf1 = [r for r in comparisons if r["metric"] == "boundary_f1"]
    untruncated_intervals(data)

    def pick(family, dataset, contrast, metric="boundary_f1"):
        return next(
            r
            for r in data
            if r["family"] == family
            and r["dataset"] == dataset
            and r["contrast"] == contrast
            and r["metric"] == metric
        )

    put("n.comparisons", str(len(comparisons)))
    put("n.bf1", str(len(bf1)))
    for k in ("k05", "k1", "k15"):
        put(f"n.trunc.{k}", str(sum(truthy(r[f"truncated_{k}"]) for r in comparisons)))
        put(f"n.trunc.bf1.{k}", str(sum(truthy(r[f"truncated_{k}"]) for r in bf1)))
    isic = [r["se_reduction"] for r in bf1 if r["dataset"] == "isic2018"]
    small = [
        r["se_reduction"]
        for r in bf1
        if r["dataset"] in ("acdc", "synapse") and r["family"] != "stage 3"
    ]
    put("red.isic.max", f"{100 * max(isic):.1f}")
    put("red.small.min", f"{100 * min(small):.0f}")
    put("red.small.max", f"{100 * max(small):.0f}")
    s3 = pick("stage 3", "acdc", "D+KD-band - D")
    put("red.s3", f"{100 * s3['se_reduction']:.1f}")
    put("s3.adj.low", num(s3["low_k1"], 4))
    put("s3.adj.ci", ci(s3["low_k1"], s3["high_k1"], 4))
    put("s3.adj.k05.ci", ci(s3["low_k05"], s3["high_k05"], 4))
    put("s3.adj.k15.ci", ci(s3["low_k15"], s3["high_k15"], 4))
    for contrast, key in (
        ("distillation+gate without surface", "tr.nosurf"),
        ("distillation+gate with surface", "tr.surf"),
        ("decoder without surface", "dec"),
    ):
        r = pick("stage 1", "pooled", contrast)
        put(f"{key}.adj.ci", ci(r["low_k1"], r["high_k1"], 4))
        put(f"{key}.adj.k05.ci", ci(r["low_k05"], r["high_k05"], 4))
        put(f"{key}.adj.k15.ci", ci(r["low_k15"], r["high_k15"], 4))
        # untruncated SE^2 - SHARED, computed first (commit bbf2d9e), pooled here from the per-dataset rows
        part = [pick("stage 1", d, contrast) for d in ("isic2018", "acdc", "synapse")]
        w = 1 / np.array([p["untruncated_se_without_shared"] ** 2 for p in part])
        est = np.array([p["delta"] for p in part])
        mean, se = float((w * est).sum() / w.sum()), math.sqrt(1 / w.sum())
        put(f"{key}.untrunc.ci", ci(mean - 1.96 * se, mean + 1.96 * se, 4))
    flips = summary["zero_exclusion_changes_k1"]
    put("n.flips.k1", str(len(flips)))
    put("n.flips.k1.bf1", str(sum(f["metric"] == "boundary_f1" for f in flips)))
    for k, label in (("k0.5", "k05"), ("k1", "k1"), ("k1.5", "k15")):
        put(f"n.excl.{label}", str(summary["zero_exclusion_by_factor"][k]))
    put("n.excl.pre", str(summary["zero_exclusion_by_factor"]["prespecified"]))
    put("n.rows", str(len(data)))
    put("cov.reps", f"{int(float(cov[0]['reps'])):,}")
    pre = [float(c["coverage_prespecified"]) for c in cov]
    adj = [float(c["coverage_adjusted_k1"]) for c in cov]
    put("cov.pre.min", f"{min(pre):.3f}")
    put("cov.pre.max", f"{max(pre):.3f}")
    put("cov.adj.min", f"{min(adj):.3f}")
    put("cov.adj.max", f"{max(adj):.3f}")
    put("cov.n", str(len(cov)))
    pooled_decomposition(data, pick)
    small_rows = [
        r for r in bf1 if r["dataset"] in ("acdc", "synapse") and r["family"] != "stage 3"
    ]
    red_untr = [1 - r["untruncated_se_without_shared"] / r["se"] for r in small_rows]
    put("red.untrunc.small.min", f"{100 * min(red_untr):.0f}")
    put("red.untrunc.small.max", f"{100 * max(red_untr):.0f}")
    # without truncation the adjusted variance falls below SHARED itself when SE^2 < 2 SHARED
    put(
        "n.untrunc.below.shared",
        str(sum(r["se"] ** 2 < 2 * r["shared_variance"] for r in comparisons)),
    )
    se_cols = ("se", "se_k05", "se_k1", "se_k15")
    put(
        "n.nonmono",
        str(
            sum(
                any(r[a] < r[b] for a, b in zip(se_cols, se_cols[1:], strict=False))
                for r in comparisons
            )
        ),
    )
    put("n.above.pre.k1", str(sum(r["se_k1"] > r["se"] for r in comparisons)))
    put("n.above.pre.k15", str(sum(r["se_k15"] > r["se"] for r in comparisons)))

    def mark(r, k):
        return r"\checkmark" if truthy(r[f"excludes_zero_{k}"]) else "--"

    lines = [
        "% generated by analysis/revision/make_restructure_assets.py from revision/restructure_posthoc/shared_component.csv",
        r"\begin{tabular}{lllrlcrlccl}",
        r"\toprule",
        r" & & & & \multicolumn{2}{c}{Prespecified} & \multicolumn{5}{c}{Shared term removed (post hoc)} \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-11}",
        r"Family & Dataset & Contrast & $\Delta$ & 95\% CI & $\neq0$ & Shared (\% SE$^2$) & 95\% CI ($k=1$) & $\neq0$ ($k=0.5$/$1$/$1.5$) & Trunc. & 95\% CI ($k=1$, untrunc.) \\",
        r"\midrule",
    ]
    order = [
        "stage 1",
        "ISIC2018 ten-run extension (all seeds)",
        "stage 2 balanced display",
        "stage 2 original frozen",
        "stage 3",
    ]
    for family in order:
        for r in [x for x in data if x["family"] == family and x["metric"] == "boundary_f1"]:
            pooled = r["dataset"] == "pooled"
            share = "--" if pooled else f"{100 * r['shared_fraction_of_se2']:.0f}"
            pre = r"\checkmark" if truthy(r["pre_excludes_zero"]) else "--"
            lines.append(
                f"{FAMILY[family]} & {DS[r['dataset']]} & {NAMES[r['contrast']]} & {num(r['delta'], 4)} & "
                f"{ci(r['low'], r['high'], 4)} & {pre} & {share} & {ci(r['low_k1'], r['high_k1'], 4)} & "
                f"{mark(r, 'k05')}/{mark(r, 'k1')}/{mark(r, 'k15')} & {'yes' if truthy(r['truncated_k1']) else 'no'} & "
                f"{ci(r['low_untr'], r['high_untr'], 4)} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [
        r"\midrule",
        r"\multicolumn{11}{l}{\emph{Stage 3, secondary metrics (descriptive)}} \\",
    ]
    for metric in ("dsc", "iou", "hd95", "assd"):
        r = pick("stage 3", "acdc", "D+KD-band - D", metric)
        d = DIGITS[metric]
        pre = r"\checkmark" if truthy(r["pre_excludes_zero"]) else "--"
        lines.append(
            f"Stage 3 & ACDC & {METRIC[metric]} & {num(r['delta'], d)} & {ci(r['low'], r['high'], d)} & {pre} & "
            f"{100 * r['shared_fraction_of_se2']:.0f} & {ci(r['low_k1'], r['high_k1'], d)} & "
            f"{mark(r, 'k05')}/{mark(r, 'k1')}/{mark(r, 'k15')} & {'yes' if truthy(r['truncated_k1']) else 'no'} & "
            f"{ci(r['low_untr'], r['high_untr'], d)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s20.tex").write_text("\n".join(lines) + "\n")

    other = [f for f in flips if f["metric"] != "boundary_f1"]
    put(
        "flips.other",
        "; ".join(f"{FAMILY[f['family']]} {DS[f['dataset']]} {METRIC[f['metric']]}" for f in other)
        or "none",
    )
    cov_lines = [
        "% generated by analysis/revision/make_restructure_assets.py from revision/restructure_posthoc/coverage.csv",
        r"\begin{tabular}{llllrr}",
        r"\toprule",
        r"Family & Dataset & Contrast & Runs; units & Prespecified & Shared term removed \\",
        r"\midrule",
    ]
    for c in cov:
        cov_lines.append(
            f"{FAMILY[c['family']]} & {DS[c['dataset']]} & {NAMES[c['contrast']]} & "
            f"{int(float(c['n_a']))}+{int(float(c['n_b']))}; {int(float(c['units']))} & "
            f"{float(c['coverage_prespecified']):.3f} & {float(c['coverage_adjusted_k1']):.3f} \\\\"
        )
    cov_lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s20_coverage.tex").write_text("\n".join(cov_lines) + "\n")


def untruncated_intervals(data: list[dict]) -> None:
    """Add low_untr/high_untr from SE^2 - SHARED without truncation (the first computation,
    commit bbf2d9e): pre-specified t df per comparison; pooled rows by inverse variance with
    normal intervals."""
    for r in data:
        if r["dataset"] != "pooled":
            q = stats.t.ppf(0.975, r["df"])
            r["low_untr"] = r["delta"] - q * r["untruncated_se_without_shared"]
            r["high_untr"] = r["delta"] + q * r["untruncated_se_without_shared"]
    for r in data:
        if r["dataset"] == "pooled":
            part = [
                x
                for x in data
                if x["family"] == r["family"]
                and x["contrast"] == r["contrast"]
                and x["metric"] == r["metric"]
                and x["dataset"] != "pooled"
            ]
            w = 1 / np.array([p["untruncated_se_without_shared"] ** 2 for p in part])
            mean = float((w * np.array([p["delta"] for p in part])).sum() / w.sum())
            se = math.sqrt(1 / w.sum())
            r["low_untr"], r["high_untr"] = mean - 1.96 * se, mean + 1.96 * se


TRUNCATED = {
    (False, False): "none",
    (True, False): "between-run",
    (False, True): "test-unit",
    (True, True): "both",
}


def pooled_decomposition(data: list[dict], pick) -> None:
    """Supplementary S20 table: per-dataset SE, inverse-variance weight and estimate behind the
    pooled D+F+G - D Boundary F1 interval at k = 0 (pre-specified), 0.5, 1, 1.5, and untruncated
    at k = 1. Fails unless the pooled intervals recomputed here equal shared_component.csv."""
    contrast = "distillation+gate without surface"
    part = [pick("stage 1", d, contrast) for d in ("isic2018", "acdc", "synapse")]
    pooled = pick("stage 1", "pooled", contrast)
    variants = [
        ("Prespecified ($k=0$)", "", "k0", 0.0),
        ("Shared term removed, $k=0.5$", "_k05", "k05", 0.5),
        ("Shared term removed, $k=1$", "_k1", "k1", 1.0),
        ("Shared term removed, $k=1.5$", "_k15", "k15", 1.5),
        (
            "Shared term removed, $k=1$, without truncation (first computation)",
            "_untr",
            "untr",
            1.0,
        ),
    ]
    lines = [
        "% generated by analysis/revision/make_restructure_assets.py from revision/restructure_posthoc/shared_component.csv",
        r"\begin{tabular}{llcrrrl}",
        r"\toprule",
        r" & Dataset & Truncated & SE & Weight (\%) & $\Delta$ & 95\% CI \\",
        r"\midrule",
    ]
    for title, tag, key, k in variants:
        se_col = "untruncated_se_without_shared" if key == "untr" else f"se{tag}"
        ses = np.array([p[se_col] for p in part])
        w = 1 / ses**2
        share = w / w.sum()
        mean = float((share * np.array([p["delta"] for p in part])).sum())
        se = math.sqrt(1 / w.sum())
        low, high = mean - 1.96 * se, mean + 1.96 * se
        if max(abs(low - pooled[f"low{tag}"]), abs(high - pooled[f"high{tag}"])) > 1e-12:
            raise ValueError(f"pooled decomposition does not reproduce {key}")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{title}}}}} \\")
        for p, s_, sh in zip(part, ses, share, strict=True):
            if key in ("k0", "untr"):
                trunc = "--"
            else:
                shared = k * p["shared_variance"]
                trunc = TRUNCATED[(p["se_run"] ** 2 < shared, p["se_unit"] ** 2 < shared)]
            lines.append(
                f" & {DS[p['dataset']]} & {trunc} & {s_:.4f} & {100 * sh:.1f} & "
                f"{num(p['delta'], 4)} & {ci(p[f'low{tag}'], p[f'high{tag}'], 4)} \\\\"
            )
            if p["dataset"] == "acdc":
                put(f"pd.acdc.se.{key}", f"{s_:.4f}")
                put(f"pd.acdc.w.{key}", f"{100 * sh:.0f}")
        lines.append(
            rf" & \emph{{pooled}} & & {se:.4f} & 100.0 & {num(mean, 4)} & {ci(low, high, 4)} \\"
        )
        put(f"pd.pooled.est.{key}", num(mean, 4))
        put(f"pd.pooled.se.{key}", f"{se:.4f}")
        lines.append(r"\addlinespace")
    acdc = part[1]
    for name, value in (
        ("run", acdc["se_run"] ** 2),
        ("unit", acdc["se_unit"] ** 2),
        ("shared", acdc["shared_variance"]),
        ("floor.k15", 1.5 * acdc["shared_variance"]),
        ("se2.k0", acdc["se"] ** 2),
        ("var.untr", acdc["untruncated_se_without_shared"] ** 2),
    ):
        put(f"pd.acdc.{name}", f"{1e5 * value:.2f}")
    lines[-1:] = [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s20_pooled.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    four_decimal_keys()
    teacher_keys()
    s20()
    lines = ["% generated by analysis/revision/make_restructure_assets.py; do not edit"]
    lines += [rf"\expandafter\def\csname R@{k}\endcsname{{{v}}}" for k, v in sorted(KEYS.items())]
    (GEN / "numbers_restructure.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(KEYS)} keys")


if __name__ == "__main__":
    main()
