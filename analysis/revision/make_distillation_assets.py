# ruff: noqa: E501
"""Keys, tables and figure for the external-teacher distillation studies (second-round revision).

Reads existing result files for presentation; no inference, training or contrast analysis.
The comparison-table helper recomputes descriptive absolute means/SDs only:
  results/kd_unified/{contrasts,costs}.csv                exploratory D+KD-band study (8ec14fe)
  results/kd_acdc_synapse/benchmark.json                    teacher and student parameter counts
  results/acdc_confirm/results/{results.json,contrasts.csv,per_class.csv,per_seed.csv}
  results/acdc_confirm/seed_correlation_supplement.json   (Dev-3)
  results/acdc_confirm/{preprocessing_dev_gate,preprocessing_impact,official_test_integrity,
                         official_test_preprocessing}.json, PROTOCOL_LOG.md (timestamp entry)
  protocols/acdc_confirm/queue_acdc_confirm.yaml                         registered sources (Dev-1 counts)
  results/audit/experiment_c_blinding_check/experiment_c_status.json   (D7 counts)
  results/tables/families/{expansion_interaction,expansion_contrasts,historical_april}.csv
                                                           four-decimal claim-boundary A1 keys
Writes paper/generated/numbers_kd.tex (+ keys_kd.json), tab_kd.tex,
tab_s_kd_all.tex, tab_s_kd_costs.tex, tab_s_confirm.tex, tab_s_confirm_perclass.tex,
and paper/figures/kd_confirmation.{pdf,png} in the style of make_narrative_assets.py and
make_expansion_assets.py.

Usage: PYTHONPATH=src:. .venv/bin/python analysis/revision/make_distillation_assets.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
from make_manuscript_assets import prose_words  # noqa: E402

GEN = ROOT / "paper/generated"
FIG = ROOT / "paper/figures"
KDU = ROOT / "results/kd_unified"
CONF = ROOT / "results/acdc_confirm"
FAM = ROOT / "results/tables/families"
DS = {"isic2018": "ISIC2018", "acdc": "ACDC", "synapse": "Synapse"}
METRICS = ["dsc", "iou", "boundary_f1", "hd95", "assd"]
MLABEL = {
    "dsc": "DSC",
    "iou": "IoU",
    "boundary_f1": "Boundary F1",
    "hd95": "HD95 (px)",
    "assd": "ASSD (px)",
}
# Palette and style of make_narrative_assets.py / make_expansion_assets.py; D+KD-band takes the
# next categorical slot (validated with the dataviz palette checker, light surface).
INK, BLUE, MUTED, BAND = "#2b2b2b", "#2a78d6", "#7d7c78", "#e9e8e4"
KD_COLOR = "#eda100"
KEYS: dict[str, str] = {}


def put(key: str, value: str) -> None:
    if key in KEYS and KEYS[key] != value:
        raise ValueError(f"key {key} defined twice with different values")
    KEYS[key] = value


def num(value: float, digits: int = 4, sign: bool = True) -> str:
    text = f"{float(value):+.{digits}f}" if sign else f"{float(value):.{digits}f}"
    return text.replace("-", "$-$")


def digits(metric: str) -> int:
    return 2 if metric in ("hd95", "assd") else 4


def ci(low: float, high: float, d: int = 4) -> str:
    return f"[{num(low, d)}, {num(high, d)}]"


def pfmt(p: float) -> str:
    p = float(p)
    if p < 0.001:
        mantissa, exponent = f"{p:.1e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"{p:.3f}"


def read_csv(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def one(rows: list[dict], **match) -> dict:
    hits = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    if len(hits) != 1:
        raise ValueError(f"expected one row for {match}, found {len(hits)}")
    return hits[0]


def reading(low: float, high: float) -> str:
    """Manuscript reading for secondary results (claim boundary A4, author decision 8)."""
    return "interval excludes zero" if low > 0 or high < 0 else "not established"


# ---------------------------------------------------------------- exploratory KD (A2)
def exploratory() -> dict[str, dict]:
    rows = read_csv(KDU / "contrasts.csv")
    out = {}
    for ds in DS:
        ref = "1-10" if ds == "isic2018" else "1-3"
        for metric in METRICS:
            r = one(
                rows,
                dataset=ds,
                comparison="KD-band - decoder-only",
                reference_seeds=ref,
                metric=metric,
            )
            d = digits(metric)
            base = f"KD.{ds}.{metric}"
            put(base + ".delta", num(r["delta"], d))
            put(base + ".ci", ci(r["low"], r["high"], d))
            put(base + ".p", pfmt(r["p"]))
            if metric == "boundary_f1":
                if r["primary"] != "True":
                    raise ValueError(f"{ds} Boundary F1 row is not the primary row")
                put(base + ".holm", f"{float(r['p_holm_over_datasets']):.3f}")
                put(base + ".na", r["n_a"])
                put(base + ".nb", r["n_b"])
                out[ds] = r
    # Seed-matched ISIC2018 sensitivity (decoder seeds 1-3), descriptive only.
    r = one(
        rows,
        dataset="isic2018",
        comparison="KD-band - decoder-only",
        reference_seeds="1-3",
        metric="boundary_f1",
    )
    put("KD.isic2018.matched.delta", num(r["delta"]))
    put("KD.isic2018.matched.ci", ci(r["low"], r["high"]))
    return out


def balanced_exploratory() -> dict[str, dict]:
    """Read the explicitly post hoc display; preserve every frozen KD binding."""
    rows = read_csv(ROOT / "results/kd_posthoc_balanced/contrasts.csv")
    out = {}
    for ds in DS:
        r = one(rows, dataset=ds, analysis="post hoc balanced presentation")
        base = f"KDB.{ds}.boundary_f1"
        put(base + ".delta", num(r["delta"]))
        put(base + ".ci", ci(r["low"], r["high"]))
        put(base + ".p", pfmt(r["p"]))
        put(base + ".holm", f"{float(r['p_holm_over_datasets']):.3f}")
        put(base + ".na", r["n_a"])
        put(base + ".nb", r["n_b"])
        out[ds] = r
    return out


def tab_kd_all(rows_all: list[dict]) -> None:
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Metric & D+KD-band & Decoder-only & $\Delta$ [95\% CI] & $p$ \\",
        r"\midrule",
    ]
    for ds in DS:
        ref = "1-10" if ds == "isic2018" else "1-3"
        for metric in METRICS:
            r = one(
                rows_all,
                dataset=ds,
                comparison="KD-band - decoder-only",
                reference_seeds=ref,
                metric=metric,
            )
            d = digits(metric)
            lines.append(
                f"{DS[ds] if metric == 'dsc' else ''} & {MLABEL[metric]} & {num(r['arm_a_mean'], d, False)} & "
                f"{num(r['arm_b_mean'], d, False)} & {num(r['delta'], d)} {ci(r['low'], r['high'], d)} & {pfmt(r['p'])} \\\\"
            )
        if ds != "synapse":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_kd_all.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- confirmation (A3, A4, A5, D4)
def confirmation() -> dict:
    res = json.loads((CONF / "results/results.json").read_text())
    prim = res["primary"]["contrast"]
    dec = res["primary"]["decision"]
    if dec["outcome"] != "success" or dec["size"] != "of unclear size":
        raise ValueError("unexpected registered decision")
    put("CF.delta", num(prim["delta"]))
    put("CF.low", num(prim["low"]))
    put("CF.high", num(prim["high"]))
    put("CF.ci", ci(prim["low"], prim["high"]))
    put("CF.p", pfmt(prim["p"]))
    put("CF.kd.mean", num(prim["arm_a_mean"], 4, False))
    put("CF.dec.mean", num(prim["arm_b_mean"], 4, False))
    put("CF.kd.sd", num(prim["arm_a_sd"], 4, False))
    put("CF.dec.sd", num(prim["arm_b_sd"], 4, False))
    put("CF.n", str(prim["n_a"]))
    if prim["n_a"] != prim["n_b"]:
        raise ValueError("unequal arms")
    put("CF.seed.first", str(min(res["primary"]["seeds"]["kd_band"])))
    put("CF.seed.last", str(max(res["primary"]["seeds"]["kd_band"])))
    power = res["achieved_power"]
    put("CF.power", f"{power['delta_0.015_sd_bound_90']:.3f}")
    put("CF.units", str(power["units"]))
    cons = res["consistency_boundary_f1"]
    put("CF.above", str(int(cons["kd_runs_above_decoder_mean"])))
    put("CF.overlap", num(cons["min_kd_minus_max_decoder"]))
    hours = res["training_epoch_hours"]
    put("CF.hours.kd", f"{hours['kd_band']:.2f}")
    put("CF.hours.dec", f"{hours['decoder']:.2f}")
    put("CF.hours.diff", f"{hours['kd_band'] - hours['decoder']:.2f}")
    put("CF.hours.pct", f"{100 * (hours['kd_band'] / hours['decoder'] - 1):.1f}")

    rows = read_csv(CONF / "results/contrasts.csv")
    for metric in METRICS:
        r = one(rows, analysis="replication seeds, primary set", metric=metric)
        d = digits(metric)
        put(f"CF.{metric}.delta", num(r["delta"], d))
        put(f"CF.{metric}.ci", ci(r["low"], r["high"], d))
        put(f"CF.{metric}.reading", reading(float(r["low"]), float(r["high"])))
    paired = one(
        rows,
        analysis="replication seeds, primary set; seed-paired sensitivity",
        metric="boundary_f1",
    )
    put("CF.paired.delta", num(paired["delta"]))
    put("CF.paired.ci", ci(paired["low"], paired["high"]))
    explo = one(rows, analysis="exploratory seeds 1-3, official test", metric="boundary_f1")
    put("CF.explo.delta", num(explo["delta"]))
    put("CF.explo.ci", ci(explo["low"], explo["high"]))
    internal = one(
        rows, analysis="replication seeds, internal 20-patient split (seen)", metric="boundary_f1"
    )
    put("CF.internal.delta", num(internal["delta"]))
    put("CF.internal.ci", ci(internal["low"], internal["high"]))
    corr = json.loads((CONF / "seed_correlation_supplement.json").read_text())
    r = [x for x in corr["rows"] if x["split"] == "official_test" and x["metric"] == "boundary_f1"]
    put("CF.seedcorr", num(r[0]["seed_correlation"], 3))

    per_class = read_csv(CONF / "results/per_class.csv")
    names = {"RV": "RV", "myocardium": "Myo", "LV": "LV"}
    for structure, short in names.items():
        for metric in ("dsc", "boundary_f1", "hd95"):
            r = one(per_class, structure=structure, metric=metric)
            d = digits(metric)
            put(f"CF.{short}.{metric}.delta", num(r["delta"], d))
            put(f"CF.{short}.{metric}.ci", ci(r["low"], r["high"], d))
            put(f"CF.{short}.{metric}.reading", reading(float(r["low"]), float(r["high"])))

    per_seed = read_csv(CONF / "results/per_seed.csv")
    values = {
        arm: [
            float(r["value"])
            for r in per_seed
            if r["arm"] == arm and r["split"] == "official_test" and r["metric"] == "boundary_f1"
        ]
        for arm in ("kd_band", "decoder")
    }
    for arm, key in (("kd_band", "kd"), ("decoder", "dec")):
        if len(values[arm]) != prim["n_a"]:
            raise ValueError(f"per-seed count mismatch for {arm}")
        put(f"CF.{key}.min", num(min(values[arm]), 4, False))
        put(f"CF.{key}.max", num(max(values[arm]), 4, False))
    if (
        abs((min(values["kd_band"]) - max(values["decoder"])) - cons["min_kd_minus_max_decoder"])
        > 1e-12
    ):
        raise ValueError("per-seed values do not reproduce the recorded separation")
    return {"res": res, "rows": rows, "per_class": per_class, "values": values}


def tab_confirm(c: dict) -> None:
    rows = c["rows"]
    lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Analysis & Metric & $\Delta$ [95\% CI] & Reading in the article \\",
        r"\midrule",
    ]
    prim = c["res"]["primary"]["contrast"]
    lines.append(
        rf"Primary (confirmatory), seeds 201--217, 50 held-out patients & Boundary F1 & {num(prim['delta'])} {ci(prim['low'], prim['high'])} & Success, of unclear size \\"
    )
    lines.append(r"\addlinespace")
    for metric in ("dsc", "iou", "hd95", "assd"):
        r = one(rows, analysis="replication seeds, primary set", metric=metric)
        d = digits(metric)
        lines.append(
            f"Secondary, same runs and patients & {MLABEL[metric]} & {num(r['delta'], d)} {ci(r['low'], r['high'], d)} & {reading(float(r['low']), float(r['high']))} \\\\"
        )
    lines.append(r"\addlinespace")
    paired = one(
        rows,
        analysis="replication seeds, primary set; seed-paired sensitivity",
        metric="boundary_f1",
    )
    explo = one(rows, analysis="exploratory seeds 1-3, official test", metric="boundary_f1")
    internal = one(
        rows, analysis="replication seeds, internal 20-patient split (seen)", metric="boundary_f1"
    )
    lines += [
        rf"Seed-paired sensitivity (17 pairs) & Boundary F1 & {num(paired['delta'])} {ci(paired['low'], paired['high'])} & descriptive \\",
        rf"Exploratory models (seeds 1--3) on the held-out patients & Boundary F1 & {num(explo['delta'])} {ci(explo['low'], explo['high'])} & descriptive, not a confirmation \\",
        rf"Replication seeds on the internal 20 patients (already seen) & Boundary F1 & {num(internal['delta'])} {ci(internal['low'], internal['high'])} & seed replication only \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (GEN / "tab_s_confirm.tex").write_text("\n".join(lines) + "\n")
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Structure & DSC & Boundary F1 & HD95 (px) \\",
        r"\midrule",
    ]
    for structure, label in (
        ("RV", "Right ventricle"),
        ("myocardium", "Myocardium"),
        ("LV", "Left ventricle"),
    ):
        cells = []
        for metric in ("dsc", "boundary_f1", "hd95"):
            r = one(c["per_class"], structure=structure, metric=metric)
            d = digits(metric)
            cells.append(f"{num(r['delta'], d)} {ci(r['low'], r['high'], d)}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_confirm_perclass.tex").write_text("\n".join(lines) + "\n")


def tab_kd_main(explo: dict[str, dict], c: dict) -> None:
    prim = c["res"]["primary"]["contrast"]
    tests = {
        "isic2018": "official, 1,000 images",
        "acdc": "internal, 20 patients",
        "synapse": "internal, 12 cases",
    }
    lines = [
        r"\begin{tabular}{llcclcc}",
        r"\toprule",
        r"Stage & Dataset & Runs & Test set & $\Delta$ Boundary F1 [95\% CI] & $p$ & Holm $p$ \\",
        r"\midrule",
    ]
    for ds, r in explo.items():
        lines.append(
            f"2. KD exploration & {DS[ds]} & {r['n_a']} vs {r['n_b']} & {tests[ds]} & {num(r['delta'])} {ci(r['low'], r['high'])} & "
            f"{pfmt(r['p'])} & {float(r['p_holm_over_datasets']):.3f} \\\\"
        )
    lines += [
        r"\addlinespace",
        rf"3. ACDC confirmation & ACDC & {prim['n_a']} vs {prim['n_b']} & held-out, 50 patients & {num(prim['delta'])} {ci(prim['low'], prim['high'])} & {pfmt(prim['p'])} & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (GEN / "tab_kd.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- costs (D8)
def costs() -> None:
    bench = json.loads((ROOT / "results/kd_acdc_synapse/benchmark.json").read_text())
    params = {}
    for e in bench["results"]:
        params[(e["dataset"], e["role"])] = int(e["parameters"])
    put("KDC.acdc.teacher.params", f"{params[('acdc', 'teacher')]:,}")
    put("KDC.acdc.student.params", f"{params[('acdc', 'student')]:,}")
    put("KDC.acdc.ratio", f"{params[('acdc', 'teacher')] / params[('acdc', 'student')]:.0f}")
    rows = read_csv(KDU / "costs.csv")
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Run & Epochs & Median epoch (s) & Summed epoch time (h) \\",
        r"\midrule",
    ]
    for r in rows:
        run = (
            r["run"]
            .replace("_", r"\_")
            .replace(" (existing decoder-only reference)", r" (decoder-only reference)")
        )
        lines.append(
            f"{run} & {r['epochs']} & {float(r['median_epoch_seconds']):.2f} & {float(r['sum_epoch_hours']):.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_kd_costs.tex").write_text("\n".join(lines) + "\n")
    teacher = one(rows, run="acdc_kd_teacher_seed100")
    put("KDC.acdc.teacher.hours", f"{float(teacher['sum_epoch_hours']):.3f}")
    put("KDC.acdc.teacher.epoch", f"{float(teacher['median_epoch_seconds']):.2f}")
    # Teacher cache size from the registered shape (pre-registration section 8): 1350 x 4 x 224 x 224 float16.
    prereg = (ROOT / "protocols/acdc_confirm/acdc_confirmation.md").read_text()
    m = re.search(r"\((\d+) × (\d+) × (\d+) × (\d+), float16\)", prereg)
    if not m:
        raise ValueError("teacher cache shape not found in the pre-registration")
    size = np.prod([int(x) for x in m.groups()]) * 2 / 1e9
    put("KDC.acdc.cache.gb", f"{size:.2f}")
    put("KDC.acdc.cache.slices", f"{int(m.group(1)):,}")


# ---------------------------------------------------------------- deviations, data, timestamp, Experiment C
def provenance() -> None:
    queue = yaml.safe_load((ROOT / "protocols/acdc_confirm/queue_acdc_confirm.yaml").read_text())
    counts = {len(r["sources"]) for r in queue["runs"]}
    if len(counts) != 1:
        raise ValueError("runs register different source counts")
    registered = counts.pop()
    study = {
        s["path"]
        for s in queue["runs"][0]["sources"]
        if s["path"].startswith("scripts/acdc_confirm/")
    }
    put("DEV.registered", str(registered))
    put("DEV.study", str(len(study)))
    put("DEV.recorded", str(registered - len(study)))
    put("DEV.runs", str(len(queue["runs"])))

    gate = json.loads((CONF / "preprocessing_dev_gate.json").read_text())
    put("PRE.gate.diff", f"{gate['image_max_abs_diff']:.4f}")
    impact = json.loads((CONF / "preprocessing_impact.json").read_text())
    mad = impact["max_abs_run_level_difference"]
    put("PRE.impact.bf1", f"{mad['boundary_f1']:.5f}")
    put("PRE.impact.dsc", f"{mad['dsc']:.5f}")
    integ = json.loads((CONF / "official_test_integrity.json").read_text())
    pre = json.loads((CONF / "official_test_preprocessing.json").read_text())
    if (
        not integ["passed"]
        or integ["patients"] != pre["patients"]
        or integ["slices"] != pre["slices"]
    ):
        raise ValueError("official test integrity and preprocessing records disagree")
    put("OT.patients", str(integ["patients"]))
    put("OT.slices", f"{integ['slices']:,}")
    put("PRE.gate.slices", f"{gate['slices']:,}")
    put("PRE.impact.patients", str(impact["validation_patients"]))
    shift = impact["kd_minus_decoder_validation_contrast"]["boundary_f1"]["shift"]
    put("PRE.impact.shift.bf1", num(shift, 5))
    gates = read_csv(FAM / "gate_structure.csv")
    put("GATE.labels.runmax", f"{100 * max(float(r['labels_changed']) for r in gates):.2f}")

    log = (ROOT / "protocols/acdc_confirm/PROTOCOL_LOG.md").read_text()
    entry = [line for line in log.splitlines() if "attestation VERIFIED" in line]
    if len(entry) != 1:
        raise ValueError("expected one verified-timestamp entry")
    text = entry[0]
    put(
        "TS.block",
        re.search(r"block header from two independent block explorers.*?height (\d+)", text).group(
            1
        ),
    )
    put(
        "TS.blocktime",
        re.search(r"Block header time (\S+?Z)", text)
        .group(1)
        .replace("T", " ")
        .replace("Z", " UTC"),
    )
    launch = [line for line in log.splitlines() if "after the proof commit" in line]
    put(
        "TS.launch",
        re.search(r"started at (\S+?Z) after", launch[0])
        .group(1)
        .replace("T", " ")
        .replace("Z", " UTC"),
    )
    put(
        "TS.testaccess",
        re.search(r"first test-set access at (\S+?Z)", text)
        .group(1)
        .replace("T", " ")
        .replace("Z", " UTC"),
    )

    def utc(stamp: str) -> datetime:
        return datetime.strptime(stamp.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")

    gap = utc(KEYS["TS.testaccess"]) - utc(KEYS["TS.blocktime"])
    put("TS.gap.h", f"{gap.total_seconds() / 3600:.1f}")
    lag = utc(KEYS["TS.blocktime"]) - utc(KEYS["TS.launch"])
    put("TS.lag.min", f"{lag.total_seconds() / 60:.0f}")

    status = json.loads(
        (ROOT / "results/audit/experiment_c_blinding_check/experiment_c_status.json").read_text()
    )
    put("EXC.registered", str(status["registered_runs"]))
    put("EXC.completed", str(status["completed"]))
    put("EXC.incomplete", str(status["started_incomplete"]))
    put("EXC.never", str(status["never_started"]))
    put("EXC.transcripts", str(status["transcript_scan"]["files"]))
    put("EXC.reads", str(status["access_evidence"]["final_metrics_read_after_write"]))
    put("EXC.calls", f"{status['transcript_scan']['tool_calls']:,}")
    if (
        status["completed"] + status["started_incomplete"] + status["never_started"]
        != status["registered_runs"]
    ):
        raise ValueError("Experiment C counts do not add up")


# ---------------------------------------------------------------- claim-boundary A1 at four decimals
def in_network() -> None:
    rows = read_csv(FAM / "expansion_interaction.csv")

    def pooled(effect: str, metric: str) -> dict:
        hits = [
            r
            for r in rows
            if r["set"] == "transfer"
            and r["effect"] == effect
            and r["metric"] == metric
            and r["dataset"].startswith("pooled")
            and r["seeds"] == "1-3"
        ]
        if len(hits) != 1:
            raise ValueError(f"pooled row {effect} {metric}")
        return hits[0]

    for effect, key in (
        ("distillation+gate without surface", "nosurf"),
        ("distillation+gate with surface", "surf"),
    ):
        for metric in ("boundary_f1", "dsc", "hd95"):
            r = pooled(effect, metric)
            d = digits(metric)
            put(f"A1.{key}.{metric}.est", num(r["estimate"], d))
            put(f"A1.{key}.{metric}.ci", ci(r["low"], r["high"], d))
        r = pooled(effect, "boundary_f1")
        put(f"A1.{key}.q", re.search(r"Q p=([0-9.]+)", r["dataset"]).group(1))
    acdc = [
        r
        for r in rows
        if r["set"] == "transfer"
        and r["effect"] == "distillation+gate without surface"
        and r["metric"] == "boundary_f1"
        and r["dataset"] == "acdc"
        and r["seeds"] == "1-3"
    ]
    put("A1.acdc.nosurf.est", num(acdc[0]["estimate"]))
    put("A1.acdc.nosurf.ci", ci(acdc[0]["low"], acdc[0]["high"]))
    con = read_csv(FAM / "expansion_contrasts.csv")
    r = one(con, metric="boundary_f1", first="full", second="decoder")
    put("A1.full.all.delta", num(r["all_delta"]))
    put("A1.full.all.ci", ci(r["all_low"], r["all_high"]))
    put("A1.full.fresh.delta", num(r["fresh_delta"]))
    put("A1.full.fresh.ci", ci(r["fresh_low"], r["fresh_high"]))
    hist = read_csv(FAM / "historical_april.csv")
    h = one(hist, id="H5", metric="boundary_f1", dataset="pooled")
    put("A1.hist.delta", num(h["delta"]))
    put("A1.hist.ci", ci(h["low"], h["high"]))
    put("A1.hist.holm", f"{float(h['p_holm']):.1f}")


# ---------------------------------------------------------------- second-pass keys and table variants
def second_pass() -> None:
    """Keys that replace hand-typed literals, and run-level-only variants of two generated tables."""
    res = {
        (r["dataset"], r["model"]): r
        for r in read_csv(ROOT / "results/tables/resource_profile.csv")
    }
    unet, bds = res[("isic2018", "U-Net")], res[("isic2018", "BDS-Lite (deployed)")]
    put("PAR.unet.m", f"{int(unet['params_deployed']) / 1e6:.2f}")
    put("PAR.unet.m1", f"{int(unet['params_deployed']) / 1e6:.1f}")
    put("PAR.gate", f"{int(bds['params_deployed']) - int(unet['params_deployed']):,}")
    put(
        "PAR.flops.pct",
        f"{100 * (float(bds['gflops_b1']) / float(unet['gflops_b1']) - 1):.2f}",
    )
    # split sizes from the tracked locked manifests (ACDC, Synapse) and the official ISIC2018 test
    for name, pattern in (("acdc", r"(patient\d+)"), ("synapse", r"(case\d+)")):
        split = json.loads((ROOT / f"splits/{name}_split.json").read_text())
        for part in ("train", "val", "test"):
            ids = {re.match(pattern, s).group(1) for s in split[part]}
            put(f"DS.{name}.{part}.units", str(len(ids)))
    official = ROOT / "results/official_isic/test/U_seed1/per_case.csv"
    put("DS.isic2018.test.n", f"{len({r['id'] for r in read_csv(official)}):,}")
    # typical half-width of the ten-run Boundary F1 contrasts
    con = [r for r in read_csv(FAM / "expansion_contrasts.csv") if r["metric"] == "boundary_f1"]
    half = sorted((float(r["all_high"]) - float(r["all_low"])) / 2 for r in con)
    put("X.halfwidth", f"{half[len(half) // 2]:.2f}")
    # three-slice control as 0-1 DSC differences, from the committed CTX keys (percentage points)
    audit = (GEN / "numbers_audit.tex").read_text()
    for end in ("low", "high"):
        value = float(re.search(r"R@CTX\.dsc\." + end + r"\\endcsname\{([^}]*)\}", audit).group(1))
        put(f"CTX01.dsc.{end}", f"{value / 100:.3f}")
    # qualitative figure from the retrained set (manifest v3)
    man = json.loads((ROOT / "paper/figures/qualitative_manifest_v3.json").read_text())
    extreme = unstable = 0
    for ds in ("isic2018", "acdc", "synapse"):
        for case in man[ds]["cases"]:
            if case["role"] == "median":
                continue
            sign = 1.0 if "gain" in case["role"] else -1.0
            extreme += 1
            unstable += min(sign * x for x in case["per_seed_delta"]) < 2.0
    put("QUALR.extreme.n", str(extreme))
    put("QUALR.extreme.unstable", str(unstable))
    cases = {c["role"]: c for c in man["isic2018"]["cases"]}
    put(
        "QUALR.isic.gain.dec.spurious",
        str(cases["largest BDS-Lite gain"]["panels"]["M2"]["spurious"]),
    )
    put(
        "QUALR.isic.loss.bds.spurious",
        str(cases["largest BDS-Lite loss"]["panels"]["M6"]["spurious"]),
    )
    # run-level-only variant of the all-arms table (image-level Holm markers removed)
    table = (GEN / "tab_all_arms.tex").read_text()
    table = table.replace("^{*\\dagger}", "^{\\dagger}").replace("$^{*}$", "")
    if "^{*" in table:
        raise ValueError("image-level marker left in the all-arms table")
    (GEN / "tab_all_arms_runlevel.tex").write_text(
        table.replace(
            "% generated by",
            "% derived from tab_all_arms.tex (image-level markers removed); originally generated by",
        )
    )
    lit = (GEN / "tab_lit_compact.tex").read_text()
    lit = lit.replace(", original", " (earlier pipeline)").replace(", retrained", "")
    # Retain source-paper HD nomenclature in active published-context displays.
    lit = lit.replace(r"3D DSC (\%) & HD95", r"3D DSC (\%) & HD$^\dagger$")
    published_table = (GEN / "tab_literature.tex").read_text()
    (GEN / "tab_literature_source_labels.tex").write_text(
        published_table.replace(" & HD95 & ", " & HD (source) & ")
    )
    # Supplement citations refer to the main bibliography, without its own natbib run.
    supp_lit = lit
    for key, citation in {
        "Chen2021TransUNetArxiv": "Chen et al. (2021)",
        "Cao2022SwinUnet": "Cao et al. (2023)",
        "Codella2019Skin": "Codella et al. (2019)",
    }.items():
        supp_lit = supp_lit.replace(r"\citet{" + key + "}", citation)
    (GEN / "tab_lit_full.tex").write_text(supp_lit)
    # Recompute descriptive absolute scores from the existing per-run CSVs only.
    # Frozen contrast analyses and the historical supplementary context stay intact.
    from make_comparison_table import build

    build()
    resource_main_table()
    lb_resource_table()
    central_table()


def resource_main_table() -> None:
    """Select the overhead-focused main profile; preserve all measurements in S19."""
    resource = (GEN / "tab_resource.tex").read_text()
    main_models = {
        "U-Net",
        "BDS-Lite (deployed)",
        "BDS-Lite (gate-collapsed)",
        "Attention U-Net",
    }
    compact = [
        "% selected rows/columns of tab_resource.tex; full measurements retained in S19",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"\toprule",
        r"Model & Deploy params & GFLOPs$^a$ & GPU b1 & GPU b16 \\",
        r"\midrule",
    ]
    profile = {
        r["model"]: r
        for r in read_csv(ROOT / "results/tables/resource_profile.csv")
        if r["dataset"] == "isic2018"
    }
    for line in resource.splitlines():
        if " & " in line and not line.startswith("Model &"):
            cells = line.removesuffix(r" \\").split(" & ")
            if cells[0] in main_models:
                cells[2] = f"{float(profile[cells[0]]['gflops_b1']):.3f}"
                compact.append(" & ".join(cells[i] for i in (0, 1, 2, 3, 5)) + r" \\")
    compact += [r"\bottomrule", r"\end{tabular*}"]
    (GEN / "tab_resource_main.tex").write_text("\n".join(compact) + "\n")


def lb_resource_table() -> None:
    """Keep every recorded LB-UNet profile field accessible in S19."""
    data = {
        row["dataset"]: row
        for row in read_csv(ROOT / "results/tables/resource_profile.csv")
        if row["model"] == "LB-UNet"
    }
    datasets = ("isic2018", "synapse")
    lines = [
        "% existing adapted LB-UNet measurements from results/tables/resource_profile.csv",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Measurement & ISIC2018 & Synapse \\",
        r"\midrule",
    ]
    for label, field, precision in (
        ("Total parameters", "params_total", 0),
        ("Deployed parameters", "params_deployed", 0),
        ("GFLOPs (FlopCounterMode)", "gflops_b1", 6),
        ("GFLOPs (released-graph hooks)", "gflops_b1_released_hook", 6),
        ("CPU batch 1, one thread (ms)", "cpu_ms_b1_1thread", 4),
        ("CPU batch 1, all threads (ms)", "cpu_ms_b1_allthreads", 4),
    ):
        values = [f"{float(data[d][field]):,.{precision}f}" for d in datasets]
        lines.append(label + " & " + " & ".join(values) + r" \\")
    for mode in ("fp32", "amp"):
        for batch in (1, 8, 16):
            prefix = f"gpu_{mode}_b{batch}_ms_"
            values = [
                f"${float(data[d][prefix + 'median']):.4f} \\pm {float(data[d][prefix + 'iqr']):.4f}$"
                for d in datasets
            ]
            lines.append(f"GPU {mode}, batch {batch} (ms) & " + " & ".join(values) + r" \\")
    for label, field, precision in (
        ("CUDA kernels, batch 1", "cuda_kernels_b1", 0),
        ("CUDA kernel time, batch 1 (ms)", "cuda_kernel_time_ms_b1", 4),
    ):
        values = [f"{float(data[d][field]):,.{precision}f}" for d in datasets]
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_lb_resource.tex").write_text("\n".join(lines) + "\n")


def central_table() -> None:
    """Format the central pooled table from frozen contrasts; no analysis is run."""
    # The forest plot carries dataset-level Boundary F1; the main table compares
    # pooled endpoints at a readable size. Preserve the complete table in S13.
    interaction = read_csv(FAM / "expansion_interaction.csv")
    central = [
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llr@{}}",
        r"\toprule",
        r"Metric & Contrast & Pooled $\Delta$ [95\% CI] \\",
        r"\midrule",
    ]
    for metric in ("boundary_f1", "dsc", "hd95"):
        for effect, label in (
            ("decoder without surface", "Decoder vs U-Net"),
            ("distillation+gate without surface", "Transfer, no surface loss"),
            ("distillation+gate with surface", "Transfer, with surface loss"),
        ):
            row = next(
                r
                for r in interaction
                if r["effect"] == effect
                and r["metric"] == metric
                and r["dataset"].startswith("pooled")
            )
            d = digits(metric)
            central.append(
                f"{MLABEL[metric] if label == 'Decoder vs U-Net' else ''} & {label} & "
                f"{num(row['estimate'], d)} {ci(row['low'], row['high'], d)} \\\\"
            )
        if metric != "hd95":
            central.append(r"\midrule")
    central += [r"\bottomrule", r"\end{tabular*}"]
    (GEN / "tab_central_main.tex").write_text("\n".join(central) + "\n")


# ---------------------------------------------------------------- figure
def figure(explo: dict[str, dict], c: dict) -> None:
    """Render the frozen comparisons with separate space for labels; no new analysis."""
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.linewidth": 0.5})
    fig = plt.figure(figsize=(7.2, 2.65))
    grid = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.7, 2.4], wspace=0.13)
    left = fig.add_subplot(grid[0])
    labels_ax = fig.add_subplot(grid[1])
    right = fig.add_subplot(grid[2])
    arms = [("decoder", "Decoder-only", BLUE, "o"), ("kd_band", "D+KD-band", KD_COLOR, "v")]
    for y, (arm, _, color, marker) in enumerate(arms):
        values = np.array(c["values"][arm])
        jitter = ((np.arange(len(values)) % 3) - 1) * 0.10
        left.scatter(values, y + jitter, s=15, marker=marker, color=color, linewidths=0.5, zorder=3)
        left.plot([values.mean()] * 2, [y - 0.30, y + 0.30], color=color, lw=1.3, zorder=4)
    left.set_yticks(range(len(arms)), [a[1] for a in arms])
    left.set_ylim(-0.6, len(arms) - 0.4)
    left.set_xlabel("Boundary F1", labelpad=4)
    left.set_title("ACDC run means", loc="left", fontsize=8.5, pad=10)
    left.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    left.tick_params(length=2, width=0.4, labelsize=8)
    left.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        left.spines[side].set_visible(False)

    prim = c["res"]["primary"]["contrast"]
    groups = [
        (
            "Stage 2: Exploratory KD",
            [
                (
                    f"{DS[ds]} ({r['n_a']} vs {r['n_b']})",
                    float(r["delta"]),
                    float(r["low"]),
                    float(r["high"]),
                    "three",
                )
                for ds, r in explo.items()
            ],
        ),
        (
            "Stage 3: ACDC Confirmation",
            [
                (
                    f"ACDC ({prim['n_a']} vs {prim['n_b']})",
                    prim["delta"],
                    prim["low"],
                    prim["high"],
                    "conf",
                )
            ],
        ),
    ]
    styles = {
        "three": dict(marker="o", color=MUTED, fill="white", size=4.5, width=1.0),
        "conf": dict(marker="D", color=INK, fill=INK, size=4.7, width=1.3),
    }
    right.axvspan(-0.01, 0.01, color=BAND, lw=0, zorder=0)
    right.axvline(0, color=MUTED, lw=0.6, zorder=1)
    y = 0.0
    for title, items in groups:
        labels_ax.text(
            1.0,
            y,
            title,
            ha="right",
            va="center",
            fontsize=8,
            color=INK,
            transform=blended_transform_factory(labels_ax.transAxes, labels_ax.transData),
        )
        y += 0.9
        for name, estimate, low, high, kind in items:
            style = styles[kind]
            labels_ax.text(
                1.0,
                y,
                name,
                ha="right",
                va="center",
                fontsize=8,
                color=INK,
                transform=blended_transform_factory(labels_ax.transAxes, labels_ax.transData),
            )
            right.plot(
                [low, high],
                [y, y],
                color=style["color"],
                lw=style["width"],
                solid_capstyle="butt",
                zorder=2,
            )
            right.plot(
                estimate,
                y,
                style["marker"],
                ms=style["size"],
                mfc=style["fill"],
                mec=style["color"],
                mew=0.9,
                zorder=3,
            )
            y += 1.0
        y += 0.5
    for axis in (labels_ax, right):
        axis.set_ylim(y - 0.1, -0.6)
    labels_ax.set_axis_off()
    right.set_yticks([])
    right.set_xlim(-0.07, 0.16)
    right.set_xticks([-0.05, 0.0, 0.05, 0.10, 0.15])
    right.set_xlabel(r"$\Delta$ Boundary F1", labelpad=4)
    right.set_title("Difference from decoder-only", loc="left", fontsize=8.5, pad=10)
    right.tick_params(length=2, width=0.4, labelsize=8)
    for side in ("top", "right", "left"):
        right.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.88, bottom=0.20)
    fig.savefig(FIG / "kd_confirmation.pdf", metadata={"CreationDate": None})
    fig.savefig(FIG / "kd_confirmation.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- word counts
def abstract_words(path: Path) -> int:
    tex = path.read_text(errors="ignore")
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    body = re.sub(r"(?m)(?<!\\)%.*$", "", m.group(1))
    body = re.sub(r"\\R\{[^}]*\}", "N", body)
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?", "", body)
    return len(body.split())


def word_counts() -> None:
    original = ROOT / "recovered/peerj-submission/main.tex"
    revised = ROOT / "paper/main.tex"
    if not original.exists():
        return
    before, after = prose_words(original), prose_words(revised)
    put("WCR.orig", f"{before:,}")
    put("WCR.rev", f"{after:,}")
    put("WCR.pct", f"{100 * (after / before - 1):+.0f}".replace("-", "$-$"))
    put("WCR.abs.orig", str(abstract_words(original)))
    put("WCR.abs.rev", str(abstract_words(revised)))


def main() -> None:
    # Table S53 and the stage-2 figure use the original frozen analysis (ISIC2018: 3 vs 10 runs);
    # the post hoc balanced 3-vs-3 display is computed for its S17 sensitivity keys only.
    explo = exploratory()
    balanced_exploratory()
    tab_kd_all(read_csv(KDU / "contrasts.csv"))
    c = confirmation()
    tab_confirm(c)
    tab_kd_main(explo, c)
    costs()
    provenance()
    in_network()
    second_pass()
    figure(explo, c)
    lines = ["% generated by analysis/revision/make_distillation_assets.py; do not edit"]
    lines += [f"\\expandafter\\def\\csname R@{k}\\endcsname{{{KEYS[k]}}}" for k in sorted(KEYS)]
    (GEN / "numbers_kd.tex").write_text("\n".join(lines) + "\n")
    (GEN / "keys_kd.json").write_text(
        json.dumps({k: KEYS[k] for k in sorted(KEYS)}, indent=0) + "\n"
    )
    print(len(KEYS), "keys")


if __name__ == "__main__":
    main()
