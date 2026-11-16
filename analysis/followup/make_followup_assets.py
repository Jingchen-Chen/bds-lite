"""Manuscript keys, tables and the figure for the follow-up controls and the post hoc analyses.

Reads only committed analysis outputs (no statistics are computed here):
  results/followup/results/{results.json, contrasts.csv, per_class.csv, per_seed.csv}
  results/followup/results/posthoc_{stage_shift,reference_estimators,error_location}.json
Writes paper/generated/numbers_followup.tex (+ keys_followup.json),
tab_followup_main.tex, tab_s_followup_all.tex, tab_s_followup_perclass.tex,
tab_s_reference_estimators.tex, tab_s_error_location.tex, tab_s_stage_shift.tex and
paper/figures/kd_followup.{pdf,png} (the style of Fig. 2, make_distillation_assets.figure).
Sections whose inputs do not exist yet are skipped.

    PYTHONPATH=src:. .venv/bin/python analysis/followup/make_followup_assets.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
from make_distillation_assets import BAND, BLUE, INK, KD_COLOR, MUTED, ci, num, pfmt  # noqa: E402

RES = ROOT / "results/followup/results"
GEN = ROOT / "paper/generated"
FIG = ROOT / "paper/figures"
KEYS: dict[str, str] = {}
MLABEL = {"dsc": "DSC", "iou": "IoU", "boundary_f1": "Boundary F1", "hd95": "HD95", "assd": "ASSD"}
FULL_COLOR, PEER_COLOR = "#1baa8c", "#9b5fc0"
ARM_TEX = {
    "decoder": "D",
    "kd_band": "D+KD-band",
    "kd_full": "D+KD-full",
    "kd_band_peer": "D+KD-band$_{\\mathrm{cmp}}$",
    "decoder_aug": "D$_{\\mathrm{aug}}$",
    "kd_band_aug": "D+KD-band$_{\\mathrm{aug}}$",
}
READING_TEX = {
    "relevant gain": "relevant gain",
    "relevant harm": "relevant harm",
    "no relevant effect": "no relevant effect",
    "gain, size unclear": "gain, size unclear",
    "harm, size unclear": "harm, size unclear",
    "undetermined": "undetermined",
}


def put(key: str, value: str) -> None:
    if key in KEYS and KEYS[key] != value:
        raise ValueError(f"key {key} defined twice with different values")
    KEYS[key] = value


def rfmt(x: float) -> str:
    """Correlation with sign; a value that rounds to zero is printed as 0.00, not -0.00."""
    return "0.00" if abs(x) < 0.005 else f"{x:+.2f}".replace("-", "$-$")


def d(metric: str) -> int:
    return 2 if metric in ("hd95", "assd") else 4


def read_csv(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def comparison_tex(row: dict) -> str:
    first, second = row["comparison"].split(" - ")
    inverse = {
        "D": "decoder",
        "D+KD-band": "kd_band",
        "D+KD-full": "kd_full",
        "D+KD-band (compact teacher)": "kd_band_peer",
        "D, augmented": "decoder_aug",
        "D+KD-band, augmented": "kd_band_aug",
    }
    return f"{ARM_TEX[inverse[first]]} $-$ {ARM_TEX[inverse[second]]}"


# ---------------------------------------------------------------- follow-up results
def followup() -> dict | None:
    path = RES / "results.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    rows = read_csv(RES / "contrasts.csv")
    for r in rows:
        m, key = r["metric"], f"FU.{r['id']}.{r['metric']}"
        put(key + ".delta", num(float(r["delta"]), d(m)))
        put(key + ".ci", ci(float(r["low"]), float(r["high"]), d(m)))
        put(key + ".low", num(float(r["low"]), d(m)))
        put(key + ".high", num(float(r["high"]), d(m)))
        put(key + ".p", pfmt(float(r["p"])))
        put(key + ".reading", r["reading"])
        if r.get("p_holm"):
            put(key + ".holm", pfmt(float(r["p_holm"])))
        if r.get("category"):
            put(key + ".cat", r["category"])
        put(key + ".na", r["n_a"])
        put(key + ".nb", r["n_b"])
        put(key + ".mean.a", num(float(r["arm_a_mean"]), d(m), sign=False))
        put(key + ".mean.b", num(float(r["arm_b_mean"]), d(m), sign=False))
        put(key + ".sd.a", num(float(r["arm_a_sd"]), d(m), sign=False))
        put(key + ".sd.b", num(float(r["arm_b_sd"]), d(m), sign=False))
    s3 = next(float(r["delta"]) for r in rows if r["id"] == "S3" and r["metric"] == "boundary_f1")
    for r in rows:
        if r["metric"] == "boundary_f1" and r["id"] in ("C1", "C2", "C3", "C4"):
            put(f"FU.{r['id']}.share", f"{100 * float(r['delta']) / s3:.0f}")
    mapping = result.get("mapping")
    if mapping:
        put("FU.case.band", mapping["band_case"])
        put("FU.case.teacher", mapping["teacher_case"])
    for arm, hours in result["training_epoch_hours"].items():
        put(f"FU.hours.{arm}", f"{hours:.2f}")
    for arm, seeds in result["runs_per_arm"].items():
        put(f"FU.n.{arm}", str(len(seeds)))
    t = result["teacher"]
    if "val" in t:
        put("FU.teacher.val.bf1", f"{t['val']['boundary_f1']:.4f}")
        put("FU.teacher.val.dsc", f"{t['val']['dsc']:.4f}")
        put("FU.teacher.epoch", str(t["selected_epoch"]))
        put("FU.teacher.train.bf1", f"{t['train_metrics']['boundary_f1']:.4f}")
        put("FU.teacher.band.tv", f"{t['train_band_tv_distance_mean']:.3f}")
        put("FU.teacher.image.tv", f"{t['train_image_tv_distance_mean']:.4f}")
    put("FU.ext.val.bf1", f"{t['external_val_metrics']['boundary_f1']:.4f}")
    put("FU.ext.train.bf1", f"{t['external_train_metrics']['boundary_f1']:.4f}")
    put("FU.ext.band.tv", f"{t['external_train_band_tv_distance_mean']:.3f}")
    put("FU.ext.image.tv", f"{t['external_train_image_tv_distance_mean']:.4f}")
    if "test" in t:
        put("FU.teacher.test.bf1", f"{t['test']['boundary_f1']:.4f}")
        put("FU.teacher.test.dsc", f"{t['test']['dsc']:.4f}")
    for arm, epochs in result["probe_means"].items():
        for epoch in ("0", "150"):
            if epoch in epochs and "kd_norm_ratio" in epochs[epoch]:
                put(f"FU.probe.{arm}.{epoch}.ratio", f"{epochs[epoch]['kd_norm_ratio']:.2f}")
                put(f"FU.probe.{arm}.{epoch}.loss", f"{epochs[epoch]['loss_kd']:.3f}")
                if "kd_cosine" in epochs[epoch]:
                    put(f"FU.probe.{arm}.{epoch}.cos", f"{epochs[epoch]['kd_cosine']:.2f}")
    # main table: Boundary F1 family + context; DSC and HD95 alongside
    order = ["S3", "C2", "C3", "C1", "C4"] + (["E3"] if any(r["id"] == "E3" for r in rows) else [])
    lines = [
        r"\begin{tabular}{@{}llllll@{}}",
        r"\toprule",
        r"ID & Comparison & Boundary F1 $\Delta$ [95\% CI] & Holm $p$ & Reading & DSC $\Delta$ [95\% CI] \\",  # noqa: E501
        r"\midrule",
    ]
    for cid in order:
        bf1 = next(r for r in rows if r["id"] == cid and r["metric"] == "boundary_f1")
        dsc = next(r for r in rows if r["id"] == cid and r["metric"] == "dsc")
        holm = pfmt(float(bf1["p_holm"])) if bf1.get("p_holm") else "--"
        label = "Stage 3" if cid == "S3" else cid
        lines.append(
            f"{label} & {comparison_tex(bf1)} & {num(float(bf1['delta']))} "
            f"{ci(float(bf1['low']), float(bf1['high']))} & {holm} & {READING_TEX[bf1['reading']]} "
            f"& {num(float(dsc['delta']))} {ci(float(dsc['low']), float(dsc['high']))} \\\\"
        )
        if cid == "S3":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_followup_main.tex").write_text("\n".join(lines) + "\n")
    # supplement: all metrics
    lines = [
        r"\begin{tabular}{@{}lllllll@{}}",
        r"\toprule",
        r"ID & Metric & $\Delta$ [95\% CI] & $p$ & Holm $p$ & Reading & Arm means (SD) \\",
        r"\midrule",
    ]
    for cid in order + [x for x in ("E3a", "E3b") if any(r["id"] == x for r in rows)]:
        for m in ("boundary_f1", "dsc", "iou", "hd95", "assd"):
            r = next(x for x in rows if x["id"] == cid and x["metric"] == m)
            holm = pfmt(float(r["p_holm"])) if r.get("p_holm") else "--"
            means = (
                f"{float(r['arm_a_mean']):.{d(m)}f} ({float(r['arm_a_sd']):.{d(m)}f}) / "
                f"{float(r['arm_b_mean']):.{d(m)}f} ({float(r['arm_b_sd']):.{d(m)}f})"
            )
            first = (
                f"{'Stage 3' if cid == 'S3' else cid}: {comparison_tex(r)}"
                if m == "boundary_f1"
                else ""
            )
            lines.append(
                f"{first} & {MLABEL[m]} & {num(float(r['delta']), d(m))} "
                f"{ci(float(r['low']), float(r['high']), d(m))} & {pfmt(float(r['p']))} & {holm} & "
                f"{r['reading']} & {means} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (GEN / "tab_s_followup_all.tex").write_text("\n".join(lines) + "\n")
    # supplement: per structure
    per_class = read_csv(RES / "per_class.csv")
    lines = [
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Comparison & Structure & DSC & Boundary F1 & HD95 (px) \\",
        r"\midrule",
    ]
    for cid in order:
        for structure in ("RV", "myocardium", "LV"):
            cells = []
            for m in ("dsc", "boundary_f1", "hd95"):
                r = next(
                    x
                    for x in per_class
                    if x["id"] == cid and x["structure"] == structure and x["metric"] == m
                )
                cells.append(
                    f"{num(float(r['delta']), d(m))} {ci(float(r['low']), float(r['high']), d(m))}"
                )
                put(f"FU.{cid}.{structure}.{m}.delta", num(float(r["delta"]), d(m)))
                put(f"FU.{cid}.{structure}.{m}.ci", ci(float(r["low"]), float(r["high"]), d(m)))
            name = (
                (f"{'Stage 3' if cid == 'S3' else cid}: {comparison_tex(r)}")
                if structure == "RV"
                else ""
            )
            lines.append(f"{name} & {structure} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (GEN / "tab_s_followup_perclass.tex").write_text("\n".join(lines) + "\n")
    return dict(result=result, rows=rows, per_seed=read_csv(RES / "per_seed.csv"))


# ---------------------------------------------------------------- post hoc
def stage_shift() -> None:
    path = RES / "posthoc_stage_shift.json"
    if not path.exists():
        return
    rows = [r for r in json.loads(path.read_text())["rows"] if r["metric"] == "boundary_f1"]

    def cell(kind, arm, seeds, patients):
        return next(
            r
            for r in rows
            if r["kind"].startswith(kind)
            and r["arm"] == arm
            and r["seeds"] == seeds
            and r["patients"] == patients
        )

    lines = [
        r"\begin{tabular}{@{}lllrr@{}}",
        r"\toprule",
        r"Arm & Seeds (trainer) & Test patients & Slice-level mean (SD) & Patient-level mean (SD) \\",  # noqa: E501
        r"\midrule",
    ]
    for arm in ("decoder", "kd_band"):
        for seeds, seeds_label in (("early", "1--3"), ("stage3", "201--217")):
            for patients, plabel in (
                ("internal_test", "20 internal"),
                ("official_test", "50 official"),
            ):
                s = cell("cell, slice", arm, seeds, patients)
                u = cell("cell, patient", arm, seeds, patients)
                tag = f"SH.{arm}.{seeds}.{patients}"
                put(tag + ".slice", f"{s['mean']:.4f}")
                put(tag + ".patient", f"{u['mean']:.4f}")
                lines.append(
                    f"{ARM_TEX[arm]} & {seeds_label} & {plabel} & {s['mean']:.4f} ({s['sd']:.4f}) "
                    f"& {u['mean']:.4f} ({u['sd']:.4f}) \\\\"
                )
        if arm == "decoder":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_stage_shift.tex").write_text("\n".join(lines) + "\n")
    for r in rows:
        if r["kind"].startswith("seed set"):
            put(f"SH.seedset.{r['arm']}.{r['patients']}.delta", num(r["delta"]))
            put(f"SH.seedset.{r['arm']}.{r['patients']}.ci", ci(r["low"], r["high"]))
        if r["kind"].startswith("patient set"):
            put(f"SH.patients.{r['arm']}.{r['seeds']}.delta", num(r["delta"]))
            put(f"SH.patients.{r['arm']}.{r['seeds']}.ci", ci(r["low"], r["high"]))


def reference_estimators() -> None:
    path = RES / "posthoc_reference_estimators.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["rows"]
    lines = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Comparison & Metric & $\Delta$ & Prespecified 95\% CI & Mixed model 95\% CI & Hierarchical bootstrap 95\% CI \\",  # noqa: E501
        r"\midrule",
    ]
    names = {
        "S3": "Stage 3: D+KD-band $-$ D",
        "C1": "C1: D+KD-band $-$ D+KD-full",
        "C2": "C2: D+KD-full $-$ D",
        "C3": "C3: D+KD-band$_{\\mathrm{cmp}}$ $-$ D",
        "C4": "C4: D+KD-band $-$ D+KD-band$_{\\mathrm{cmp}}$",
    }
    for r in rows:
        tag = f"REF.{r['id']}.{r['metric']}"
        put(tag + ".mixed", ci(r["mixed_low"], r["mixed_high"]))
        put(tag + ".boot", ci(r["bootstrap_low"], r["bootstrap_high"]))
        lines.append(
            f"{names[r['id']] if r['metric'] == 'boundary_f1' else ''} & {MLABEL[r['metric']]} & "
            f"{num(r['delta'])} & {ci(r['prespecified_low'], r['prespecified_high'])} & "
            f"{ci(r['mixed_low'], r['mixed_high'])} & {ci(r['bootstrap_low'], r['bootstrap_high'])} \\\\"  # noqa: E501
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_reference_estimators.tex").write_text("\n".join(lines) + "\n")


def error_location() -> None:
    path = RES / "posthoc_error_location.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    variant = "all D runs"
    shares = data["structures"][variant]
    corr = [c for c in data["correlations"] if c["variant"] == variant]
    split = [c for c in data["correlations"] if c["variant"] != variant]
    arms = list(dict.fromkeys(c["arm"] for c in corr))
    head = " & ".join(f"{ARM_TEX[a]} gain & $\\rho$" for a in arms)
    lines = [
        r"\begin{tabular}{@{}l" + "r" * (1 + 2 * len(arms)) + r"@{}}",
        r"\toprule",
        r"Structure & D errors within 2\,px & " + head + r" \\",
        r"\midrule",
    ]
    for structure in ("RV", "myocardium", "LV"):
        share = shares[structure]["near_share_pooled"]
        put(f"EL.{structure}.near", f"{100 * share:.0f}")
        cells = []
        for a in arms:
            c = next(x for x in corr if x["arm"] == a and x["structure"] == structure)
            s = next(x for x in split if x["arm"] == a and x["structure"] == structure)
            cells.append(
                f"{num(c['mean_gain'])} & {c['gain_vs_near_share']['rho']:+.2f}".replace("-", "$-$")
            )
            put(
                f"EL.{a}.{structure}.rho",
                rfmt(c["gain_vs_near_share"]["rho"]),
            )
            put(
                f"EL.{a}.{structure}.rho.split",
                rfmt(s["gain_vs_near_share"]["rho"]),
            )
            put(f"EL.{a}.{structure}.p", f"{c['gain_vs_near_share']['p']:.2f}")
        lines.append(f"{structure} & {100 * share:.0f}\\% & " + " & ".join(cells) + r" \\")
    sp = [c["gain_vs_near_share"]["rho"] for c in split if not c["structure"].startswith("all")]
    put("EL.split.min", rfmt(min(sp)))
    put("EL.split.max", rfmt(max(sp)))
    dcorr = [c["gain_vs_d_boundary_f1"]["rho"] for c in corr if "gain_vs_d_boundary_f1" in c]
    put("EL.dcorr.min", rfmt(min(dcorr)))
    put("EL.dcorr.max", rfmt(max(dcorr)))
    for a in arms:
        c = next(x for x in corr if x["arm"] == a and x["structure"].startswith("all"))
        put(f"EL.{a}.all.rho", rfmt(c["gain_vs_near_share"]["rho"]))
        put(f"EL.{a}.all.p", f"{c['gain_vs_near_share']['p']:.2f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_error_location.tex").write_text("\n".join(lines) + "\n")


def thermal_resumes() -> None:
    """Guarded thermal resumes per queue, from the supervisor's event log."""
    path = ROOT / "results/followup/supervision/events.jsonl"
    if not path.exists():
        return
    counts, queue = {}, None
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "followup_supervisor_started":
            queue = "e3" if "e3" in event["queue"] else "e1e2"
        elif event["event"] == "guarded_resume_pending_cooling" and queue:
            counts[queue] = counts.get(queue, 0) + 1
    for q, n in counts.items():
        put(f"FU.thermal.resumes{'' if q == 'e1e2' else '.' + q}", str(n))


def pooling_weights() -> None:
    """Inverse-variance weight share of each dataset in the Stage 1 pooled Boundary F1 rows."""
    rows = read_csv(ROOT / "results/tables/families/expansion_interaction.csv")
    for effect, tag in (
        ("decoder without surface", "dec"),
        ("distillation+gate without surface", "nosurf"),
        ("distillation+gate with surface", "surf"),
    ):
        picked = {
            r["dataset"]: 1 / float(r["se"]) ** 2
            for r in rows
            if r["effect"] == effect
            and r["metric"] == "boundary_f1"
            and r["seeds"] == "1-3"
            and r["dataset"] in ("isic2018", "acdc", "synapse")
        }
        total = sum(picked.values())
        for ds, w in picked.items():
            put(f"PW.{tag}.{ds}", f"{100 * w / total:.0f}")


def other_arms() -> None:
    """Keys for the main-text evidence summary of arms outside the core comparisons."""
    fam = ROOT / "results/tables/families"
    for r in read_csv(fam / "expansion_contrasts.csv"):
        if r["metric"] != "boundary_f1":
            continue
        tag = f"OA.{r['first']}.{r['second']}"
        put(tag + ".delta", num(float(r["all_delta"])))
        put(tag + ".ci", ci(float(r["all_low"]), float(r["all_high"])))
        put(tag + ".fresh", num(float(r["fresh_delta"])))
        put(tag + ".holm", pfmt(float(r["fresh_p_holm"])))
        put(tag + ".status", r["status"])
    for r in read_csv(fam / "expansion_components.csv"):
        if not r["dataset"].startswith("pooled") or r["metric"] not in ("boundary_f1", "hd95"):
            continue
        tag = f"OA.K.{r['component']}.{r['metric']}"
        put(tag + ".est", num(float(r["estimate"]), d(r["metric"])))
        put(tag + ".ci", ci(float(r["low"]), float(r["high"]), d(r["metric"])))
        put(tag + ".class", r["classification"])


# ---------------------------------------------------------------- figure
def figure(data: dict) -> None:
    """Fig. 2 style: run means per arm (left), differences with 95% CIs (right)."""
    rows, per_seed = data["rows"], data["per_seed"]
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.linewidth": 0.5})
    fig = plt.figure(figsize=(7.2, 2.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.7, 2.4], wspace=0.13)
    left = fig.add_subplot(grid[0])
    labels_ax = fig.add_subplot(grid[1])
    right = fig.add_subplot(grid[2])
    arms = [
        ("decoder", "D", BLUE, "o"),
        ("kd_full", "D+KD-full", FULL_COLOR, "s"),
        ("kd_band_peer", r"D+KD-band$_\mathrm{cmp}$", PEER_COLOR, "^"),
        ("kd_band", "D+KD-band", KD_COLOR, "v"),
    ]
    for y, (arm, _, color, marker) in enumerate(arms):
        values = np.array(
            [
                float(r["value"])
                for r in per_seed
                if r["arm"] == arm and r["metric"] == "boundary_f1"
            ]
        )
        jitter = ((np.arange(len(values)) % 3) - 1) * 0.10
        left.scatter(values, y + jitter, s=15, marker=marker, color=color, linewidths=0.5, zorder=3)
        left.plot([values.mean()] * 2, [y - 0.30, y + 0.30], color=color, lw=1.3, zorder=4)
    left.set_yticks(range(len(arms)), [a[1] for a in arms])
    left.set_ylim(-0.6, len(arms) - 0.4)
    left.set_xlabel("Boundary F1", labelpad=4)
    left.set_title("ACDC run means (patients 101–150)", loc="left", fontsize=8.5, pad=10)
    left.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    left.tick_params(length=2, width=0.4, labelsize=8)
    left.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        left.spines[side].set_visible(False)

    def item(cid, label, kind):
        r = next(x for x in rows if x["id"] == cid and x["metric"] == "boundary_f1")
        return (label, float(r["delta"]), float(r["low"]), float(r["high"]), kind)

    groups = [
        (
            "Against D",
            [
                item("S3", "D+KD-band (Stage 3)", "conf"),
                item("C2", "C2  D+KD-full", "fu"),
                item("C3", r"C3  D+KD-band$_\mathrm{cmp}$", "fu"),
            ],
        ),
        (
            "Between KD arms",
            [
                item("C1", "C1  D+KD-band − D+KD-full", "fu"),
                item("C4", r"C4  D+KD-band − D+KD-band$_\mathrm{cmp}$", "fu"),
            ],
        ),
    ]
    styles = {
        "fu": dict(marker="o", color=MUTED, fill="white", size=4.5, width=1.0),
        "conf": dict(marker="D", color=INK, fill=INK, size=4.7, width=1.3),
    }
    right.axvspan(-0.01, 0.01, color=BAND, lw=0, zorder=0)
    right.axvline(0, color=MUTED, lw=0.6, zorder=1)
    y = 0.0
    transform = blended_transform_factory(labels_ax.transAxes, labels_ax.transData)
    for title, items in groups:
        labels_ax.text(
            1.0, y, title, ha="right", va="center", fontsize=8, color=INK, transform=transform
        )
        y += 0.9
        for name, estimate, low, high, kind in items:
            style = styles[kind]
            labels_ax.text(
                1.0, y, name, ha="right", va="center", fontsize=8, color=INK, transform=transform
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
    lows = [float(r["low"]) for r in rows if r["metric"] == "boundary_f1"]
    highs = [float(r["high"]) for r in rows if r["metric"] == "boundary_f1"]
    right.set_xlim(min(-0.02, min(lows) - 0.005), max(0.035, max(highs) + 0.005))
    right.set_xlabel(r"$\Delta$ Boundary F1", labelpad=4)
    right.set_title("Differences (95% CI)", loc="left", fontsize=8.5, pad=10)
    right.tick_params(length=2, width=0.4, labelsize=8)
    for side in ("top", "right", "left"):
        right.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.88, bottom=0.18)
    fig.savefig(FIG / "kd_followup.pdf", metadata={"CreationDate": None})
    fig.savefig(FIG / "kd_followup.png", dpi=300)
    plt.close(fig)


def main() -> None:
    data = followup()
    stage_shift()
    reference_estimators()
    error_location()
    pooling_weights()
    other_arms()
    thermal_resumes()
    if data and all(any(r["id"] == c for r in data["rows"]) for c in ("C1", "C2", "C3", "C4")):
        figure(data)
    lines = ["% generated by analysis/followup/make_followup_assets.py; do not edit"]
    lines += [f"\\expandafter\\def\\csname R@{k}\\endcsname{{{KEYS[k]}}}" for k in sorted(KEYS)]
    (GEN / "numbers_followup.tex").write_text("\n".join(lines) + "\n")
    (GEN / "keys_followup.json").write_text(json.dumps(KEYS, indent=0, sort_keys=True) + "\n")
    print(len(KEYS), "keys")


if __name__ == "__main__":
    main()
