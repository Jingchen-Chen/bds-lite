"""Registered analysis of the follow-up controls (revision/followup_registration.md sections 7-9).

Deterministic; reads only the one-time evaluation outputs (hash-checked against their evaluation
records) and the training records. Every follow-up result is secondary and exploratory.

- Planned family (Boundary F1, Holm over C1-C4):
  C1 D+KD-band - D+KD-full; C2 D+KD-full - D; C3 D+KD-band(compact teacher) - D;
  C4 D+KD-band - D+KD-band(compact teacher). Secondary metrics: the same four, Holm within metric.
- Context: the Stage 3 D+KD-band - D contrast, recomputed with the same function and checked
  against revision/acdc_confirm/results/results.json.
- Outcome categories P/N/E/U and the cases B1-B5, T1-T5 and the narrative of section 9, computed
  here from the numbers so that the reading is not chosen by hand.
- Per structure, per seed, KD gradient probes, training time, the E2 teacher, and E3 (if run).

    PYTHONPATH=src:. .venv/bin/python analysis/followup/analyze.py
Outputs: revision/followup/results/{results.json, contrasts.csv, per_class.csv, per_seed.csv,
probes.csv}.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import expansion_statistics as ex  # noqa: E402

from analysis.kd3.unified_analysis import label_values  # noqa: E402
from scripts.kd.common import sha  # noqa: E402
from scripts.kd_followup.common import (  # noqa: E402
    E3_SEEDS,
    EVAL,
    PEER_CACHE_RUN,
    PEER_TEACHER_RUN,
    RUNS,
    SEEDS,
    WORKSPACE,
    failed_runs,
)
from scripts.kd_followup.common import run_name as followup_run  # noqa: E402

OUT = WORKSPACE / "results"
STAGE3_EVAL = ROOT / "results/acdc_confirm_eval"
STAGE3_RESULTS = ROOT / "revision/acdc_confirm/results/results.json"
SPLIT = "official_test"
METRICS = ex.METRICS
SESOI = ex.SESOI
STRUCTURES = {1: "RV", 2: "myocardium", 3: "LV"}
ARM_LABEL = {
    "decoder": "D",
    "kd_band": "D+KD-band",
    "kd_full": "D+KD-full",
    "kd_band_peer": "D+KD-band (compact teacher)",
    "decoder_aug": "D, augmented",
    "kd_band_aug": "D+KD-band, augmented",
}
FAMILY = [
    ("C1", "kd_band", "kd_full", "Is the band restriction needed?"),
    ("C2", "kd_full", "decoder", "How large is the general KD effect from this teacher?"),
    ("C3", "kd_band_peer", "decoder", "Does a same-capacity, non-pretrained teacher give a gain?"),
    ("C4", "kd_band", "kd_band_peer", "What does the teacher's advantage contribute?"),
]
CONTEXT = [("S3", "kd_band", "decoder", "Stage 3 pre-registered contrast (context, unchanged)")]
E3_ROWS = [
    ("E3", "kd_band_aug", "decoder_aug", "E3 primary: D+KD-band - D under augmentation"),
    ("E3a", "decoder_aug", "decoder", "descriptive: augmentation effect on D (10 vs 17)"),
    ("E3b", "kd_band_aug", "kd_band", "descriptive: augmentation effect on D+KD-band (10 vs 17)"),
]


def stage3_name(arm: str, seed: int) -> str:
    return f"acdc_confirm_{arm}_seed{seed}"


def verified_paths(record_path: Path, base: Path) -> dict[str, Path]:
    """run -> per_case CSV, after checking every artifact hash of the evaluation record."""
    record = json.loads(record_path.read_text())
    out = {}
    for r in record["runs"]:
        if r["split"] != SPLIT:
            continue
        for name, digest in r["artifacts"].items():
            if sha(base / r["run"] / name) != digest:
                raise ValueError(f"evaluation output changed: {r['run']}/{name}")
        out[r["run"]] = base / r["run"] / f"per_case_{SPLIT}.csv"
    return out


def arm_runs() -> dict[str, dict[int, Path]]:
    stage3 = verified_paths(STAGE3_EVAL / "evaluation_record.json", STAGE3_EVAL)
    arms = {arm: {s: stage3[stage3_name(arm, s)] for s in SEEDS} for arm in ("decoder", "kd_band")}
    failed = failed_runs()
    for queue, names in (
        ("e1e2", {"kd_full": SEEDS, "kd_band_peer": SEEDS}),
        ("e3", {"decoder_aug": E3_SEEDS, "kd_band_aug": E3_SEEDS}),
    ):
        record = EVAL / f"evaluation_record_{queue}.json"
        if not record.exists():
            continue
        paths = verified_paths(record, EVAL)
        for arm, seeds in names.items():
            arms[arm] = {
                s: paths[followup_run(arm, s)] for s in seeds if followup_run(arm, s) not in failed
            }
    return arms


def category(c: dict, metric: str) -> str:
    """Registration 9.1 on the Holm-adjusted p (primary family) or p_holm within metric."""
    if c["p_holm"] < 0.05:
        return "P" if c["delta"] > 0 else "N"
    if -SESOI[metric] < c["low"] and c["high"] < SESOI[metric]:
        return "E"
    return "U"


def band_case(c1: str, c2: str) -> str:
    if c1 == "P":
        return "B1"
    if c1 == "E":
        return "B2"
    if c1 == "N":
        return "B4"
    return "B3" if c2 == "P" else "B5"


def teacher_case(c3: str, c4: str) -> str:
    if c3 == "N":
        return "T4"
    if c3 == "P":
        return "T3" if c4 == "P" else ("T2" if c4 in ("U", "E") else "T5")
    return "T1" if c4 == "P" else "T5"


NARRATIVE = {
    ("T13", "B1"): "teacher advantage associated with the gain; band restriction added to it",
    ("T13", "B234"): "gain associated with KD from a stronger teacher, not with boundary focus",
    ("T13", "B5"): "teacher advantage associated with the gain; band specificity unresolved",
    ("T2", "B1"): "boundary-band KD helped without a teacher advantage; band added to full KD",
    ("T2", "B234"): "general soft-label KD effect, neither boundary nor teacher specific",
    ("T2", "B5"): "KD from a peer teacher helped; band specificity unresolved",
    ("T45", "any"): "teacher question unresolved; Stage 3 conclusion kept, attribution open",
}


def narrative(b: str, t: str) -> str:
    if t in ("T4", "T5"):
        return NARRATIVE[("T45", "any")]
    group = "T13" if t in ("T1", "T3") else "T2"
    bgroup = "B1" if b == "B1" else ("B5" if b == "B5" else "B234")
    return NARRATIVE[(group, bgroup)]


def row(cid: str, first: str, second: str, question: str, metric: str, c: dict, **extra):
    return dict(
        id=cid,
        comparison=f"{ARM_LABEL[first]} - {ARM_LABEL[second]}",
        question=question,
        metric=metric,
        delta=c["delta"],
        low=c["low"],
        high=c["high"],
        p=c["p"],
        p_holm=c.get("p_holm"),
        se_run=c["se_run"],
        se_unit=c["se_unit"],
        df=c["df"],
        n_a=c["n_a"],
        n_b=c["n_b"],
        arm_a_mean=c["arm_a_mean"],
        arm_b_mean=c["arm_b_mean"],
        arm_a_sd=c["arm_a_sd"],
        arm_b_sd=c["arm_b_sd"],
        reading=ex.classify(c["low"], c["high"], SESOI[metric]),
        **extra,
    )


def probe_summary(arm: str, seeds) -> list[dict]:
    rows = []
    for seed in seeds:
        name = stage3_name(arm, seed) if arm in ("decoder", "kd_band") else followup_run(arm, seed)
        path = RUNS / name / "kd_probes.json"
        if not path.exists():
            continue
        for p in json.loads(path.read_text()):
            rows.append(
                dict(
                    arm=arm,
                    seed=seed,
                    epoch=p["epoch"],
                    loss_kd=p["loss_kd"],
                    kd_norm_ratio=p["kd_norm_ratio"],
                    kd_cosine=p["kd_cosine"],
                    boundary_norm_ratio=p["boundary_norm_ratio"],
                )
            )
    return rows


def epoch_hours(arm: str, seeds) -> float:
    total = 0.0
    for seed in seeds:
        name = stage3_name(arm, seed) if arm in ("decoder", "kd_band") else followup_run(arm, seed)
        total += sum(h["seconds"] for h in json.loads((RUNS / name / "history.json").read_text()))
    return total / 3600


def teacher_block() -> dict:
    out = {}
    cache = RUNS / PEER_CACHE_RUN / "cache_complete.json"
    if cache.exists():
        record = json.loads(cache.read_text())
        final = json.loads((RUNS / PEER_TEACHER_RUN / "final_metrics.json").read_text())
        out["val"] = {m: v["mean"] for m, v in final["metrics"]["val"].items()}
        out["selected_epoch"] = final["selected_epoch"]
        out["cache_sha256"] = record["cache_sha256"]
        out["cache_check_max_abs_difference"] = record["check"]["max_abs_difference"]
        diagnostics = json.loads((RUNS / PEER_CACHE_RUN / "diagnostics.json").read_text())
        out["train_band_tv_distance_mean"] = diagnostics["train"]["band_tv_distance_mean"]
        out["train_image_tv_distance_mean"] = diagnostics["train"]["image_tv_distance_mean"]
        out["train_metrics"] = diagnostics["train"]["metrics"]
    external = json.loads((RUNS / "acdc_kd_teacher_cache/diagnostics.json").read_text())
    out["external_train_band_tv_distance_mean"] = external["train"]["band_tv_distance_mean"]
    out["external_train_image_tv_distance_mean"] = external["train"]["image_tv_distance_mean"]
    out["external_train_metrics"] = external["train"]["metrics"]
    out["external_val_metrics"] = external["val"]["metrics"]
    record = EVAL / "evaluation_record_e1e2.json"
    if record.exists():
        path = verified_paths(record, EVAL)[PEER_TEACHER_RUN]
        values = ex.unit_values(path, "acdc")
        out["test"] = {m: float(np.mean(list(values[m].values()))) for m in METRICS}
    return out


def main() -> None:
    arms = arm_runs()
    values = {
        arm: {s: ex.unit_values(p, "acdc") for s, p in runs.items()} for arm, runs in arms.items()
    }
    lists = {arm: list(v.values()) for arm, v in values.items()}
    contrasts, per_class, per_seed = [], [], []
    result = dict(
        status="secondary, exploratory (registration section 1)",
        split=SPLIT,
        runs_per_arm={arm: sorted(v) for arm, v in arms.items()},
        failed_runs=list(failed_runs().values()),
    )
    # Stage 3 context, recomputed and checked
    s3 = ex.contrast(lists["kd_band"], lists["decoder"], "boundary_f1")
    frozen = json.loads(STAGE3_RESULTS.read_text())["primary"]["contrast"]
    if abs(s3["delta"] - frozen["delta"]) > 1e-12 or abs(s3["low"] - frozen["low"]) > 1e-12:
        raise ValueError("recomputed Stage 3 contrast differs from the committed result")
    family_ready = all(a in lists for a in ("kd_full", "kd_band_peer"))
    cats = {}
    for metric in METRICS:
        block = []
        for cid, first, second, question in CONTEXT:
            c = ex.contrast(lists[first], lists[second], metric)
            contrasts.append(row(cid, first, second, question, metric, c, family=False))
        if family_ready:
            for cid, first, second, question in FAMILY:
                c = ex.contrast(lists[first], lists[second], metric)
                block.append((cid, first, second, question, c))
            ex.holm([b[4] for b in block])
            for cid, first, second, question, c in block:
                cat = category(c, metric)
                if metric == "boundary_f1":
                    cats[cid] = cat
                contrasts.append(
                    row(
                        cid,
                        first,
                        second,
                        question,
                        metric,
                        c,
                        family=metric == "boundary_f1",
                        category=cat,
                    )
                )
        if all(a in lists for a in ("decoder_aug", "kd_band_aug")):
            for cid, first, second, question in E3_ROWS:
                c = ex.contrast(lists[first], lists[second], metric)
                contrasts.append(row(cid, first, second, question, metric, c, family=False))
    if family_ready:
        b, t = band_case(cats["C1"], cats["C2"]), teacher_case(cats["C3"], cats["C4"])
        result["mapping"] = dict(categories=cats, band_case=b, teacher_case=t)
        result["mapping"]["narrative"] = narrative(b, t)
    # per structure: every planned and context comparison, DSC / Boundary F1 / HD95
    comparisons = CONTEXT + (FAMILY if family_ready else [])
    if all(a in lists for a in ("decoder_aug", "kd_band_aug")):
        comparisons = comparisons + E3_ROWS[:1]
    for label, structure in STRUCTURES.items():
        lv = {
            arm: [label_values(p, "acdc", label) for p in runs.values()]
            for arm, runs in arms.items()
        }
        for cid, first, second, question in comparisons:
            for metric in ["dsc", "boundary_f1", "hd95"]:
                c = ex.contrast(lv[first], lv[second], metric)
                per_class.append(row(cid, first, second, question, metric, c, structure=structure))
    for arm, runs in values.items():
        for seed, run in runs.items():
            for metric in METRICS:
                v = [x for x in run[metric].values() if math.isfinite(x)]
                per_seed.append(dict(arm=arm, seed=seed, metric=metric, value=float(np.mean(v))))
    probes = [r for arm, runs in arms.items() for r in probe_summary(arm, sorted(runs))]
    result["probe_means"] = {
        arm: {
            str(epoch): {
                k: float(np.mean([r[k] for r in probes if r["arm"] == arm and r["epoch"] == epoch]))
                for k in ("loss_kd", "kd_norm_ratio", "kd_cosine")
                if all(r[k] is not None for r in probes if r["arm"] == arm and r["epoch"] == epoch)
            }
            for epoch in sorted({r["epoch"] for r in probes if r["arm"] == arm})
        }
        for arm in arms
    }
    result["training_epoch_hours"] = {arm: epoch_hours(arm, sorted(r)) for arm, r in arms.items()}
    result["teacher"] = teacher_block()
    result["contrasts"] = contrasts
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    for name, rows in [
        ("contrasts", contrasts),
        ("per_class", per_class),
        ("per_seed", per_seed),
        ("probes", probes),
    ]:
        if not rows:
            continue
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (OUT / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result.get("mapping", {}), indent=2))
    for r in contrasts:
        if r["metric"] == "boundary_f1":
            print(
                f"{r['id']:4s} {r['comparison']:52s} {r['delta']:+.4f} "
                f"[{r['low']:+.4f}, {r['high']:+.4f}] p={r['p']:.3g} "
                f"holm={r['p_holm'] if r['p_holm'] is None else round(r['p_holm'], 4)} "
                f"{r.get('category', '')} {r['reading']}"
            )


if __name__ == "__main__":
    main()
