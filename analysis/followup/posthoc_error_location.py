"""POST HOC: where D's errors lie versus how much each KD arm gains (registration section 10.1).

Hypothesis-generating only; labelled post hoc wherever reported. Patients 101-150.
- Near-boundary error share of D: per structure c and patient u, the wrong pixels of class c
  (false positives and false negatives) within 2 px of the reference inner contour of class c,
  over all wrong pixels of class c; the definition of analysis/revision/error_decomposition.py
  (inner = gt XOR erode(gt); near = EDT(~inner) <= 2; no contour -> all errors far). Counts are
  pooled over slices and over the D runs used.
- Gain: per structure and patient, the mean Boundary F1 (and DSC) of the KD arm's runs minus that
  of D's runs (analysis/kd3/unified_analysis.py::label_values).
- Association: Spearman correlation across patients, per structure and over all patient-structure
  pairs; also the correlation of the gain with D's own Boundary F1 (lower scores leave more room).
- Shared-noise control: D's runs are split by seed (201-208 for the error share, 209-217 for the
  gain), so that chance errors of the same D runs do not enter both quantities.

    PYTHONPATH=src:. .venv/bin/python analysis/followup/posthoc_error_location.py
Outputs: results/followup/results/posthoc_error_location.{json,csv}
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT / "analysis/followup"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import family_statistics as fs  # noqa: E402
from analyze import STRUCTURES, arm_runs  # noqa: E402

from analysis.kd3.unified_analysis import label_values  # noqa: E402

OFFICIAL = ROOT / "data/processed/acdc_official_test/test"
OUT = ROOT / "results/followup/results"
SPLIT_A, SPLIT_B = tuple(range(201, 209)), tuple(range(209, 218))
KD_ARMS = ("kd_band", "kd_full", "kd_band_peer")


def error_counts(pred_path: Path, targets: dict) -> dict:
    """(patient, label) -> [near, far] wrong-pixel counts for one run."""
    counts = defaultdict(lambda: [0, 0])
    with np.load(pred_path) as z:
        preds = dict(zip(z["ids"], z["predictions"], strict=True))
    for sid, (gt_all, nears) in targets.items():
        patient = fs.cluster_of("acdc", sid)
        pred_all = preds[sid]
        for label in STRUCTURES:
            gt, pred = gt_all == label, pred_all == label
            wrong = gt ^ pred
            near = nears[label]
            counts[(patient, label)][0] += int((wrong & near).sum())
            counts[(patient, label)][1] += int((wrong & ~near).sum())
    return counts


def load_targets() -> dict:
    targets = {}
    for path in sorted(OFFICIAL.glob("*.npz")):
        with np.load(path) as z:
            mask = z["mask"].astype(np.int64)
        nears = {}
        for label in STRUCTURES:
            gt = mask == label
            inner = gt ^ ndi.binary_erosion(gt)
            nears[label] = (
                ndi.distance_transform_edt(~inner) <= 2 if inner.any() else np.zeros_like(gt)
            )
        targets[path.stem] = (mask, nears)
    return targets


def share(counts_list: list[dict]) -> dict:
    pooled = defaultdict(lambda: [0, 0])
    for counts in counts_list:
        for key, (near, far) in counts.items():
            pooled[key][0] += near
            pooled[key][1] += far
    return pooled


def spearman(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    rho, p = stats.spearmanr(x[keep], y[keep])
    return dict(rho=float(rho), p=float(p), n=int(keep.sum()))


def main() -> None:
    arms = arm_runs()
    targets = load_targets()
    eval_dirs = {
        arm: {s: p.parent for s, p in runs.items()}
        for arm, runs in arms.items()
        if arm != "kd_full"
    }
    counts = {
        s: error_counts(d / "preds_official_test.npz", targets)
        for s, d in eval_dirs["decoder"].items()
    }
    per_label = {
        arm: {
            label: [label_values(p, "acdc", label) for p in runs.values()] for label in STRUCTURES
        }
        for arm, runs in arms.items()
    }
    seeds_d = sorted(arms["decoder"])

    def mean_metric(arm: str, label: int, metric: str, seeds=None) -> dict:
        runs = arms[arm]
        chosen = [i for i, s in enumerate(sorted(runs)) if seeds is None or s in seeds]
        rows = [per_label[arm][label][i][metric] for i in chosen]
        units = set.intersection(*[set(r) for r in rows])
        return {u: float(np.mean([r[u] for r in rows])) for u in units}

    rows, summary = [], dict(label="post hoc", structures={}, correlations=[])
    for variant, err_seeds, gain_seeds in (
        ("all D runs", seeds_d, None),
        ("split: error share from D seeds 201-208, gain vs D seeds 209-217", SPLIT_A, SPLIT_B),
    ):
        pooled = share([counts[s] for s in err_seeds])
        for arm in [a for a in KD_ARMS if a in arms]:
            xs_all, ys_all = [], []
            for label, structure in STRUCTURES.items():
                d_bf1 = mean_metric("decoder", label, "boundary_f1", gain_seeds)
                k_bf1 = mean_metric(arm, label, "boundary_f1")
                patients = sorted(set(d_bf1) & set(k_bf1))
                near = {
                    u: pooled[(u, label)][0] / max(1, sum(pooled[(u, label)])) for u in patients
                }
                gain = {u: k_bf1[u] - d_bf1[u] for u in patients}
                for u in patients:
                    rows.append(
                        dict(
                            variant=variant,
                            arm=arm,
                            structure=structure,
                            patient=u,
                            near_share=near[u],
                            wrong_pixels=sum(pooled[(u, label)]),
                            d_boundary_f1=d_bf1[u],
                            gain_boundary_f1=gain[u],
                        )
                    )
                xs = [near[u] for u in patients]
                ys = [gain[u] for u in patients]
                xs_all += xs
                ys_all += ys
                total_near = sum(pooled[(u, label)][0] for u in patients)
                total = sum(sum(pooled[(u, label)]) for u in patients)
                summary["structures"].setdefault(variant, {})[structure] = dict(
                    near_share_pooled=total_near / total,
                    near_share_patient_median=float(np.median(xs)),
                )
                summary["correlations"].append(
                    dict(
                        variant=variant,
                        arm=arm,
                        structure=structure,
                        gain_vs_near_share=spearman(xs, ys),
                        gain_vs_d_boundary_f1=spearman([d_bf1[u] for u in patients], ys),
                        mean_gain=float(np.mean(ys)),
                    )
                )
            summary["correlations"].append(
                dict(
                    variant=variant,
                    arm=arm,
                    structure="all patient-structure pairs",
                    gain_vs_near_share=spearman(xs_all, ys_all),
                )
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "posthoc_error_location.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "posthoc_error_location.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["structures"], indent=1))
    for c in summary["correlations"]:
        g = c["gain_vs_near_share"]
        print(
            c["variant"][:9],
            c["arm"],
            c["structure"],
            f"rho={g['rho']:+.3f} p={g['p']:.3g} n={g['n']}",
            f"mean_gain={c.get('mean_gain', float('nan')):+.4f}",
        )


if __name__ == "__main__":
    main()
