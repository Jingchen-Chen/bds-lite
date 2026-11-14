"""Shared constants for the post hoc frozen-teacher evaluation (results/teacher_eval/PROTOCOL.md).

Inference only on official ACDC patients 101-150. Reuses the unchanged student evaluator,
loader and aggregation; writes only under results/teacher_eval/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "src", ROOT / "analysis/revision"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

OUT = ROOT / "results/teacher_eval"
RUNS = ROOT / "results/revision_runs"
SPLIT = "official_test"
CONFIRM_EVAL = ROOT / "results/acdc_confirm_eval"
FOLLOWUP_EVAL = ROOT / "results/followup_eval"
STRUCTURES = {1: "RV", 2: "myocardium", 3: "LV"}
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10_000, 20261111
SANITY_TOLERANCE = 1e-4
EXPECTED_EXTERNAL_PARAMETERS = 24_436_804

# group -> (evaluation folder, run prefix, seeds, evaluation record)
GROUPS = {
    "D": (CONFIRM_EVAL, "acdc_confirm_decoder", range(201, 218), "evaluation_record.json"),
    "D+KD-band": (CONFIRM_EVAL, "acdc_confirm_kd_band", range(201, 218), "evaluation_record.json"),
    "D+KD-full": (
        FOLLOWUP_EVAL,
        "acdc_followup_kd_full",
        range(201, 218),
        "evaluation_record_e1e2.json",
    ),
    "D+KD-band_c": (
        FOLLOWUP_EVAL,
        "acdc_followup_kd_band_peer",
        range(201, 218),
        "evaluation_record_e1e2.json",
    ),
    "D_aug": (
        FOLLOWUP_EVAL,
        "acdc_followup_decoder_aug",
        range(201, 211),
        "evaluation_record_e3.json",
    ),
    "D+KD-band_aug": (
        FOLLOWUP_EVAL,
        "acdc_followup_kd_band_aug",
        range(201, 211),
        "evaluation_record_e3.json",
    ),
}
EXPECTED_RUNS = {g: len(v[2]) for g, v in GROUPS.items()}

# Mandatory sanity checks: paper values (4 decimals, tolerance 1e-4) and the frozen full-precision
# arm means (revision/acdc_confirm/results/results.json; revision/followup/results/contrasts.csv;
# revision/evidence_freeze/final_e3_analysis.csv), the latter checked at 1e-10.
PAPER_BF1 = {"D": 0.7797, "D+KD-band": 0.7975, "D_aug": 0.8469, "D+KD-band_aug": 0.8432}
FROZEN_BF1 = {
    "D": 0.7796910271473292,
    "D+KD-band": 0.7975114099217608,
    "D+KD-full": 0.7902551039018086,
    "D+KD-band_c": 0.7915202362691431,
    "D_aug": 0.8468818139546442,
    "D+KD-band_aug": 0.8431735089530801,
}

TEACHERS = {
    "external": dict(
        run="acdc_kd_teacher_seed100",
        best_sha256="735747303c444e13dd57457e3540ee410b5de43421b68fd92e2eda69c88dafa0",
        cache_run="acdc_kd_teacher_cache",
        cache_sha256="42266ea089e495d651cf654ae7c5c287a2b40d77cc1131ea78f3a0d059a6077e",
        students=("D+KD-band", "D+KD-full", "D+KD-band_aug"),
        prior_eval=ROOT / "results/acdc_confirm_eval_posthoc/acdc_kd_teacher_seed100",
    ),
    "compact": dict(
        run="acdc_followup_e2_teacher_seed100",
        best_sha256="0bf28bcc7bce3ca75cc8fb3fbba277120c92702b78edb7979c147893df1f7c1e",
        cache_run="acdc_followup_e2_teacher_cache",
        cache_sha256="fdbcc62425757b95343b0c5034e747bc6957fa8b056ccd4d14681d84348bbed7",
        students=("D+KD-band_c",),
        prior_eval=FOLLOWUP_EVAL / "acdc_followup_e2_teacher_seed100",
    ),
}


def run_names(group: str) -> list[str]:
    _, prefix, seeds, _ = GROUPS[group]
    return [f"{prefix}_seed{s}" for s in seeds]


def case_file(group: str, run: str) -> Path:
    return GROUPS[group][0] / run / f"per_case_{SPLIT}.csv"


def pred_file(group: str, run: str) -> Path:
    return GROUPS[group][0] / run / f"preds_{SPLIT}.npz"


def teacher_dir(teacher: str) -> Path:
    return OUT / f"{teacher}_teacher"
