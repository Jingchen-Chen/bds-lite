"""Inference-only evaluation of the two frozen ACDC teachers on official patients 101-150.

results/teacher_eval/PROTOCOL.md. Refuses to run unless sanity_checks.json passed. For each
teacher: verifies the checkpoint sha256 against the registered value, the run's own
final_metrics.json and the teacher-cache record; verifies the cache file the students read and
the hash every relevant student run recorded; builds the model exactly as in its training and
earlier evaluation (external: scripts.kd_mc.teacher.build_teacher, one channel replicated to
three, ACDC input passed unchanged; compact: scripts.acdc_confirm.evaluate_test.load, the
student loader); counts parameters; then runs the unchanged scripts/revision/train_run.py::
evaluate on the hash-checked official loader, writing only under results/teacher_eval/. The new
predictions and per-case rows are compared with the earlier one-time evaluations (read only).

    PYTHONPATH=src:. .venv/bin/python analysis/teacher_eval/evaluate_teachers.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO, REPO / "src", REPO / "analysis/revision", HERE):
    sys.path.insert(0, str(_path))
from common import OUT, ROOT, RUNS, SPLIT, TEACHERS, teacher_dir  # noqa: E402

from scripts.acdc_confirm.evaluate_test import load, official_loader  # noqa: E402
from scripts.kd.common import sha, write_json  # noqa: E402
from scripts.kd_mc.teacher import build_teacher  # noqa: E402
from scripts.revision.train_run import evaluate  # noqa: E402

STUDENT_PREFIXES = {
    "external": ("acdc_confirm_kd_band", "acdc_followup_kd_full", "acdc_followup_kd_band_aug"),
    "compact": ("acdc_followup_kd_band_peer",),
}


def verify(teacher: str) -> dict:
    spec = TEACHERS[teacher]
    folder = RUNS / spec["run"]
    cache = RUNS / spec["cache_run"]
    final = json.loads((folder / "final_metrics.json").read_text())
    complete = json.loads((cache / "cache_complete.json").read_text())
    actual = sha(folder / "best.pt")
    cache_actual = sha(cache / "train_teacher_prob_fp16.npy")
    student_hashes = {}
    for prefix in STUDENT_PREFIXES[teacher]:
        for provenance in sorted(RUNS.glob(f"{prefix}_seed*/provenance.json")):
            record = json.loads(provenance.read_text())["data_hashes"]["teacher_cache"]
            student_hashes[provenance.parent.name] = record["train_teacher_prob_fp16.npy"]
    checks = dict(
        best_sha256=actual,
        equals_registered=actual == spec["best_sha256"],
        equals_final_metrics=final["artifacts"]["best.pt"] == actual,
        equals_cache_record=complete["teacher_best_sha256"] == actual,
        cache_sha256=cache_actual,
        cache_equals_record=complete["cache_sha256"] == cache_actual == spec["cache_sha256"],
        students_reading_cache=len(student_hashes),
        all_students_recorded_this_cache=set(student_hashes.values()) == {cache_actual},
        selected_epoch=final["selected_epoch"],
        epochs=final["epochs"],
        test_evaluated_during_training=final["test_evaluated"],
    )
    prefixes = ("equals", "cache_equals", "all_")
    failed = [k for k in checks if k.startswith(prefixes) and not checks[k]]
    if failed or not student_hashes:
        raise ValueError(f"{teacher} teacher identity check failed: {failed}")
    return checks


def build(teacher: str) -> tuple[torch.nn.Module, dict]:
    folder = RUNS / TEACHERS[teacher]["run"]
    config = json.loads((folder / "config.json").read_text())
    if config["dataset"]["name"] != "acdc" or config["project"]["seed"] != 100:
        raise ValueError("not the ACDC seed-100 teacher")
    if teacher == "external":
        model_cfg = config["model"]
        expected = dict(
            name="smp_unet_resnet34",
            in_channels=1,
            num_classes=4,
            input_adapter="replicate_to_3_channels",
            input_norm="none",
        )
        if model_cfg != expected or config["kd"]["role"] != "teacher":
            raise ValueError(f"external teacher config differs: {model_cfg}")
        model = build_teacher(config, pretrained=False)
        payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model = model.cuda().eval()
        input_processing = (
            "ManifestDataset 1x224x224 per-volume z-scored image; TeacherNetMC.forward expands to "
            "3 identical channels (in_channels=3 pretrained ResNet-34 stem); input_norm 'none' "
            "(no ImageNet mean/std) -- identical to scripts/kd_mc/train_mc.py training and cache"
        )
    else:
        if config["model"]["name"] != "bdslite_unet" or config["kd"].get("lambda_kd") != 0.0:
            raise ValueError("compact teacher is not the D-recipe student architecture")
        model = load(folder, config)
        input_processing = (
            "ManifestDataset 1x224x224 per-volume z-scored image passed unchanged to the student "
            "architecture (in_channels=1), identical to the D-recipe training"
        )
    parameters = sum(p.numel() for p in model.parameters())
    return model, dict(parameters=parameters, input_processing=input_processing, config=config)


def compare_prior(teacher: str, out: Path) -> dict:
    prior = TEACHERS[teacher]["prior_eval"]
    new = np.load(out / f"preds_{SPLIT}.npz")
    old = np.load(prior / f"preds_{SPLIT}.npz")
    same_ids = new["ids"].tolist() == old["ids"].tolist()
    differing = int((new["predictions"] != old["predictions"]).sum()) if same_ids else None
    return dict(
        prior_evaluation=str(prior.relative_to(ROOT)),
        same_ids=same_ids,
        differing_pixels=differing,
        predictions_identical=same_ids and differing == 0,
        per_case_csv_identical=(out / f"per_case_{SPLIT}.csv").read_bytes()
        == (prior / f"per_case_{SPLIT}.csv").read_bytes(),
    )


def main() -> None:
    sanity = json.loads((OUT / "sanity_checks.json").read_text())
    if not sanity["passed"]:
        raise RuntimeError(f"sanity checks failed; teachers not evaluated: {sanity['problems']}")
    for teacher in TEACHERS:
        if teacher_dir(teacher).exists():
            raise FileExistsError(f"{teacher_dir(teacher)} exists; refusing to overwrite")
    record = dict(
        label="post hoc, descriptive, inference only (author request 2026-11-11)",
        started_utc=datetime.now(timezone.utc).isoformat(),
        split="official_test (ACDC patients 101-150)",
        loader="scripts/acdc_confirm/evaluate_test.py::official_loader (hash-checked)",
        evaluator="scripts/revision/train_run.py::evaluate (unchanged; argmax only)",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        teachers={},
    )
    data = official_loader()
    for teacher in TEACHERS:
        identity = verify(teacher)
        model, info = build(teacher)
        if teacher == "external" and info["parameters"] != 24_436_804:
            raise ValueError(f"external teacher has {info['parameters']} parameters")
        out = teacher_dir(teacher)
        out.mkdir(parents=True)
        with torch.no_grad():
            metrics = evaluate(model, data, 4, out, SPLIT)
        record["teachers"][teacher] = dict(
            run=TEACHERS[teacher]["run"],
            identity=identity,
            parameters=info["parameters"],
            input_processing=info["input_processing"],
            row_level_metrics_from_evaluate=metrics,
            artifacts={f: sha(out / f) for f in [f"preds_{SPLIT}.npz", f"per_case_{SPLIT}.csv"]},
            reproduction_of_prior_evaluation=compare_prior(teacher, out),
        )
        print(teacher, info["parameters"], json.dumps(metrics), flush=True)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(OUT / "evaluation_record.json", record)


if __name__ == "__main__":
    main()
