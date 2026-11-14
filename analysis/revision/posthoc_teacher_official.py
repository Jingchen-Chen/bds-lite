"""Post hoc, descriptive: the external ACDC teacher on the official held-out patients 101-150.

Authorized by the author on 2026-11-09 (restructure revision, decision 7). Read-only inference
with one frozen checkpoint; no training, no change to stage-3 files, estimators or decisions.

- checkpoint: results/revision_runs/acdc_kd_teacher_seed100/best.pt, sha256 must equal the value
  recorded in that run's test_metrics.json and in acdc_kd_teacher_cache/cache_complete.json;
- model: scripts.kd_mc.teacher.build_teacher(config, pretrained=False), strict state-dict load
  (the loader of analysis/kd3/evaluate_test.py);
- data: scripts.acdc_confirm.evaluate_test.official_loader() (manifest and every NPZ hash-checked;
  batch 16, 4 workers, no shuffle, as in stage 3);
- evaluation: the unchanged scripts/revision/train_run.py::evaluate;
- output: results/acdc_confirm_eval_posthoc/acdc_kd_teacher_seed100/ (refuses to overwrite).

Run with the KD environment (segmentation_models_pytorch 0.6.0):
    PYTHONPATH=src:. /home/jc/bds-followup/.venv-kd/bin/python \
        analysis/revision/posthoc_teacher_official.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.acdc_confirm.evaluate_test import official_loader  # noqa: E402
from scripts.kd.common import sha, write_json  # noqa: E402
from scripts.kd_mc.teacher import build_teacher  # noqa: E402
from scripts.revision.train_run import evaluate  # noqa: E402

RUN = ROOT / "results/revision_runs/acdc_kd_teacher_seed100"
OUT = ROOT / "results/acdc_confirm_eval_posthoc/acdc_kd_teacher_seed100"
EXPECTED_SHA256 = "735747303c444e13dd57457e3540ee410b5de43421b68fd92e2eda69c88dafa0"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists; this post hoc evaluation is not repeated")
    recorded = {
        "test_metrics.json": json.loads((RUN / "test_metrics.json").read_text())["best_sha256"],
        "cache_complete.json": json.loads(
            (RUN.parent / "acdc_kd_teacher_cache/cache_complete.json").read_text()
        )["teacher_best_sha256"],
    }
    actual = sha(RUN / "best.pt")
    if actual != EXPECTED_SHA256 or set(recorded.values()) != {EXPECTED_SHA256}:
        raise ValueError(f"checkpoint hash mismatch: {actual} vs {recorded}")
    config = json.loads((RUN / "config.json").read_text())
    if config["kd"]["role"] != "teacher" or config["dataset"]["name"] != "acdc":
        raise ValueError("not the ACDC teacher")
    model = build_teacher(config, pretrained=False)
    payload = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model = model.cuda().eval()
    OUT.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    with torch.no_grad():
        metrics = evaluate(model, official_loader(), 4, OUT, "official_test")
    write_json(
        OUT / "evaluation_record.json",
        dict(
            label="post hoc, descriptive (author authorization 2026-11-09, decision 7)",
            started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            checkpoint=str((RUN / "best.pt").relative_to(ROOT)),
            checkpoint_sha256=actual,
            recorded_sha256=recorded,
            selected_epoch=payload.get("epoch"),
            split="official_test (patients 101-150)",
            evaluator="scripts/revision/train_run.py::evaluate (unchanged)",
            loader="scripts/acdc_confirm/evaluate_test.py::official_loader (hash-checked)",
            torch=torch.__version__,
            cuda=torch.version.cuda,
            metrics=metrics,
            artifacts={
                f: sha(OUT / f) for f in ["preds_official_test.npz", "per_case_official_test.csv"]
            },
        ),
    )
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
