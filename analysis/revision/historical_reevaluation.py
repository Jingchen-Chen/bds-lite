"""Frozen inference for protocols/historical_reevaluation/HISTORICAL_REEVALUATION_PROTOCOL.md. CPU only; never trains.

Stages (run in this order by default):
  october    parity on 64 validation images, then official ISIC test, arms G, C, D.
  april      ACDC test, clean ISIC 390, Synapse test, parity against the legacy per-case files,
             then official ISIC test only for models that passed.
  tolerance  Boundary F1 at 1, 2, 4 px from frozen ten-run predictions. Experiment C excluded.
  validate   completeness, hashes, independent DSC recomputation.

--smoke loads every model strictly and runs one small development batch. It reads no test data.
"""

from __future__ import annotations

import os

# A CUDA context would register as a foreign GPU process and stop the training supervisor.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from analysis.revision.recover_predictions import case_metrics
from bds_lite.data.boundary import mask_to_boundary
from bds_lite.evaluation.metrics import boundary_f1
from bds_lite.models import BDSLiteUNet, UNet
from bds_lite.training import build_model
from bds_lite.training.runner import inference_forward
from scripts.revision.train_run import OfficialTestDataset
from scripts.revision.training_support import ManifestDataset

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/historical_reevaluation"
PROTOCOL = ROOT / "protocols/historical_reevaluation/HISTORICAL_REEVALUATION_PROTOCOL.md"
CLASSES = {"isic2018": 2, "isic390": 2, "acdc": 4, "synapse": 9}
CHANNELS = {"isic2018": 3, "isic390": 3, "acdc": 1, "synapse": 1}
SEEDS = (1, 2, 3)
APRIL = {
    "a_decoder": ("bds_lite_boundary_only", False),
    "a_distill": ("bds_lite_boundary_distill", False),
    "a_gate": ("bds_lite_boundary_gate", True),
    "a_gate_surface": ("bds_lite_gate_surface", True),
}
REFERENCE = {"a_unet": "unet", "a_full": "bds_lite_full"}
OCTOBER = {
    "o_G": "pathway_controls_20261018/isic2018_G",
    "o_C": "pathway_controls_20261018/isic2018_C",
    "o_D": "boundary_specificity_20261018/isic2018_D",
}
PARITY_MEAN, PARITY_ROW, PARITY_SHARE = 0.002, 0.01, 0.02
TEMP_PAUSE, TEMP_RESUME = 82, 72
TOLERANCES = (1, 2, 4)
TEN_RUN = {
    "unet": "m0_unet",
    "decoder": "m2_aux_boundary_only",
    "full": "m6_full",
    "gsl": "gsl",
}
_last_check = 0.0


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def log(**event) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    event = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
    with (OUT / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    print(event, flush=True)


def reason_to_wait(paused: bool = False) -> str | None:
    """GSL training is CPU-bound (plan.md 0.3), and the laptop shares its cooling."""
    listing = subprocess.run(["pgrep", "-af", "train_"], capture_output=True, text=True).stdout
    if re.search(r"train_\w+\.py --config \S*gsl", listing):
        return "gsl"
    query = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"]
    try:
        temperature = int(subprocess.run(query, capture_output=True, text=True).stdout.split()[0])
    except (IndexError, ValueError, OSError):
        return None
    limit = TEMP_RESUME if paused else TEMP_PAUSE
    return f"gpu {temperature} C" if temperature >= limit else None


def guard(force: bool = False) -> None:
    global _last_check
    if (OUT / "STOP").exists():
        log(event="stop requested")
        raise SystemExit(0)
    if not force and time.time() - _last_check < 20:
        return
    _last_check = time.time()
    waiting = reason_to_wait()
    if waiting:
        log(event="paused", reason=waiting)
    while waiting:
        time.sleep(15)
        if (OUT / "STOP").exists():
            log(event="stop requested")
            raise SystemExit(0)
        waiting = reason_to_wait(paused=True)
        if not waiting:
            log(event="resumed")


def april_checkpoint(arm: str, dataset: str, seed: int) -> Path:
    name = {**{k: v[0] for k, v in APRIL.items()}, **REFERENCE}[arm]
    prefix = "isic2018" if dataset in ("isic2018", "isic390") else dataset
    return ROOT / f"outputs/runs/{prefix}_{name}_seed{seed}/best.pt"


def october_checkpoint(arm: str, seed: int) -> Path:
    return ROOT / f"recovered/followup/results/{OCTOBER[arm]}_seed{seed}/best.pt"


def load_april(arm: str, dataset: str, seed: int) -> torch.nn.Module:
    payload = torch.load(
        april_checkpoint(arm, dataset, seed), map_location="cpu", weights_only=False
    )
    state = payload["model"]
    if arm == "a_unet":
        model = UNet(CHANNELS[dataset], CLASSES[dataset], 32)
    else:
        gate = True if arm == "a_full" else APRIL[arm][1]
        if gate != any(k.startswith("boundary_gate.") for k in state):
            raise ValueError(f"gate tensors disagree with the arm definition: {arm}")
        model = BDSLiteUNet(CHANNELS[dataset], CLASSES[dataset], 32, 1, gate, 0.25)
        state = {
            k.replace("projection_head.", "seg_to_boundary_feature.", 1): v
            for k, v in state.items()
        }
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_october(arm: str, seed: int) -> torch.nn.Module:
    payload = torch.load(october_checkpoint(arm, seed), map_location="cpu", weights_only=False)
    model = build_model(payload["config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def manifest(dataset: str, split: str) -> ManifestDataset:
    path = ROOT / f"recovered/followup/splits/{dataset}/{split}.json"
    return ManifestDataset(
        ROOT / f"data/processed/{dataset}",
        split,
        json.loads(path.read_text()),
        CLASSES[dataset],
        CHANNELS[dataset],
    )


def clean390() -> ManifestDataset:
    """The test list of the partition these models actually used; files live in train/ and val/."""
    data = manifest("isic2018", "val")
    root = ROOT / "data/processed/isic2018"
    split = json.loads((ROOT / "splits/isic2018_split.json").read_text())
    train_ids = {Path(i).name for i in split["train"]} | {Path(i).name for i in split["val"]}
    ids = sorted(Path(i).name for i in split["test"])
    if len(ids) != 390 or set(ids) & train_ids:
        raise ValueError("clean 390 list is not disjoint from the models' own partition")
    files = [
        root / s / f"{i}.npz"
        for i in ids
        for s in ("train", "val")
        if (root / s / f"{i}.npz").is_file()
    ]
    if len(files) != 390:
        raise FileNotFoundError("clean 390 sample missing")
    data.files = files
    return data


def preload(data) -> tuple[list[str], torch.Tensor, list[np.ndarray]]:
    guard(force=True)
    ids, images, masks = [], [], []
    for index in range(len(data)):
        item = data[index]
        ids.append(item["id"])
        images.append(item["image"].float())
        masks.append(item["mask"].numpy())
        if index % 64 == 0:
            guard()
    return ids, torch.stack(images).contiguous(), masks


@torch.inference_mode()
def predict(model, images: torch.Tensor) -> np.ndarray:
    out = []
    for start in range(0, images.shape[0], 16):
        guard()
        logits = inference_forward(model, images[start : start + 16])["seg_logits"]
        out.append(logits.argmax(1).numpy().astype(np.uint8))
    return np.concatenate(out)


def evaluate(model, cohort: str, dataset: str, seed: int, ckpt: Path, loaded) -> Path:
    folder = OUT / cohort / f"{dataset}_seed{seed}"
    if (folder / "provenance.json").exists():
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    ids, images, masks = loaded
    preds = predict(model, images)
    rows = []
    for index, (sample, pred, mask) in enumerate(zip(ids, preds, masks, strict=True)):
        rows += [dict(id=sample, **case_metrics(pred, mask, k)) for k in range(1, CLASSES[dataset])]
        if index % 32 == 0:
            guard()
    with (folder / "per_case.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(folder / "predictions.npz", ids=np.array(ids), predictions=preds)
    (folder / "provenance.json").write_text(
        json.dumps(
            dict(
                checkpoint=str(ckpt.relative_to(ROOT)),
                checkpoint_sha256=sha(ckpt),
                n_samples=len(ids),
                evaluator="recover_predictions.case_metrics; CPU fp32; argmax",
                per_case_sha256=sha(folder / "per_case.csv"),
                protocol_sha256=sha(PROTOCOL),
            ),
            indent=1,
        )
        + "\n"
    )
    log(event="done", cohort=cohort, dataset=dataset, seed=seed, n=len(ids))
    return folder


def compare(saved: dict[tuple[str, int], float], folder: Path) -> dict:
    now = {}
    with (folder / "per_case.csv").open() as stream:
        for row in csv.DictReader(stream):
            now[(row["id"], int(row["label"]))] = float(row["dsc"])
    keys = sorted(set(saved) & set(now))
    old, new = np.array([saved[k] for k in keys]), np.array([now[k] for k in keys])
    mean_diff = float(abs(old.mean() - new.mean()))
    share = float((np.abs(old - new) > PARITY_ROW).mean())
    return dict(
        rows=len(keys),
        mean_abs_diff=mean_diff,
        share_rows_beyond=share,
        passed=bool(keys) and mean_diff <= PARITY_MEAN and share <= PARITY_SHARE,
    )


def legacy_rows(arm: str, dataset: str, seed: int) -> dict[tuple[str, int], float]:
    name = {**{k: v[0] for k, v in APRIL.items()}, **REFERENCE}[arm]
    prefix = "isic2018" if dataset == "isic390" else dataset
    saved = {}
    path = ROOT / f"outputs/evaluations/{prefix}_{name}_seed{seed}_test.csv"
    with path.open() as stream:
        for row in csv.DictReader(stream):
            label, dsc = int(row["class_id"]), float(row["dsc"])
            hd95 = float(row["hd95"]) if row["hd95"] not in ("", "nan", "None") else 0.0
            if label > 0 and not (dsc >= 0.999999 and hd95 == 0.0):  # legacy both-empty rows
                saved[(row["sample_id"], label)] = dsc
    return saved


def write_parity(report: dict) -> None:
    (OUT / "parity.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


def read_parity() -> dict:
    path = OUT / "parity.json"
    return json.loads(path.read_text()) if path.exists() else {}


def stage_october(official) -> None:
    report = read_parity()
    data = manifest("isic2018", "val")
    data.files = data.files[:64]
    development = preload(data)
    for arm in OCTOBER:
        for seed in SEEDS:
            key = f"{arm}_seed{seed}"
            if key not in report:
                ids, images, masks = development
                preds = predict(load_october(arm, seed), images)
                folder = OUT / "parity_october" / key
                folder.mkdir(parents=True, exist_ok=True)
                rows = [
                    dict(id=i, **case_metrics(p, m, 1))
                    for i, p, m in zip(ids, preds, masks, strict=True)
                ]
                with (folder / "per_case.csv").open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
                saved = {}
                source = october_checkpoint(arm, seed).parent / "per_case.csv"
                with source.open() as stream:
                    for row in csv.DictReader(stream):
                        saved[(row["id"], 1)] = float(row["dsc"])
                report[key] = compare(saved, folder)
                write_parity(report)
                log(event="parity", model=key, **report[key])
            if report[key]["passed"]:
                evaluate(
                    load_october(arm, seed),
                    arm,
                    "isic2018",
                    seed,
                    october_checkpoint(arm, seed),
                    official,
                )
            else:
                log(event="parity failed; official test not evaluated", model=key)


def stage_april(official) -> None:
    report = read_parity()
    for dataset, data in (
        ("acdc", manifest("acdc", "test")),
        ("isic390", clean390()),
        ("synapse", manifest("synapse", "test")),
    ):
        loaded = preload(data)
        arms = list(APRIL) + (list(REFERENCE) if dataset == "isic390" else [])
        for arm in arms:
            for seed in SEEDS:
                key = f"{arm}_{dataset}_seed{seed}"
                ckpt = april_checkpoint(arm, dataset, seed)
                folder = evaluate(load_april(arm, dataset, seed), arm, dataset, seed, ckpt, loaded)
                if key not in report:
                    report[key] = compare(legacy_rows(arm, dataset, seed), folder)
                    write_parity(report)
                    log(event="parity", model=key, **report[key])
    for arm in APRIL:
        for seed in SEEDS:
            keys = [f"{arm}_{d}_seed{seed}" for d in ("acdc", "isic390", "synapse")]
            if all(report[k]["passed"] for k in keys):
                ckpt = april_checkpoint(arm, "isic2018", seed)
                evaluate(load_april(arm, "isic2018", seed), arm, "isic2018", seed, ckpt, official)
            else:
                log(event="parity failed; official test not evaluated", arm=arm, seed=seed)


def stage_tolerance(official) -> None:
    stage2 = {
        r["run_id"]
        for r in yaml.safe_load(
            (ROOT / "protocols/confirmation/queue_confirmation.yaml").read_text()
        )["runs"]
    }
    ids, _, masks = official
    target = {
        i: mask_to_boundary((m == 1).astype(np.uint8)) for i, m in zip(ids, masks, strict=True)
    }
    folder = OUT / "tolerance"
    folder.mkdir(parents=True, exist_ok=True)
    for arm, name in TEN_RUN.items():
        for seed in range(1, 11):
            path = folder / f"{arm}_seed{seed}.csv"
            if path.exists():
                continue
            run = f"isic2018_{name}_seed{seed}"
            if run in stage2:
                raise ValueError(f"Experiment C run requested: {run}")
            if arm in ("unet", "full") and seed <= 3:
                source = (
                    ROOT / f"results/official_isic/test/{'U' if arm == 'unet' else 'F'}_seed{seed}"
                )
            else:
                source = ROOT / f"results/revision_runs/{run}"
            saved = np.load(source / "preds_test.npz")
            rows = []
            for index, (sample, pred) in enumerate(
                zip(saved["ids"].tolist(), saved["predictions"], strict=True)
            ):
                boundary = mask_to_boundary((pred == 1).astype(np.uint8))
                row = dict(id=sample)
                for tolerance in TOLERANCES:
                    row[f"t{tolerance}"] = boundary_f1(
                        boundary, target[sample], tolerance=tolerance
                    )
                rows.append(row)
                if index % 64 == 0:
                    guard()
            temporary = path.with_suffix(".tmp")
            with temporary.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            temporary.rename(path)
            log(event="tolerance done", arm=arm, seed=seed)


def stage_validate(sets: dict) -> None:
    rng = random.Random(20261107)
    report = read_parity()
    failures, checked = [], 0
    expected = [
        (a, d, s) for a in APRIL for d in ("acdc", "isic390", "synapse", "isic2018") for s in SEEDS
    ]
    expected += [(a, "isic390", s) for a in REFERENCE for s in SEEDS]
    expected += [(a, "isic2018", s) for a in OCTOBER for s in SEEDS]
    protocol = sha(PROTOCOL)
    for arm, dataset, seed in expected:
        folder = OUT / arm / f"{dataset}_seed{seed}"
        name = f"{arm}/{dataset}_seed{seed}"
        if not (folder / "provenance.json").exists():
            failures.append(f"{name}: missing")
            continue
        info = json.loads((folder / "provenance.json").read_text())
        ids, _, masks = sets[dataset]
        saved = np.load(folder / "predictions.npz")
        preds = saved["predictions"]
        reference = dict(zip(ids, masks, strict=True))
        rows = {}
        with (folder / "per_case.csv").open() as stream:
            for row in csv.DictReader(stream):
                rows[(row["id"], int(row["label"]))] = row
        if (
            info["per_case_sha256"] != sha(folder / "per_case.csv")
            or info["protocol_sha256"] != protocol
        ):
            failures.append(f"{name}: hash")
        if saved["ids"].tolist() != ids or preds.dtype != np.uint8 or preds.shape[1:] != (224, 224):
            failures.append(f"{name}: predictions")
        if len(rows) != len(ids) * (CLASSES[dataset] - 1):
            failures.append(f"{name}: row count")
        for index in rng.sample(range(len(ids)), 16):
            sample = ids[index]
            for label in range(1, CLASSES[dataset]):
                p, t = preds[index] == label, reference[sample] == label
                if p.any() or t.any():
                    dsc = 2.0 * (p & t).sum() / (p.sum() + t.sum())
                    if abs(dsc - float(rows[(sample, label)]["dsc"])) > 1e-4:
                        failures.append(f"{name}: DSC recomputation {sample}")
        checked += 1
    summary = dict(
        folders_checked=checked,
        failures=failures,
        parity_failed=sorted(k for k, v in report.items() if not v["passed"]),
        status="PASSED" if not failures else "FAILED",
    )
    (OUT / "validation.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(
        event="validation",
        status=summary["status"],
        failures=len(failures),
        parity_failed=len(summary["parity_failed"]),
    )


def smoke() -> None:
    data = manifest("isic2018", "val")
    item = data[0]["image"].float()[None]
    for arm in OCTOBER:
        print(arm, tuple(predict(load_october(arm, 1), item).shape), flush=True)
    for arm in list(APRIL) + list(REFERENCE):
        print(arm, tuple(predict(load_april(arm, "isic2018", 1), item).shape), flush=True)
    for dataset in ("acdc", "synapse"):
        one = manifest(dataset, "val")[0]["image"].float()[None]
        for arm in APRIL:
            print(arm, dataset, tuple(predict(load_april(arm, dataset, 1), one).shape), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible; refusing to run")
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke()
        return
    if not (OUT / "FREEZE.json").exists():
        raise SystemExit("protocol not frozen")
    log(event="start", pid=os.getpid())
    official = preload(OfficialTestDataset())
    stage_october(official)
    stage_april(official)
    stage_tolerance(official)
    sets = {"isic2018": official}
    for name, data in (
        ("acdc", manifest("acdc", "test")),
        ("isic390", clean390()),
        ("synapse", manifest("synapse", "test")),
    ):
        sets[name] = preload(data)
    stage_validate(sets)
    log(event="finished")


if __name__ == "__main__":
    main()
