"""Queue-only, epoch-resumable training for the ACDC/Synapse KD study (teacher and students).

Mirrors scripts/kd/train_kd.py (itself a mirror of scripts/revision/train_run.py::main), reusing the
unchanged epoch_train, validate_one_epoch, evaluate, and resume state. Differences:
- the manifests' test split is never built: no test prediction or metric is computed here;
- multiclass L_kd (scripts/kd_mc/loss.py) from cached K-class teacher probabilities;
- the multiclass teacher wrapper (scripts/kd_mc/teacher.py).
Stop marker: revision/STOP_KD_TRAINING.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from bds_lite.training import BDSLiteLoss, build_model
from bds_lite.training.losses import weights_from_config
from bds_lite.training.runner import validate_one_epoch
from bds_lite.utils import seed_everything
from scripts.kd.common import CUDA_VERSION, TORCH_VERSION, sha, write_json
from scripts.kd.train_kd import environment
from scripts.kd_mc.common import GPU_LOCK, ROOT, RUNS, STOP, cache_run, kd_sources, require_protocol
from scripts.kd_mc.data import CACHE_NAME, TeacherCacheDatasetMC
from scripts.kd_mc.loss import KDCriterionMC, kd_loss_mc
from scripts.kd_mc.teacher import build_teacher
from scripts.revision.train_run import cpu_state, epoch_train, evaluate
from scripts.revision.training_support import (
    ManifestDataset,
    atomic_torch_save,
    restore_rng,
    rng_state,
)

STOP_REQUESTED = False
PROBE_EPOCHS = {0, 1, 25, 50, 75, 100, 125, 150}
EPOCHS = 150


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def verified_cache(dataset: str) -> tuple[Path, str]:
    folder = RUNS / cache_run(dataset)
    marker = folder / "cache_complete.json"
    if not marker.is_file():
        raise FileNotFoundError("teacher cache step has not completed")
    record = json.loads(marker.read_text())
    if record["gate"]["status"] != "PASSED":
        raise RuntimeError("teacher failed the quality gate; students do not start")
    digest = sha(folder / CACHE_NAME)
    if digest != record["cache_sha256"]:
        raise ValueError("teacher cache bytes differ from the completed cache record")
    return folder, digest


def probe_batch(dataset, size: int = 8) -> dict:
    """First `size` training slices in manifest order with any foreground label; no random draws.

    (Many ACDC/Synapse slices are background-only, which would give an empty band.)
    """
    items = []
    for index in range(len(dataset)):
        item = dataset[index]
        if int(item["mask"].max()) > 0:
            items.append(item)
        if len(items) == size:
            break
    return {k: torch.stack([item[k] for item in items]).cuda() for k in items[0] if k != "id"}


def kd_gradient_probe(model, batch, criterion) -> dict:
    """FP32 train-mode gradient norms of weighted L_seg, L_boundary and L_kd over all parameters."""
    state = rng_state()
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    was_training = model.training
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    base, w = criterion.base, criterion.base.weights
    try:
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            outputs = model(batch["image"].float())
            logits, mask = outputs["seg_logits"], batch["mask"]
            terms = {
                "seg": w.ce * F.cross_entropy(logits, mask) + w.dice * base.dice(logits, mask),
                "boundary": w.boundary
                * (
                    F.binary_cross_entropy_with_logits(
                        outputs["boundary_logits"], batch["boundary"]
                    )
                    + base.boundary_dice(outputs["boundary_logits"], batch["boundary"])
                ),
                "kd": criterion.lambda_kd * kd_loss_mc(logits, batch["teacher_prob"]),
            }
            total, _ = criterion(outputs, batch)
            if not torch.allclose(sum(terms.values()), total, rtol=1e-5, atol=1e-6):
                raise AssertionError("probe objective differs from the active training loss")
            grads = {}
            for name, loss in terms.items():
                gs = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                grads[name] = torch.cat(
                    [
                        (g if g is not None else torch.zeros_like(p)).flatten()
                        for g, p in zip(gs, parameters, strict=True)
                    ]
                ).double()
        seg_norm = torch.linalg.vector_norm(grads["seg"])
        result = {"loss_" + k: float(v.detach()) for k, v in terms.items()}
        result["seg_norm"] = float(seg_norm)
        for name in ["boundary", "kd"]:
            norm = torch.linalg.vector_norm(grads[name])
            result[name + "_norm"] = float(norm)
            result[name + "_norm_ratio"] = float(norm / seg_norm) if seg_norm > 0 else None
            result[name + "_cosine"] = (
                float(torch.dot(grads["seg"], grads[name]) / (seg_norm * norm))
                if seg_norm > 0 and norm > 0
                else None
            )
        return result
    finally:
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])
        restore_rng(state)
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    lock_fd = int(os.environ["BDS_REVISION_GPU_LOCK_FD"])
    if os.fstat(lock_fd).st_ino != GPU_LOCK.stat().st_ino:
        raise RuntimeError("queue lock identity mismatch")
    config = yaml.safe_load(args.config.read_text())
    require_protocol(config)
    role, dataset = config["kd"]["role"], config["dataset"]["name"]
    run = args.output.resolve()
    if not run.is_relative_to(RUNS):
        raise ValueError("new outputs must be under results/revision_runs")
    if torch.__version__ != TORCH_VERSION or torch.version.cuda != CUDA_VERSION:
        raise RuntimeError("environment differs from freeze")
    torch.set_num_threads(4)
    seed_everything(config["project"]["seed"])
    classes = config["dataset"]["num_classes"]
    revision = config["revision"]
    cfg_hash = sha(args.config)
    sources = kd_sources()
    cache_dir, cache_sha = verified_cache(dataset) if role == "student" else (None, None)
    sets, fingerprints = {}, {}
    for split in ["train", "val"]:  # the manifest's test split is deliberately not built
        record = revision["manifests"][split]
        path = ROOT / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError("manifest bytes changed")
        manifest = json.loads(path.read_text())
        options = (
            ROOT / config["dataset"]["processed_dir"],
            split,
            manifest,
            classes,
            config["dataset"]["in_channels"],
            split == "train" and bool(config["loss"]["boundary_weight"]),
            split == "train" and bool(config["loss"]["surface_weight"]),
        )
        if role == "student" and split == "train":
            sets[split] = TeacherCacheDatasetMC(*options, cache_dir=cache_dir)
        else:
            sets[split] = ManifestDataset(*options)
        fingerprints[split] = {p.stem: sha(p) for p in sets[split].files}
    if cache_sha is not None:
        fingerprints["teacher_cache"] = {CACHE_NAME: cache_sha}
    generator = torch.Generator().manual_seed(config["project"]["seed"])
    loaders = {
        split: DataLoader(
            dataset_,
            batch_size=config["trainer"]["batch_size"],
            shuffle=split == "train",
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
            generator=generator if split == "train" else None,
        )
        for split, dataset_ in sets.items()
    }
    if role == "student":
        model = build_model(config).cuda()
        criterion = KDCriterionMC(
            BDSLiteLoss(weights_from_config(config)), config["kd"]["lambda_kd"]
        )
    else:
        model = build_teacher(config, pretrained=not args.resume).cuda()
        criterion = BDSLiteLoss(weights_from_config(config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["trainer"]["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    start, step, history, probes, best, best_state, best_epoch = 1, 0, [], [], -1.0, None, 0
    batch_for_probe = probe_batch(sets["train"]) if role == "student" else None
    if args.resume:
        previous = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        if (
            previous["config_sha256"] != cfg_hash
            or previous["source_hashes"] != sources
            or previous["data_hashes"] != fingerprints
        ):
            raise ValueError("resume config/code/data mismatch; preserve and investigate")
        model.load_state_dict(previous["model"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        start, step = previous["epoch"] + 1, previous["step"]
        history, probes = previous["history"], previous["probes"]
        best, best_state, best_epoch = (
            previous["best_dsc"],
            previous["best_state"],
            previous["best_epoch"],
        )
        generator.set_state(previous["loader_rng"])
        restore_rng(previous["rng"])
    else:
        if (run / "config.json").exists() or (run / "last.pt").exists():
            raise FileExistsError("fresh run refuses existing scientific outputs")
        write_json(run / "config.json", config)
        write_json(
            run / "provenance.json",
            dict(
                config_sha256=cfg_hash,
                source_hashes=sources,
                data_hashes=fingerprints,
                environment=environment(),
            ),
        )
        if role == "student":
            probes.append(dict(epoch=0, **kd_gradient_probe(model, batch_for_probe, criterion)))
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    for epoch in range(start, EPOCHS + 1):
        began = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        criterion.set_epoch(epoch)
        train, step = epoch_train(
            model, loaders["train"], criterion, optimizer, scheduler, epoch, step, [], False
        )
        val = validate_one_epoch(model, loaders["val"], criterion, torch.device("cuda"), classes)
        if not np.isfinite(val["val_dsc"]):
            raise FloatingPointError("nonfinite validation selection metric")
        if val["val_dsc"] > best:
            best, best_epoch, best_state = val["val_dsc"], epoch, cpu_state(model)
        seconds = time.monotonic() - began
        memory = dict(
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        )
        if role == "student" and epoch in PROBE_EPOCHS:
            probes.append(dict(epoch=epoch, **kd_gradient_probe(model, batch_for_probe, criterion)))
        history.append(dict(epoch=epoch, train=train, val=val, seconds=seconds, **memory))
        write_json(
            run / "progress.json",
            dict(epoch=epoch, step=step, best_dsc=best, best_epoch=best_epoch, last=history[-1]),
        )
        print(json.dumps(history[-1]), flush=True)
        stop_requested = STOP_REQUESTED or STOP.exists()
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS or stop_requested:
            atomic_torch_save(
                dict(
                    epoch=epoch,
                    step=step,
                    model=cpu_state(model),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    rng=rng_state(),
                    loader_rng=generator.get_state(),
                    best_dsc=best,
                    best_state=best_state,
                    best_epoch=best_epoch,
                    history=history,
                    probes=probes,
                    config_sha256=cfg_hash,
                    source_hashes=sources,
                    data_hashes=fingerprints,
                ),
                run / "last.pt",
            )
            atomic_torch_save(
                dict(model=best_state, epoch=best_epoch, config=config), run / "best.pt"
            )
            write_json(run / "history.json", history)
            write_json(run / "kd_probes.json", probes)
        if stop_requested:
            raise SystemExit(130)
    model.load_state_dict(best_state, strict=True)
    metrics = {"val": evaluate(model, loaders["val"], classes, run, "val")}
    artifacts = {
        name: sha(run / name)
        for name in [
            "best.pt",
            "config.json",
            "preds_val.npz",
            "history.json",
            "kd_probes.json",
            "per_case_val.csv",
            "provenance.json",
        ]
    }
    write_json(
        run / "final_metrics.json",
        dict(
            config_sha256=cfg_hash,
            epochs=EPOCHS,
            selected_epoch=best_epoch,
            best_val_dsc=best,
            metrics=metrics,
            artifacts=artifacts,
            test_evaluated=False,
        ),
    )


if __name__ == "__main__":
    main()
