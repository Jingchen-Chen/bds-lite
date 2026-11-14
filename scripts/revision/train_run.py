"""Queue-only, epoch-resumable locked-protocol training with optional WP5 probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from analysis.revision.recover_predictions import Cohort, case_metrics
from bds_lite.training import BDSLiteLoss, build_model
from bds_lite.training.losses import weights_from_config
from bds_lite.training.runner import inference_forward, move_batch_to_device, validate_one_epoch
from bds_lite.utils import seed_everything
from scripts.revision.run_queue import require_protocol
from scripts.revision.training_support import (
    ManifestDataset,
    atomic_torch_save,
    gradient_probe,
    restore_rng,
    rng_state,
)

ROOT = Path(__file__).resolve().parents[2]
STOP = False


class OfficialTestDataset(torch.utils.data.Dataset):
    """Official ISIC test, evaluated after selection on the historical development split."""

    def __init__(self):
        self.cohort = Cohort(dict(dataset="isic2018", split="official_test"))

    def __len__(self):
        return len(self.cohort.ids)

    def __getitem__(self, index):
        sid = self.cohort.ids[index]
        image, mask, _ = self.cohort.get(sid)
        return dict(
            image=torch.from_numpy(np.ascontiguousarray(image)),
            mask=torch.from_numpy(mask.astype(np.int64)),
            id=sid,
        )


def request_stop(signum, frame):
    global STOP
    STOP = True


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def epoch_train(model, loader, criterion, optimizer, scheduler, epoch, step, probes, enabled):
    model.train()
    # Preserve historical runner behavior: a fresh GradScaler at each epoch boundary.
    device = next(model.parameters()).device
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    totals, skipped, consecutive_skips, overflows = {}, 0, 0, []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if enabled and step % 50 == 0:
            probes.append(dict(epoch=epoch, step=step, **gradient_probe(model, batch, criterion)))
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            loss, scalars = criterion(model(batch["image"]), batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite training loss epoch={epoch} step={step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        overflow = scaler.get_scale() < old_scale
        skipped += int(overflow)
        consecutive_skips = consecutive_skips + 1 if overflow else 0
        if overflow:
            overflows.append(dict(step=step, sample_ids=batch["id"], scale=scaler.get_scale()))
        if consecutive_skips > 8:
            raise FloatingPointError("more than eight consecutive AMP overflows")
        if not torch.isfinite(norm) and scaler.get_scale() >= old_scale:
            raise FloatingPointError("nonfinite gradient without AMP overflow handling")
        for key, value in scalars.items():
            totals[key] = totals.get(key, 0.0) + value
    scheduler.step()
    return dict(
        **{k: v / len(loader) for k, v in totals.items()},
        lr=float(optimizer.param_groups[0]["lr"]),
        amp_skipped_steps=skipped,
        amp_overflows=overflows,
    ), step


@torch.no_grad()
def evaluate(model, loader, classes, run, split):
    model.eval()
    predictions, ids, rows = [], [], []
    for batch in loader:
        output = inference_forward(model, batch["image"].cuda())
        pred = output["seg_logits"].argmax(1).cpu().numpy().astype(np.uint8)
        for i, sample in enumerate(batch["id"]):
            truth = batch["mask"][i].numpy()
            predictions.append(pred[i])
            ids.append(sample)
            for label in range(1, classes):
                rows.append(dict(id=sample, **case_metrics(pred[i], truth, label)))
    path = run / f"preds_{split}.npz"
    with path.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, ids=np.array(ids), predictions=np.stack(predictions))
    path.with_suffix(".tmp").replace(path)
    with (run / f"per_case_{split}.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for metric in ["dsc", "iou", "boundary_f1", "hd95", "assd"]:
        values = [r[metric] for r in rows if r[metric] is not None]
        summary[metric] = dict(
            mean=float(np.mean(values)) if values else None, valid=len(values), total=len(rows)
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # Parent retains and passes the same flock open description, including if orphaned.
    lock_fd = int(os.environ["BDS_REVISION_GPU_LOCK_FD"])
    if os.fstat(lock_fd).st_ino != (ROOT / "revision/.gpu.lock").stat().st_ino:
        raise RuntimeError("queue lock identity mismatch")
    config = yaml.safe_load(args.config.read_text())
    require_protocol(config)
    run = args.output.resolve()
    if not run.is_relative_to(ROOT / "results/revision_runs"):
        raise ValueError("new outputs must be under results/revision_runs")
    if torch.__version__ != "2.11.0+cu130" or torch.version.cuda != "13.0":
        raise RuntimeError("environment differs from freeze")
    torch.set_num_threads(4)
    seed_everything(config["project"]["seed"])
    classes = config["dataset"]["num_classes"]
    revision = config["revision"]
    cfg_hash = sha(args.config)
    sources = {
        str(p.relative_to(ROOT)): sha(p)
        for folder in ["src/bds_lite", "scripts/revision"]
        for p in (ROOT / folder).rglob("*.py")
    }
    manifests, sets, fingerprints = {}, {}, {}
    for split, record in revision["manifests"].items():
        path = ROOT / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError("manifest bytes changed")
        manifests[split] = json.loads(path.read_text())
        sets[split] = ManifestDataset(
            ROOT / config["dataset"]["processed_dir"],
            split,
            manifests[split],
            classes,
            config["dataset"]["in_channels"],
            split == "train" and bool(config["loss"]["boundary_weight"]),
            split == "train" and bool(config["loss"]["surface_weight"]),
        )
        fingerprints[split] = {p.stem: sha(p) for p in sets[split].files}
    if config["dataset"]["name"] == "isic2018":
        sets["test"] = OfficialTestDataset()
        archives = ROOT / "data/revision/isic2018_official/archives"
        fingerprints["test"] = {p.name: sha(p) for p in sorted(archives.glob("*Test*.zip"))}
        if len(fingerprints["test"]) != 2:
            raise ValueError("official test archives missing")
    sources["analysis/revision/recover_predictions.py"] = sha(
        ROOT / "analysis/revision/recover_predictions.py"
    )
    generator = torch.Generator().manual_seed(config["project"]["seed"])
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config["trainer"]["batch_size"],
            shuffle=split == "train",
            num_workers=0
            if isinstance(dataset, OfficialTestDataset)
            else config["data"]["num_workers"],
            pin_memory=True,
            generator=generator if split == "train" else None,
        )
        for split, dataset in sets.items()
    }
    model = build_model(config).cuda()
    criterion = BDSLiteLoss(weights_from_config(config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["trainer"]["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    start, step, history, probes, best, best_state, best_epoch = 1, 0, [], [], -1.0, None, 0
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
            dict(config_sha256=cfg_hash, source_hashes=sources, data_hashes=fingerprints),
        )
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    for epoch in range(start, 151):
        began = time.monotonic()
        criterion.set_epoch(epoch)
        train, step = epoch_train(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            probes,
            revision["gradient_probe"],
        )
        val = validate_one_epoch(model, loaders["val"], criterion, torch.device("cuda"), classes)
        if not np.isfinite(val["val_dsc"]):
            raise FloatingPointError("nonfinite validation selection metric")
        if val["val_dsc"] > best:
            best, best_epoch, best_state = val["val_dsc"], epoch, cpu_state(model)
        history.append(dict(epoch=epoch, train=train, val=val, seconds=time.monotonic() - began))
        write_json(
            run / "progress.json",
            dict(epoch=epoch, step=step, best_dsc=best, best_epoch=best_epoch, last=history[-1]),
        )
        print(json.dumps(history[-1]), flush=True)
        stop_requested = STOP or (ROOT / "revision/STOP_TRAINING").exists()
        if epoch == 1 or epoch % 10 == 0 or epoch == 150 or stop_requested:
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
            write_json(run / "gradient_probes.json", probes)
        if stop_requested:
            raise SystemExit(130)
    model.load_state_dict(best_state, strict=True)
    metrics = {
        split: evaluate(model, loaders[split], classes, run, split) for split in ["val", "test"]
    }
    artifacts = {
        name: sha(run / name)
        for name in [
            "best.pt",
            "config.json",
            "preds_val.npz",
            "preds_test.npz",
            "history.json",
            "gradient_probes.json",
            "per_case_val.csv",
            "per_case_test.csv",
            "provenance.json",
        ]
    }
    write_json(
        run / "final_metrics.json",
        dict(
            config_sha256=cfg_hash,
            epochs=150,
            selected_epoch=best_epoch,
            best_val_dsc=best,
            metrics=metrics,
            artifacts=artifacts,
        ),
    )


if __name__ == "__main__":
    main()
