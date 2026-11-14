"""Queue-only, epoch-resumable locked-protocol training for revision comparators.

This is scripts/revision/train_run.py:main with exactly three changes:
1. model/criterion come from scripts/comparators/components.build(config);
2. the training dataset additionally provides Kervadec distance maps (C1) when
   comparator.kervadec_dist is true;
3. source hashes additionally cover scripts/comparators and the fetched LB-UNet file.
The epoch loop, AMP/GradScaler policy, clipping, checkpoint/resume contract,
selection rule, evaluation and prediction archives are imported unchanged from
train_run.py. Gradient probes are not supported here.
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
from torch.utils.data import DataLoader

from bds_lite.training.runner import validate_one_epoch
from bds_lite.utils import seed_everything
from scripts.comparators.components import build, kervadec_dist_maps
from scripts.revision.run_queue import require_protocol
from scripts.revision.train_run import (
    OfficialTestDataset,
    cpu_state,
    epoch_train,
    evaluate,
    sha,
    write_json,
)
from scripts.revision.training_support import (
    ManifestDataset,
    atomic_torch_save,
    restore_rng,
    rng_state,
)

ROOT = Path(__file__).resolve().parents[2]
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


class ComparatorDataset(ManifestDataset):
    def __init__(self, *args, kervadec=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.kervadec = kervadec

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.kervadec:
            dist = kervadec_dist_maps(item["mask"].numpy(), self.classes)
            item["kervadec_dist"] = torch.from_numpy(dist)
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
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
    if config["revision"].get("gradient_probe"):
        raise ValueError("gradient probes are not supported by the comparator adapter")
    torch.set_num_threads(4)
    seed_everything(config["project"]["seed"])
    classes = config["dataset"]["num_classes"]
    revision = config["revision"]
    comparator = config["comparator"]
    cfg_hash = sha(args.config)
    sources = {
        str(p.relative_to(ROOT)): sha(p)
        for folder in ["src/bds_lite", "scripts/revision", "scripts/comparators"]
        for p in (ROOT / folder).rglob("*.py")
    }
    sources["analysis/revision/recover_predictions.py"] = sha(
        ROOT / "analysis/revision/recover_predictions.py"
    )
    manifests, sets, fingerprints = {}, {}, {}
    for split, record in revision["manifests"].items():
        path = ROOT / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError("manifest bytes changed")
        manifests[split] = json.loads(path.read_text())
        train = split == "train"
        sets[split] = ComparatorDataset(
            ROOT / config["dataset"]["processed_dir"],
            split,
            manifests[split],
            classes,
            config["dataset"]["in_channels"],
            train and bool(comparator.get("boundary_target")),
            False,
            kervadec=train and bool(comparator.get("kervadec_dist")),
        )
        fingerprints[split] = {p.stem: sha(p) for p in sets[split].files}
    if config["dataset"]["name"] == "isic2018":
        sets["test"] = OfficialTestDataset()
        archives = ROOT / "data/revision/isic2018_official/archives"
        fingerprints["test"] = {p.name: sha(p) for p in sorted(archives.glob("*Test*.zip"))}
        if len(fingerprints["test"]) != 2:
            raise ValueError("official test archives missing")
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
    model, criterion = build(config)
    if comparator["kind"] == "c4_lbunet":
        from scripts.comparators.fetch_lbunet import PATH

        sources[str(PATH.relative_to(ROOT))] = sha(PATH)
    model = model.cuda()
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
            model, loaders["train"], criterion, optimizer, scheduler, epoch, step, probes, False
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
