"""Short engineering check of actual full-batch AMP/probe behavior, not a study run."""

import copy
import fcntl
import json
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bds_lite.training import BDSLiteLoss, build_model
from bds_lite.training.losses import weights_from_config
from bds_lite.utils import seed_everything
from scripts.revision.train_run import epoch_train
from scripts.revision.training_support import ManifestDataset, restore_rng, rng_state

ROOT = Path(__file__).resolve().parents[2]


def main():
    with (ROOT / "revision/.gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        occupied = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if occupied:
            raise RuntimeError("GPU occupied; no process interrupted")
        torch.set_num_threads(4)
        rows = []
        for dataset in ["isic2018", "acdc", "synapse"]:
            seed_everything(1)
            checkpoint = (
                ROOT / f"results/checkpoints/{dataset}_p1b_bdslite_full_current_seed1/best.pt"
            )
            config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
            manifest = json.loads(
                (ROOT / f"recovered/followup/splits/{dataset}/train.json").read_text()
            )
            data = ManifestDataset(
                ROOT / config["dataset"]["processed_dir"],
                "train",
                manifest,
                config["dataset"]["num_classes"],
                config["dataset"]["in_channels"],
                True,
                True,
            )
            batch = next(iter(DataLoader(data, batch_size=config["trainer"]["batch_size"])))
            model = build_model(config).cuda()
            initial = copy.deepcopy(model.state_dict())
            rng = rng_state()
            states, probes, durations = [], [], []
            torch.cuda.reset_peak_memory_stats()
            for enabled in [False, True]:
                model.load_state_dict(initial, strict=True)
                restore_rng(rng)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=config["trainer"]["lr"], weight_decay=1e-5
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
                criterion = BDSLiteLoss(weights_from_config(config))
                torch.cuda.synchronize()
                began = time.monotonic()
                metrics, _ = epoch_train(
                    model, [batch], criterion, optimizer, scheduler, 1, 49, probes, enabled
                )
                torch.cuda.synchronize()
                durations.append(time.monotonic() - began)
                states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            exact = all(torch.equal(v, states[1][k]) for k, v in states[0].items())
            row = dict(
                dataset=dataset,
                batch_size=config["trainer"]["batch_size"],
                updated_state_exact=exact,
                gradient_probes=len(probes),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                seconds_without_probe=durations[0],
                seconds_with_probe=durations[1],
                amp_skipped_steps=metrics["amp_skipped_steps"],
                probe=probes[0],
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            if not exact or len(probes) != 1:
                raise AssertionError("observational probe altered full-batch next update")
            del model, optimizer, scheduler, initial, states, batch
            torch.cuda.empty_cache()
        path = ROOT / "results/validation/training_gpu_engineering.json"
        path.write_text(
            json.dumps(
                dict(scope="engineering smoke, not scientific training", rows=rows), indent=2
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
