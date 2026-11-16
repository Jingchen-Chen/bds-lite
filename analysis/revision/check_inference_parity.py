"""Gate official inference on exact converter and selected-mask reproduction."""

import fcntl
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from bds_lite.data.converters import load_rgb_image
from bds_lite.training import build_model
from bds_lite.training.runner import inference_forward
from bds_lite.utils import seed_everything

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / "recovered/followup"
    ids = sorted(json.loads((source / "splits/isic2018/val.json").read_text())["sample_ids"])[:8]
    batch = []
    for sid in ids:
        raw = ROOT / f"data/raw/isic2018/ISIC2018_Task1-2_Training_Input/{sid}.jpg"
        image = load_rgb_image(raw, (224, 224))
        with np.load(source / f"data/followup/isic2018/val/{sid}.npz") as z:
            if not np.array_equal(image, z["image"]):
                raise ValueError("raw preprocessing differs")
        batch.append(image)
    results = []
    with (ROOT / "revision/.gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if active:
            raise RuntimeError("GPU occupied")
        for arm in "USF":
            for seed in [1, 2, 3]:
                seed_everything(seed)
                folder = source / "results/followup_20261018_v2" / f"isic2018_{arm}_seed{seed}"
                payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
                model = build_model(payload["config"])
                model.load_state_dict(payload["model"], strict=True)
                model.cuda().eval()
                with torch.inference_mode():
                    logits = inference_forward(
                        model, torch.from_numpy(np.stack(batch)).contiguous().cuda()
                    )
                    pred = logits["seg_logits"].argmax(1).cpu().numpy()
                mismatch = sum(
                    int(np.count_nonzero(a != np.load(folder / f"predictions/{sid}.npy")))
                    for sid, a in zip(ids, pred, strict=True)
                )
                if mismatch:
                    raise ValueError(f"{arm}/{seed}: {mismatch} different pixels")
                results.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        ids=ids,
                        pixel_mismatches=mismatch,
                        raw_preprocessing_exact=True,
                        cudnn_deterministic=True,
                    )
                )
                del model, payload
                torch.cuda.empty_cache()
    (ROOT / "results/validation/official_inference_parity.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print("All nine checkpoints reproduce eight masks each exactly")


if __name__ == "__main__":
    main()
