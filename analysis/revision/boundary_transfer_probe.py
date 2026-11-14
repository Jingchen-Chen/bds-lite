# ruff: noqa: E501
"""Does distillation move boundary information into the deployed features? Inference only.

Accuracy-independent diagnostics on existing checkpoints; no model is trained or changed. Exploratory
and post hoc: not part of prereg_revision.md. For each retrained arm, seed, and dataset:

  fidelity     R^2 of the deployed projection q_s for the teacher feature q_b (test images).
  head_q_s     average precision of the frozen teacher head applied to deployed q_s, against the
               union boundary target. Requires no fitting. head_q_b is the teacher itself.
  probe_F_s    average precision of a closed-form ridge readout from the pre-gate decoder feature
               F_s to the boundary target, fitted on validation pixels and scored on test pixels.
               This asks whether F_s itself, not only the projection, holds more boundary signal.
  probe_logit  the same readout from the K class logits, as a reference: how much boundary signal
               is available from the prediction alone.
  gate_all, gate_band, gate_off   mean |1 + alpha (g - 0.5) - 1| overall, on, and off the boundary band.

Output: revision/tables/families/boundary_transfer_probe.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from analysis.revision.evaluate_extra_cohorts import CHANNELS, CLASSES, dataset_for  # noqa: E402
from bds_lite.data.boundary import mask_to_boundary  # noqa: E402
from bds_lite.models import BDSLiteUNet, UNet  # noqa: E402

FOLLOWUP = ROOT / "recovered/followup/results/followup_20261018_v2"
RUNS = ROOT / "results/revision_runs"
CKPT = ROOT / "results/checkpoints"
OUT = ROOT / "revision/tables/families/boundary_transfer_probe.csv"
DATASETS = ["isic2018", "acdc", "synapse"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIXELS = 3000
LIMIT = {"test": 300, "val": 150}


def path_of(arm: str, ds: str, seed: int) -> Path:
    unweighted = "_unweighted" if ds == "acdc" else ""
    if arm == "M0":
        return CKPT / f"{ds}_p1b_unet_seed{seed}/best.pt"
    if arm == "GSL":
        return RUNS / f"{ds}_gsl_seed{seed}/best.pt"
    if arm == "M2":
        return RUNS / f"{ds}_m2_aux_boundary_only{unweighted}_seed{seed}/best.pt"
    if arm == "M3":
        if ds == "isic2018":
            return FOLLOWUP / f"isic2018_B_seed{seed}/best.pt"
        return RUNS / f"{ds}_m3_no_sdf{unweighted}_seed{seed}/best.pt"
    if arm == "M4":
        return RUNS / f"{ds}_m4_no_distill{unweighted}_seed{seed}/best.pt"
    if arm == "M5":
        return RUNS / f"{ds}_m5_no_gate{unweighted}_seed{seed}/best.pt"
    if arm == "M6":
        if ds == "acdc":
            return RUNS / f"acdc_m6_full_unweighted_seed{seed}/best.pt"
        return CKPT / f"{ds}_p1b_bdslite_full_current_seed{seed}/best.pt"
    raise KeyError(arm)


ALPHA = {"M2": 0.0, "M3": 0.25, "M4": 0.25, "M5": 0.0, "M6": 0.25}


def load(arm: str, ds: str, seed: int) -> torch.nn.Module:
    payload = torch.load(path_of(arm, ds, seed), map_location="cpu", weights_only=False)
    state = payload["model"]
    if arm in ("M0", "GSL"):
        model = UNet(CHANNELS[ds], CLASSES[ds], 32)
    else:
        model = BDSLiteUNet(CHANNELS[ds], CLASSES[ds], 32, 1, True, ALPHA[arm])
    state = {
        k.replace("projection_head.", "seg_to_boundary_feature.", 1): v for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def subset(ds: str, split: str) -> list[tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    """Evenly spaced images; on ACDC and Synapse only slices with foreground. Loaded once."""
    data = dataset_for(ds, split)
    index = list(range(len(data)))
    if ds != "isic2018":
        index = [i for i in index if data[i]["mask"].max() > 0]
    step = max(1, len(index) // LIMIT[split])
    items = []
    for i in index[::step][: LIMIT[split]]:
        item = data[i]
        image, mask = item["image"].float(), item["mask"]
        items.append((image, mask, mask_to_boundary(mask.numpy()).astype(bool)))
    return items


@torch.inference_mode()
def collect(model, arm: str, items_all: list, rng_seed: int) -> dict:
    """Sampled pixels with uniform probability, so boundary prevalence is preserved."""
    rng = np.random.default_rng(rng_seed)
    keep: dict[str, list[np.ndarray]] = {}
    sse = sst = 0.0
    gate = {"all": [], "band": [], "off": []}
    bds = arm not in ("M0", "GSL")
    for start in range(0, len(items_all), 8):
        items = items_all[start : start + 8]
        x = torch.stack([d[0] for d in items]).to(DEVICE)
        y = np.stack([d[2] for d in items])
        feats = model.encoder(x)
        decoder = model.seg_decoder if bds else model.decoder
        logits, f_s = decoder(feats)
        parts = {"F": f_s, "logit": logits}
        if bds:
            q_s = model.seg_to_boundary_feature(f_s)
            z_b, q_b = model.boundary_decoder(feats)
            out = model.boundary_decoder.out
            parts |= {"head_qs": out(q_s), "head_qb": z_b}
            sse += float(((q_s - q_b) ** 2).sum())
            mean = q_b.mean((0, 2, 3), keepdim=True)
            sst += float(((q_b - mean) ** 2).sum())
            scale = model.boundary_gate.scale
            g = torch.sigmoid(model.boundary_gate.to_gate(q_s))
            factor = (scale * (g - 0.5)).abs().mean(1).cpu().numpy()
            gate["all"].append(factor.ravel())
            gate["band"].append(factor[y])
            gate["off"].append(factor[~y])
        hw = y.shape[1] * y.shape[2]
        for b in range(len(items)):
            pick = rng.choice(hw, size=PIXELS, replace=False)
            keep.setdefault("y", []).append(y[b].ravel()[pick])
            for name, tensor in parts.items():
                value = tensor[b].flatten(1)[:, torch.as_tensor(pick, device=tensor.device)]
                keep.setdefault(name, []).append(value.T.float().cpu().numpy())
    result = {k: np.concatenate(v) for k, v in keep.items()}
    result["fidelity"] = 1 - sse / sst if bds else float("nan")
    for k, v in gate.items():
        result[f"gate_{k}"] = float(np.concatenate(v).mean()) if v else float("nan")
    return result


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    hits = label[order].astype(float)
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, hits.size + 1)
    return float((precision * hits).sum() / max(hits.sum(), 1))


def ridge_ap(train_x, train_y, test_x, test_y) -> float:
    mean, sd = train_x.mean(0), train_x.std(0) + 1e-6
    a = np.c_[(train_x - mean) / sd, np.ones(len(train_x))].astype(np.float64)
    w = np.linalg.solve(a.T @ a + 1.0 * np.eye(a.shape[1]), a.T @ train_y.astype(np.float64))
    b = np.c_[(test_x - mean) / sd, np.ones(len(test_x))]
    return average_precision(b @ w, test_y)


def main() -> None:
    """With --update DATASET ARM, recompute that cell and keep the other rows of the existing file."""
    update = sys.argv[2:4] if sys.argv[1:2] == ["--update"] else None
    rows = []
    if update:
        with OUT.open() as stream:
            rows = [
                {k: (v if k in ("dataset", "arm") else float(v)) for k, v in r.items()}
                for r in csv.DictReader(stream)
                if [r["dataset"], r["arm"]] != update
            ]
    for ds in [update[0]] if update else DATASETS:
        sets = {s: subset(ds, s) for s in ("val", "test")}
        print(ds, {k: len(v) for k, v in sets.items()}, flush=True)
        for arm in [update[1]] if update else ("M0", "GSL", "M2", "M3", "M4", "M5", "M6"):
            for seed in (1, 2, 3):
                model = load(arm, ds, seed)
                val = collect(model, arm, sets["val"], 1000 + seed)
                test = collect(model, arm, sets["test"], 2000 + seed)
                row = dict(dataset=ds, arm=arm, seed=seed, prevalence=float(test["y"].mean()))
                row["probe_F_s"] = ridge_ap(val["F"], val["y"], test["F"], test["y"])
                row["probe_logit"] = ridge_ap(val["logit"], val["y"], test["logit"], test["y"])
                row["fidelity"] = test["fidelity"]
                for name, key in (("head_q_s", "head_qs"), ("head_q_b", "head_qb")):
                    row[name] = (
                        average_precision(test[key][:, 0], test["y"])
                        if key in test
                        else float("nan")
                    )
                for k in ("all", "band", "off"):
                    row[f"gate_{k}"] = test[f"gate_{k}"]
                rows.append(row)
                print(
                    json.dumps(
                        {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
                    ),
                    flush=True,
                )
                del model
                with OUT.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)


if __name__ == "__main__":
    main()
