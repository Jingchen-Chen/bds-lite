# ruff: noqa: E501
"""Independent verification of the official ISIC2018 test evaluation used for family P.

Checks (results -> revision/validation/official_isic_verification.json):
1. Archive integrity: SHA-256 and member counts of the four official archives equal the
   acquisition receipt (revision/sources/isic_acquisition.json).
2. Identity: 1,000 test images and masks with identical IDs; no ID shared with the 2,594-image
   development pool (locked and seed-1 partitions); no near-duplicate content: every test image
   is compared with every development image on 16x16 grey thumbnails (Pearson r); pairs with
   r > 0.98 are listed and inspected at 64x64 (r > 0.995 = duplicate).
3. Preprocessing parity: the official-test loader equals an independent re-implementation of the
   released converter (PIL RGB bilinear 224, /255) for 50 random test images, and the converter
   reproduces stored development arrays for all available raw development JPEGs.
4. Prediction reproducibility: for each submitted checkpoint (S0/S6, seeds 1-3) 64 random test
   images are re-predicted through a separate code path and compared with the stored maps.
5. Metric re-implementation: DSC, IoU, HD95 and ASSD recomputed for all 1,000 images x 6 models
   with an independent implementation (scipy EDT surface distances) and compared with the
   per-case CSVs used by the statistics.
6. Statistics re-implementation: family P on ISIC recomputed with pandas-free numpy code and a
   different bootstrap RNG; compared with revision/tables/families/family_statistics.csv.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import wilcoxon

from analysis.revision.evaluate_extra_cohorts import load
from analysis.revision.recover_predictions import Cohort

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "data/revision/isic2018_official/archives"
RNG = np.random.default_rng(7)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def thumb(arr: np.ndarray, size: int) -> np.ndarray:
    grey = arr.mean(0) if arr.ndim == 3 else arr
    img = Image.fromarray((np.clip(grey, 0, 1) * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    v = np.asarray(img, np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-6)


def surface_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    sa, sb = a ^ ndi.binary_erosion(a), b ^ ndi.binary_erosion(b)
    if not sa.any() or not sb.any():
        return None
    da = ndi.distance_transform_edt(~sb)[sa]
    db = ndi.distance_transform_edt(~sa)[sb]
    return np.concatenate([da, db])


def main() -> None:
    report = {}
    receipt = json.loads((ROOT / "revision/sources/isic_acquisition.json").read_text())
    report["archives"] = [
        dict(
            path=r["path"],
            sha256_ok=sha(ROOT / r["path"]) == r["sha256"],
            members=len(zipfile.ZipFile(ROOT / r["path"]).namelist()),
            expected_members=r["members"],
        )
        for r in receipt
    ]
    cohort = Cohort(dict(dataset="isic2018", split="official_test"))
    test_ids = sorted(cohort.ids)
    with zipfile.ZipFile(ARCH / "ISIC2018_Task1_Test_GroundTruth.zip") as gt:
        gt_ids = sorted(
            Path(n).stem.removesuffix("_segmentation")
            for n in gt.namelist()
            if n.endswith(".png") and not n.startswith("__MACOSX")
        )
    dev = set()
    for name in ["train", "val"]:
        dev |= set(
            json.loads((ROOT / f"recovered/followup/splits/isic2018/{name}.json").read_text())[
                "sample_ids"
            ]
        )
    seed1 = json.loads((ROOT / "splits/isic2018_split.json").read_text())
    dev_seed1 = {x.split("/")[-1] for k in ["train", "val", "test"] for x in seed1[k]}
    report["identity"] = dict(
        n_test=len(test_ids),
        masks_match_images=test_ids == gt_ids,
        dev_pool=len(dev),
        seed1_pool=len(dev_seed1),
        id_overlap_locked=len(set(test_ids) & dev),
        id_overlap_seed1=len(set(test_ids) & dev_seed1),
    )
    # near-duplicates
    dev_files = sorted((ROOT / "data/processed/isic2018").glob("*/*.npz"))
    dev_files = [f for f in dev_files if "(1)" not in f.name]
    dev_t = np.stack([thumb(np.load(f)["image"], 16) for f in dev_files])
    test_arrays = {sid: cohort.get(sid)[0] for sid in test_ids}
    test_t = np.stack([thumb(test_arrays[s], 16) for s in test_ids])
    corr = test_t @ dev_t.T / dev_t.shape[1]
    candidates = np.argwhere(corr > 0.98)
    flagged = []
    for i, j in candidates:
        r64 = float(
            thumb(test_arrays[test_ids[i]], 64) @ thumb(np.load(dev_files[j])["image"], 64) / 4096
        )
        flagged.append(
            dict(test=test_ids[i], dev=dev_files[j].stem, r16=float(corr[i, j]), r64=r64)
        )
    report["near_duplicates"] = dict(
        dev_images_compared=len(dev_files),
        max_r16=float(corr.max()),
        candidates_r16_gt_0_98=flagged,
        duplicates_r64_gt_0_995=[f for f in flagged if f["r64"] > 0.995],
    )
    # preprocessing parity
    with zipfile.ZipFile(ARCH / "ISIC2018_Task1-2_Test_Input.zip") as images:
        names = {
            Path(n).stem: n
            for n in images.namelist()
            if n.lower().endswith(".jpg") and not n.startswith("__MACOSX")
        }
        sample = RNG.choice(test_ids, 50, replace=False)
        diffs = []
        for sid in sample:
            rgb = Image.open(io.BytesIO(images.read(names[sid]))).convert("RGB")
            mine = (
                np.asarray(rgb.resize((224, 224), Image.Resampling.BILINEAR), np.float32) / 255
            ).transpose(2, 0, 1)
            diffs.append(float(np.abs(mine - test_arrays[sid]).max()))
    raw = sorted((ROOT / "data/raw/isic2018/ISIC2018_Task1-2_Training_Input").glob("*.jpg"))
    conv = []
    for p in raw:
        stored = next((ROOT / "data/processed/isic2018").glob(f"*/{p.stem}.npz"), None)
        if stored is None:
            continue
        mine = (
            np.asarray(
                Image.open(p).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
                np.float32,
            )
            / 255
        ).transpose(2, 0, 1)
        conv.append(float(np.abs(mine - np.load(stored)["image"]).max()))
    report["preprocessing"] = dict(
        test_loader_vs_reimplementation_max_abs=max(diffs),
        n_test_checked=len(diffs),
        dev_converter_vs_stored_max_abs=max(conv),
        n_dev_checked=len(conv),
    )
    # predictions + metrics
    torch.set_num_threads(4)
    pred_rows, metric_rows = [], []
    for cohort_name, method in [("famO_unet", "S0"), ("famO_full", "S6")]:
        for seed in (1, 2, 3):
            folder = ROOT / f"revision/extra_cohorts/{cohort_name}/isic2018_seed{seed}/test"
            stored = np.load(folder / "predictions.npz")
            maps = dict(zip([str(i) for i in stored["ids"]], stored["predictions"], strict=True))
            model = load(cohort_name, "isic2018", seed)
            pick = RNG.choice(test_ids, 64, replace=False)
            x = torch.from_numpy(np.stack([test_arrays[s] for s in pick]))
            with torch.inference_mode():
                out = model(x) if method == "S0" else model(x, return_boundary=False)
            new = out["seg_logits"].argmax(1).numpy()
            mism = int(sum((new[k] != maps[s]).sum() for k, s in enumerate(pick)))
            pred_rows.append(dict(method=method, seed=seed, images=64, mismatched_pixels=mism))
            table = {r["id"]: r for r in csv.DictReader((folder / "per_case.csv").open())}
            worst = dict(dsc=0.0, iou=0.0, hd95=0.0, assd=0.0)
            for sid in test_ids:
                p, t = maps[sid] == 1, cohort.get(sid)[1] == 1
                inter, tot, uni = (p & t).sum(), p.sum() + t.sum(), (p | t).sum()
                vals = dict(
                    dsc=(2 * inter + 1e-6) / (tot + 1e-6), iou=(inter + 1e-6) / (uni + 1e-6)
                )
                d = surface_dist(p, t)
                if d is not None:
                    vals.update(hd95=float(np.percentile(d, 95)), assd=float(d.mean()))
                for k, v in vals.items():
                    ref = table[sid][k]
                    if ref not in ("", "None"):
                        worst[k] = max(worst[k], abs(float(v) - float(ref)))
            metric_rows.append(
                dict(method=method, seed=seed, **{f"max_abs_diff_{k}": v for k, v in worst.items()})
            )
            print(pred_rows[-1], metric_rows[-1], flush=True)
    report["predictions"] = pred_rows
    report["metrics"] = metric_rows

    # statistics re-implementation (family P, ISIC)
    def per_image(method_dir: str) -> dict[str, dict[str, float]]:
        out = {}
        tables = [
            {
                r["id"]: r
                for r in csv.DictReader(
                    (
                        ROOT
                        / f"revision/extra_cohorts/{method_dir}/isic2018_seed{s}/test/per_case.csv"
                    ).open()
                )
            }
            for s in (1, 2, 3)
        ]
        for sid in test_ids:
            out[sid] = {}
            for m in ["dsc", "iou", "boundary_f1", "hd95", "assd"]:
                vals = [float(t[sid][m]) for t in tables if t[sid][m] not in ("", "None")]
                out[sid][m] = float(np.mean(vals)) if vals else np.nan
        return out

    u, b = per_image("famO_unet"), per_image("famO_full")
    published = {
        r["metric"]: r
        for r in csv.DictReader((ROOT / "revision/tables/families/family_statistics.csv").open())
        if r["family"] == "P" and r["dataset"] == "isic2018"
    }
    rng = np.random.default_rng(12345)
    stats = []
    for m in ["dsc", "iou", "boundary_f1", "hd95", "assd"]:
        sign = -1 if m in ("hd95", "assd") else 1
        d = np.array([sign * (b[s][m] - u[s][m]) for s in test_ids])
        d = d[np.isfinite(d)]
        boots = d[rng.integers(0, d.size, (20000, d.size))].mean(1)
        stats.append(
            dict(
                metric=m,
                n=int(d.size),
                mean=float(d.mean()),
                ci=[float(x) for x in np.quantile(boots, [0.025, 0.975])],
                p=float(wilcoxon(d).pvalue),
                published_mean=float(published[m]["mean_oriented_delta"]),
                published_ci=[float(published[m]["ci95_low"]), float(published[m]["ci95_high"])],
                published_p=float(published[m]["p_raw"]),
            )
        )
    report["statistics"] = stats
    out = ROOT / "revision/validation/official_isic_verification.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ["identity", "preprocessing"]}, indent=1))
    print(
        json.dumps(report["near_duplicates"]["max_r16"]),
        len(report["near_duplicates"]["candidates_r16_gt_0_98"]),
    )
    for s in stats:
        print(s)


if __name__ == "__main__":
    main()
