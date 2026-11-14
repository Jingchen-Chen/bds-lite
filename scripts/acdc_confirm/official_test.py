"""Official ACDC test patients 101-150: integrity check and preprocessing. No model is run.

    .venv/bin/python scripts/acdc_confirm/official_test.py verify
    .venv/bin/python scripts/acdc_confirm/official_test.py preprocess

Source: original NIfTI files from the official CREATIS site, placed by the author at
/home/jc/bds-followup/data/raw/acdc_official_test/ (no third-party preprocessed copy).

Preprocessing reproduces the pipeline that produced data/processed/acdc:
- Stage 1 (third-party NIfTI -> pre-sliced H5; not in this repository; reconstructed and validated
  on the 80 local development patients, see `dev_gate`): per volume, clip above the 99th
  percentile (numpy linear), z-score with the clipped volume's mean and SD (ddof 0), and take each
  slice in SimpleITK axis order (the transpose of nibabel's [x, y]).
- Stage 2 (frozen, src/bds_lite/data/converters.py::convert_acdc_transunet): `_resize_gray`
  (PIL bilinear) and `_resize_mask` (PIL nearest) to 224 x 224; NPZ with image (1, 224, 224)
  float32, mask (224, 224) uint8, sample_id `patientNNN_frameFF_slice_Z`.
Every slice of the ED and ES volumes is kept, as in data/processed/acdc.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from bds_lite.data.converters import _resize_gray, _resize_mask  # noqa: E402
from scripts.acdc_confirm.common import (  # noqa: E402
    OFFICIAL_MANIFEST,
    OFFICIAL_PROCESSED,
    OFFICIAL_RAW,
    WORKSPACE,
)
from scripts.kd.common import sha  # noqa: E402

DEV_RAW = Path("/home/jc/bds-followup/data/raw/acdc_development_20261018")
DEV_PROCESSED = ROOT / "data/processed/acdc"
DEV_MANIFESTS = ROOT / "recovered/followup/splits/acdc"
PATIENTS = [f"patient{n:03d}" for n in range(101, 151)]
LABELS = {0, 1, 2, 3}  # 0 background, 1 RV, 2 myocardium, 3 LV (repository convention)
DEV_GATE_TOLERANCE = 0.015  # observed maximum 0.01478 z-units; masks must be exact
INTEGRITY = WORKSPACE / "official_test_integrity.json"
INTEGRITY_CSV = WORKSPACE / "official_test_volumes.csv"
DEV_GATE = WORKSPACE / "preprocessing_dev_gate.json"
PREPROCESS = WORKSPACE / "official_test_preprocessing.json"


def nib():
    import nibabel

    return nibabel


def stage1(volume: np.ndarray) -> np.ndarray:
    v = volume.astype(np.float32)
    clipped = np.clip(v, None, np.percentile(v, 99))
    return ((clipped - clipped.mean()) / clipped.std()).astype(np.float32)


def slices(image_path: Path, gt_path: Path):
    """Yield (z, image (1,224,224) float32, mask (224,224) uint8) for every slice of one volume."""
    volume = stage1(nib().load(str(image_path)).get_fdata(dtype=np.float32))
    labels = np.asarray(nib().load(str(gt_path)).dataobj).astype(np.uint8)
    for z in range(volume.shape[2]):
        image = _resize_gray(np.ascontiguousarray(volume[:, :, z].T), (224, 224))
        mask = _resize_mask(np.ascontiguousarray(labels[:, :, z].T), (224, 224))
        yield z, image[None, ...], mask


def info(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def patient_dirs(root: Path) -> dict[str, Path]:
    found = {}
    for depth in ("*", "*/*", "*/*/*"):
        for path in root.glob(depth):
            if path.is_dir() and re.fullmatch(r"patient\d{3}", path.name):
                if path.name in found:
                    raise ValueError(f"duplicate patient folder {path.name}")
                found[path.name] = path
        if found:
            break
    return found


def lv_enclosure(labels: np.ndarray) -> float | None:
    """Fraction of LV-boundary pixels whose 4-neighbours outside the LV are myocardium."""
    from scipy import ndimage as ndi

    touching = total = 0
    for z in range(labels.shape[2]):
        lab = labels[:, :, z]
        lv = lab == 3
        if lv.sum() == 0:
            continue
        ring = ndi.binary_dilation(lv) & ~lv
        touching += int((lab[ring] == 2).sum())
        total += int(ring.sum())
    return touching / total if total else None


def verify(root: Path = OFFICIAL_RAW) -> dict:
    problems, rows = [], []
    dirs = patient_dirs(root) if root.exists() else {}
    if sorted(dirs) != PATIENTS:
        problems.append(
            dict(
                missing=sorted(set(PATIENTS) - set(dirs)),
                unexpected=sorted(set(dirs) - set(PATIENTS)),
            )
        )
    files = {}
    for pid in sorted(set(dirs) & set(PATIENTS)):
        folder = dirs[pid]
        for path in sorted(folder.iterdir()):
            files[str(path.relative_to(root))] = sha(path)
        if not (folder / "Info.cfg").is_file():
            problems.append(f"{pid}: Info.cfg missing")
            continue
        cfg = info(folder / "Info.cfg")
        gts = sorted(p.name for p in folder.glob("*_gt.nii.gz"))
        frames = {}
        for phase in ("ED", "ES"):
            try:
                frames[phase] = int(cfg[phase])
            except (KeyError, ValueError):
                problems.append(f"{pid}: Info.cfg lacks {phase}")
        expected = sorted({f"{pid}_frame{f:02d}_gt.nii.gz" for f in frames.values()})
        if gts != expected:
            problems.append(f"{pid}: ground-truth files {gts} != ED/ES {expected}")
        for phase, frame in frames.items():
            image = folder / f"{pid}_frame{frame:02d}.nii.gz"
            gt = folder / f"{pid}_frame{frame:02d}_gt.nii.gz"
            if not image.is_file() or not gt.is_file():
                problems.append(f"{pid} {phase}: image or ground truth missing")
                continue
            ni, ng = nib().load(str(image)), nib().load(str(gt))
            data = ni.get_fdata(dtype=np.float32)
            labels = np.asarray(ng.dataobj)
            values = sorted(int(v) for v in np.unique(labels))
            row = dict(
                patient=pid,
                phase=phase,
                frame=frame,
                group=cfg.get("Group", ""),
                shape="x".join(map(str, data.shape)),
                slices=data.shape[2] if data.ndim == 3 else None,
                spacing="x".join(f"{float(s):.4g}" for s in ni.header.get_zooms()),
                image_dtype=str(ni.get_data_dtype()),
                gt_dtype=str(ng.get_data_dtype()),
                label_values=" ".join(map(str, values)),
                lv_enclosed_by_myo=lv_enclosure(labels) if labels.ndim == 3 else None,
            )
            rows.append(row)
            if data.ndim != 3 or labels.shape != data.shape:
                problems.append(f"{pid} {phase}: shape {data.shape} vs gt {labels.shape}")
            if not np.isfinite(data).all():
                problems.append(f"{pid} {phase}: non-finite image values")
            if not set(values) <= LABELS:
                problems.append(f"{pid} {phase}: label values {values}")
            if not np.array_equal(labels, np.round(labels)):
                problems.append(f"{pid} {phase}: non-integer labels")
            if not np.allclose(ni.affine, ng.affine, atol=1e-4):
                problems.append(f"{pid} {phase}: image/gt affine differ")
    dev = [lv_enclosure_dev(p) for p in sorted(DEV_RAW.glob("patient*"))[:10]]
    enclosure = [r["lv_enclosed_by_myo"] for r in rows if r["lv_enclosed_by_myo"] is not None]
    groups = sorted({r["group"] for r in rows})
    result = dict(
        root=str(root),
        passed=not problems,
        problems=problems,
        patients=len(set(dirs) & set(PATIENTS)),
        volumes=len(rows),
        slices=sum(r["slices"] or 0 for r in rows),
        groups={g: sum(1 for r in rows if r["group"] == g) // 2 for g in groups},
        lv_enclosed_by_myo=dict(
            official_min=min(enclosure, default=None),
            official_mean=float(np.mean(enclosure)) if enclosure else None,
            development_first10_mean=float(np.mean([d for d in dev if d is not None])),
        ),
        file_sha256=files,
    )
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    INTEGRITY.write_text(json.dumps(result, indent=2) + "\n")
    if rows:
        with INTEGRITY_CSV.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return result


def lv_enclosure_dev(folder: Path) -> float | None:
    cfg = info(folder / "Info.cfg")
    gt = folder / f"{folder.name}_frame{int(cfg['ED']):02d}_gt.nii.gz"
    return lv_enclosure(np.asarray(nib().load(str(gt)).dataobj))


def dev_gate() -> dict:
    """Reproduce every stored train/validation slice of data/processed/acdc from native NIfTI."""
    worst, mask_pixels, count, mismatched_counts = 0.0, 0, 0, []
    for split in ("train", "val"):
        ids = json.loads((DEV_MANIFESTS / f"{split}.json").read_text())["sample_ids"]
        for frame in sorted({re.match(r"(patient\d+_frame\d+)", s)[1] for s in ids}):
            pid = frame.split("_")[0]
            stored = sorted(DEV_PROCESSED.glob(f"{split}/{frame}_slice_*.npz"))
            produced = list(
                slices(DEV_RAW / pid / f"{frame}.nii.gz", DEV_RAW / pid / f"{frame}_gt.nii.gz")
            )
            if len(stored) != len(produced):
                mismatched_counts.append(frame)
            for z, image, mask in produced:
                ref = np.load(DEV_PROCESSED / split / f"{frame}_slice_{z}.npz")
                worst = max(worst, float(np.abs(image - ref["image"]).max()))
                mask_pixels += int((mask != ref["mask"]).sum())
                count += 1
    result = dict(
        slices=count,
        image_max_abs_diff=worst,
        tolerance=DEV_GATE_TOLERANCE,
        mask_mismatched_pixels=mask_pixels,
        slice_count_mismatches=mismatched_counts,
        passed=(
            count == 1536
            and mask_pixels == 0
            and not mismatched_counts
            and worst <= DEV_GATE_TOLERANCE
        ),
    )
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    DEV_GATE.write_text(json.dumps(result, indent=2) + "\n")
    return result


def preprocess() -> dict:
    record = json.loads(INTEGRITY.read_text()) if INTEGRITY.exists() else None
    if not record or not record["passed"]:
        raise RuntimeError("official test integrity check has not passed")
    for name, digest in record["file_sha256"].items():
        if sha(OFFICIAL_RAW / name) != digest:
            raise ValueError(f"raw file changed after the integrity check: {name}")
    gate = dev_gate()
    if not gate["passed"]:
        raise RuntimeError(f"preprocessing does not reproduce the development data: {gate}")
    out = OFFICIAL_PROCESSED / "test"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    dirs = patient_dirs(OFFICIAL_RAW)
    ids, digests, per_volume = [], {}, {}
    for pid in PATIENTS:
        cfg = info(dirs[pid] / "Info.cfg")
        for phase in ("ED", "ES"):
            frame = f"{pid}_frame{int(cfg[phase]):02d}"
            n = 0
            for z, image, mask in slices(
                dirs[pid] / f"{frame}.nii.gz", dirs[pid] / f"{frame}_gt.nii.gz"
            ):
                sample_id = f"{frame}_slice_{z}"
                path = out / f"{sample_id}.npz"
                np.savez_compressed(path, image=image, mask=mask, sample_id=np.asarray(sample_id))
                ids.append(sample_id)
                digests[sample_id] = sha(path)
                n += 1
            per_volume[frame] = n
    ids = sorted(ids)
    manifest = dict(
        dataset="acdc",
        split="test",
        seed=2026,  # required by the frozen ManifestDataset; the official split has no seed
        cohort="ACDC official test patients 101-150 (CREATIS NIfTI)",
        count=len(ids),
        sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        sha256_definition="sha256 of the sorted sample_ids joined by newlines",
        sample_ids=ids,
    )
    OFFICIAL_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    result = dict(
        slices=len(ids),
        volumes=len(per_volume),
        patients=len(PATIENTS),
        per_volume_slices=per_volume,
        manifest=str(OFFICIAL_MANIFEST.relative_to(ROOT)),
        manifest_file_sha256=sha(OFFICIAL_MANIFEST),
        npz_sha256=digests,
        dev_gate=gate,
        model_run=False,
    )
    PREPROCESS.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["verify", "preprocess", "dev-gate"])
    step = parser.parse_args().step
    result = dict(verify=verify, preprocess=preprocess, **{"dev-gate": dev_gate})[step]()
    shown = {k: v for k, v in result.items() if k not in {"file_sha256", "npz_sha256"}}
    print(json.dumps(shown, indent=2, default=str)[:4000])
