# ruff: noqa: E501
"""ISIC2018 official test: near-duplicate lesions of the development pool and sensitivity analysis.

ISIC images of the same lesion can appear in both the development pool and the official test
set under different IDs. For every test image we compute the maximum Pearson correlation with
every development image over the 8 dihedral transforms, using 32x32 thumbnails of each RGB
channel standardised separately (robust to global colour shifts). The reference-mask agreement
(Dice of the 32x32 masks under the best transform) is recorded as independent evidence.
No threshold separates duplicates cleanly (visual check: same-lesion pairs at r=0.956-0.957
and different lesions at r=0.953-0.956; mask agreement does not separate them either), so the
sensitivity analysis is bracketed: exclusion of test images with r > 0.99 (certain duplicates),
0.96, 0.93 and 0.90 (deliberately over-inclusive). The largest-gap threshold and a contact sheet
around it are still saved for transparency.
Sensitivity: family P (S6 vs S0) and P-replication on the official test excluding flagged images.
Outputs: revision/validation/isic_test_duplicates.csv, revision/restricted_local/isic_duplicate_threshold.png (local only: shows photographs),
revision/tables/families/isic_duplicate_sensitivity.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from analysis.revision import family_statistics as fs  # noqa: E402
from analysis.revision.recover_predictions import Cohort  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def features(images: np.ndarray, size: int = 32) -> torch.Tensor:
    x = F.interpolate(torch.from_numpy(images).float(), size=(size, size), mode="area")
    x = x - x.mean(dim=(2, 3), keepdim=True)
    x = x / (x.flatten(2).std(-1)[..., None, None] + 1e-6)
    return x


def dihedral(x: torch.Tensor) -> list[torch.Tensor]:
    out = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(-2, -1))
        out += [r, torch.flip(r, dims=(-1,))]
    return out


def main() -> None:
    cohort = Cohort(dict(dataset="isic2018", split="official_test"))
    test_ids = sorted(cohort.ids)
    test = {sid: cohort.get(sid)[:2] for sid in test_ids}
    dev_files = [
        f for f in sorted((ROOT / "data/processed/isic2018").glob("*/*.npz")) if "(1)" not in f.name
    ]
    fd_parts, dev_mask = [], []
    for start in range(0, len(dev_files), 256):
        chunk = [np.load(f) for f in dev_files[start : start + 256]]
        fd_parts.append(features(np.stack([c["image"] for c in chunk])).flatten(1))
        dev_mask.extend(c["mask"] for c in chunk)
    fd = torch.cat(fd_parts)
    dev_mask = np.stack(dev_mask)
    ft = features(np.stack([test[s][0] for s in test_ids]))
    best_r = torch.full((len(test_ids),), -1.0)
    best_j = torch.zeros(len(test_ids), dtype=torch.long)
    best_k = torch.zeros(len(test_ids), dtype=torch.long)
    for k, t in enumerate(dihedral(ft)):
        corr = t.flatten(1) @ fd.T / fd.shape[1]
        r, j = corr.max(1)
        better = r > best_r
        best_r[better], best_j[better], best_k[better] = r[better], j[better], k
    order = np.sort(best_r.numpy())[::-1]
    window = order[(order >= 0.85) & (order <= 0.99)]
    gaps = window[:-1] - window[1:]
    threshold = (
        float((window[np.argmax(gaps)] + window[np.argmax(gaps) + 1]) / 2) if gaps.size else 0.95
    )
    tm = F.interpolate(
        torch.from_numpy(np.stack([test[s][1] for s in test_ids])[:, None]).float(),
        size=(32, 32),
        mode="nearest",
    )
    dm = F.interpolate(torch.from_numpy(dev_mask[:, None]).float(), size=(32, 32), mode="nearest")
    rows = []
    for i, sid in enumerate(test_ids):
        t = dihedral(tm[i : i + 1])[int(best_k[i])][0, 0] > 0.5
        d = dm[int(best_j[i]), 0] > 0.5
        dice = float(2 * (t & d).sum() / max(int(t.sum() + d.sum()), 1))
        rows.append(
            dict(
                test_id=sid,
                dev_id=dev_files[int(best_j[i])].stem,
                transform=int(best_k[i]),
                max_r=float(best_r[i]),
                mask_dice=dice,
                flagged=bool(best_r[i] > threshold),
            )
        )
    path = ROOT / "revision/validation/isic_test_duplicates.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    flagged = {r["test_id"] for r in rows if r["flagged"]}
    near = sorted(rows, key=lambda r: abs(r["max_r"] - threshold))[:8]
    fig, axes = plt.subplots(len(near), 2, figsize=(4, 2 * len(near)))
    for ax, r in zip(axes, near, strict=True):
        ax[0].imshow(test[r["test_id"]][0].transpose(1, 2, 0))
        ax[1].imshow(
            np.load(next((ROOT / "data/processed/isic2018").glob(f"*/{r['dev_id']}.npz")))[
                "image"
            ].transpose(1, 2, 0)
        )
        ax[0].set_title(
            f"{r['test_id']} r={r['max_r']:.3f} {'FLAG' if r['flagged'] else ''}", fontsize=6
        )
        ax[1].set_title(r["dev_id"], fontsize=6)
        for a in ax:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "revision/restricted_local/isic_duplicate_threshold.png", dpi=110)
    # sensitivity
    out = []
    for fam, a, b in [("P", "S6", "S0"), ("P-replication", "R6", "R0")]:
        va, vb = fs.sample_values(a, "isic2018"), fs.sample_values(b, "isic2018")
        for metric in fs.METRICS:
            sign = -1 if metric in fs.DISTANCE else 1
            brackets = [("all", set(va))] + [
                (f"excluding_r_gt_{t}", set(va) - {r["test_id"] for r in rows if r["max_r"] > t})
                for t in (0.99, 0.96, 0.93, 0.90)
            ]
            for subset, keep in brackets:
                d = np.array(
                    [
                        sign * (va[s][metric] - vb[s][metric])
                        for s in sorted(keep)
                        if np.isfinite(va[s][metric]) and np.isfinite(vb[s][metric])
                    ]
                )
                low, high = fs.bootstrap(d)
                out.append(
                    dict(
                        family=fam,
                        metric=metric,
                        subset=subset,
                        n=d.size,
                        mean_oriented_delta=float(d.mean()),
                        ci95_low=low,
                        ci95_high=high,
                        practical=fs.judgement(low, high, fs.SESOI[metric]),
                    )
                )
    with (ROOT / "revision/tables/families/isic_duplicate_sensitivity.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    print(
        f"threshold {threshold:.4f}; flagged {len(flagged)}; top maxima {np.round(order[:5], 4)}; around threshold {np.round(window[np.argmax(gaps) - 2 : np.argmax(gaps) + 4], 4) if gaps.size else ''}"
    )
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
