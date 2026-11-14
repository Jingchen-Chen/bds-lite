"""Qualitative ACDC figure (official test, D vs D+KD-band), selected_cases.csv and fov_check.csv.

    .venv/bin/python analysis/qualitative_acdc/make_qualitative_figure.py \
        --config analysis/qualitative_acdc/config.yaml [--preview] [--palette A|B|both]

Reads only: processed NPZ (image, mask), saved predictions, raw NIfTI headers (geometry).
Writes: figures/acdc_qualitative/*.png (and *.pdf for the final run), tables/acdc_qualitative/*.csv.
Display only: panels are shown in physical millimetres with nearest-neighbour rendering; no
prediction, label or intensity value is changed. Dice values are recomputed from GT and predictions.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, no Type 3 fonts
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

GRID = 224
CLASSES = {1: "RV", 2: "MYO", 3: "LV"}
MM_PER_INCH = 25.4


# ---------------------------------------------------------------- metrics (registered convention)
def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    s = pred.sum() + gt.sum()
    return np.nan if s == 0 else 2.0 * np.logical_and(pred, gt).sum() / s


def slice_dice(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RV/MYO/LV Dice on one slice; a class absent from both is NaN (registered convention)."""
    return np.array([dice(pred == c, gt == c) for c in (1, 2, 3)])


def run_value(slice_vals: np.ndarray, patients: list[str]) -> float:
    """Registered run value: mean over slices within each patient, then mean over patients."""
    by_pat: dict[str, list[float]] = {}
    for v, p in zip(slice_vals, patients):
        if np.isfinite(v):
            by_pat.setdefault(p, []).append(v)
    return float(np.mean([np.mean(v) for v in by_pat.values()]))


# ---------------------------------------------------------------- data access
def load_npz_index(npz_dir: Path):
    files = sorted(npz_dir.glob("*.npz"))
    ids, masks = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        ids.append(str(d["sample_id"]))
        masks.append(d["mask"])
    return ids, np.stack(masks)


def load_preds(eval_root: Path, arm: str, seed: int, n: int) -> tuple[list[str], np.ndarray]:
    d = np.load(eval_root / f"acdc_confirm_{arm}_seed{seed}" / "preds_official_test.npz", allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    pred = d["predictions"]
    assert pred.shape == (n, GRID, GRID), pred.shape
    return ids, pred


def parse_id(sample_id: str) -> tuple[str, str, int]:
    pid, frame, _, z = sample_id.split("_")
    return pid, frame, int(z)


def mm_per_pixel(raw_root: Path, pid: str, frame: str) -> tuple[float, float]:
    """Effective spacing on the 224 grid, per axis: (mm per row, mm per column).

    The NPZ image is volume[:, :, z].T resized to 224 x 224: rows follow nibabel axis 1 (Y),
    columns follow axis 0 (X). Effective spacing = original spacing x original size / 224.
    """
    img = nib.load(str(raw_root / pid / f"{pid}_{frame}.nii.gz"))
    x, y = img.shape[0], img.shape[1]
    sx, sy = float(img.header.get_zooms()[0]), float(img.header.get_zooms()[1])
    return y * sy / GRID, x * sx / GRID


def phase_of(raw_root: Path, pid: str, frame: str) -> str:
    info = dict(l.split(":", 1) for l in (raw_root / pid / "Info.cfg").read_text().splitlines() if ":" in l)
    return "ED" if int(info["ED"]) == int(frame[5:]) else "ES"


# ---------------------------------------------------------------- selection rules
def select_seed(deltas: dict[int, float]) -> int:
    med = np.median(list(deltas.values()))
    return min(sorted(deltas), key=lambda s: (abs(deltas[s] - med), s))


def select_cases(ids, masks, delta_slice, percentiles):
    present = np.array([all((masks[i] == c).any() for c in (1, 2, 3)) for i in range(len(ids))])
    elig = np.where(present)[0]
    chosen, used = [], set()
    for p in percentiles:
        target = np.percentile(delta_slice[elig], p)
        for i in sorted(elig, key=lambda i: (abs(delta_slice[i] - target), i)):
            pid = parse_id(ids[i])[0]
            if pid in used:
                continue
            chosen.append((p, int(i), float(target)))
            used.add(pid)
            break
    return chosen, len(elig), elig


# ---------------------------------------------------------------- physical field of view
def centroid(gt: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(gt > 0)
    return float(ys.mean()), float(xs.mean())


def fov_needed_mm(gt: np.ndarray, dy: float, dx: float, margin_mm: float) -> float:
    """Smallest square field (mm) centred on the GT centroid that holds all foreground plus a margin."""
    cy, cx = centroid(gt)
    ys, xs = np.where(gt > 0)
    half_y = np.max(np.abs(ys - cy) + 0.5) * dy
    half_x = np.max(np.abs(xs - cx) + 0.5) * dx
    return 2 * (max(half_y, half_x) + margin_mm)


def window(gt: np.ndarray, dy: float, dx: float, fov: float):
    """Top-left pixel and size (rows, cols) of the fov x fov mm window centred on the GT centroid."""
    cy, cx = centroid(gt)
    nr, nc = int(round(fov / dy)), int(round(fov / dx))
    return int(round(cy - nr / 2)), int(round(cx - nc / 2)), nr, nc


def crop_padded(arr: np.ndarray, top: int, left: int, nr: int, nc: int, fill):
    """Crop with fill outside the array. Returns (crop, valid mask of the in-image part)."""
    out = np.full((nr, nc), fill, dtype=arr.dtype)
    valid = np.zeros((nr, nc), dtype=bool)
    r0, c0 = max(top, 0), max(left, 0)
    r1, c1 = min(top + nr, arr.shape[0]), min(left + nc, arr.shape[1])
    out[r0 - top : r1 - top, c0 - left : c1 - left] = arr[r0:r1, c0:c1]
    valid[r0 - top : r1 - top, c0 - left : c1 - left] = True
    return out, valid


# ---------------------------------------------------------------- colour helpers
def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


MACHADO = {  # Machado et al. (2009), severity 1.0, applied to linear RGB
    "protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}


def to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(rgb: np.ndarray, kind: str) -> np.ndarray:
    if kind == "normal":
        return rgb
    if kind == "grayscale":
        lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
        return np.repeat(lum[..., None], 3, axis=2)
    return to_srgb(to_linear(rgb) @ MACHADO[kind].T)


def lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-1) to CIE L*a*b* (D65)."""
    lin = to_linear(np.asarray(rgb, float))
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = m @ lin / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def min_delta_e(colours: dict[str, np.ndarray], kind: str) -> tuple[float, str]:
    """Smallest pairwise CIE76 distance between the given fills after simulating a vision type."""
    names = list(colours)
    sim = {k: simulate(colours[k][None, None, :], kind)[0, 0] for k in names}
    best, pair = np.inf, ""
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            dist = float(np.linalg.norm(lab(sim[a]) - lab(sim[b])))
            if dist < best:
                best, pair = dist, f"{a}/{b}"
    return best, pair


# ---------------------------------------------------------------- rendering
def make_row_panels(row: dict, pal: dict, d: dict):
    """The five panels of one row, all on the same physical field and intensity window."""
    top, left, nr, nc = row["window"]
    gt_c, valid = crop_padded(row["gt"], top, left, nr, nc, 0)
    d_c, _ = crop_padded(row["pD"], top, left, nr, nc, 0)
    k_c, _ = crop_padded(row["pK"], top, left, nr, nc, 0)
    img_c, _ = crop_padded(row["img"], top, left, nr, nc, 0.0)
    lo, hi = np.percentile(img_c[valid], d["intensity_percentiles"])
    gray = np.clip((img_c - lo) / (hi - lo + 1e-8), 0, 1) * valid  # outside the image: black
    colours = {c: hex_rgb(pal[n]) for c, n in CLASSES.items()}

    def overlay(labels):
        rgb = np.repeat(gray[..., None], 3, axis=2)
        for c, col in colours.items():
            m = labels == c
            a = d["overlay_alpha_per_class"].get(CLASSES[c], d["overlay_alpha"])
            rgb[m] = (1 - a) * rgb[m] + a * col
        for c, col in colours.items():
            rgb[find_boundaries(labels == c, mode="inner")] = col
        return np.clip(rgb, 0, 1)

    # Difference: label-level correctness versus GT, background included.
    d_wrong, k_wrong = d_c != gt_c, k_c != gt_c
    diff = np.repeat((gray * d["diff_background_gain"])[..., None], 3, axis=2)  # difference column only
    diff[d_wrong & ~k_wrong] = hex_rgb(pal["corrected"])
    diff[~d_wrong & k_wrong] = hex_rgb(pal["new_error"])
    both = d_wrong & k_wrong
    diff[both] = (1 - d["both_wrong_alpha"]) * diff[both] + d["both_wrong_alpha"] * hex_rgb(pal["both_wrong"])
    diff[~valid] = 0.0
    return [np.repeat(gray[..., None], 3, axis=2), overlay(gt_c), overlay(d_c), overlay(k_c), np.clip(diff, 0, 1)]


def render(rows, cfg, pal, out_stem: Path | None, preview: bool):
    d = cfg["display"]
    fov = float(d["fov_mm"])
    fs = d["font_size_pt"]
    cols = ["Image", "GT", cfg["arms"][0]["display"], cfg["arms"][1]["display"], "Change vs D"]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": d["font_family"], "font.size": fs})
    w_in = d["width_mm"] / MM_PER_INCH
    fig, axes = plt.subplots(len(rows), 5, figsize=(w_in, w_in / 5 * len(rows) + 1.0))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.14, top=0.93, wspace=0.03, hspace=0.05)
    for r, row in enumerate(rows):
        panels = make_row_panels(row, pal, d)
        if not preview:  # final PDF/PNG: nearest-neighbour integer upscaling so viewers do not smooth masks
            k = d["pdf_upscale"]
            panels = [np.repeat(np.repeat(pn, k, axis=0), k, axis=1) for pn in panels]
        nr, nc = row["window"][2], row["window"][3]
        ext = (0, nc * row["dx"], nr * row["dy"], 0)
        for c, pan in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(pan, extent=ext, interpolation=("none" if not preview else "nearest"))
            ax.set_xlim(0, fov)
            ax.set_ylim(fov, 0)
            ax.set_box_aspect(1)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=fs)
            if c in (2, 3):
                ax.text(0.03, 0.04, f"DSC {row['dice'][c - 2]:.3f}", transform=ax.transAxes, color="white",
                        fontsize=fs - 0.5, va="bottom", bbox=dict(facecolor="black", alpha=0.55, pad=1.2, lw=0))
        axes[r, 0].text(0.03, 0.97, row["label"], transform=axes[r, 0].transAxes, color="white",
                        fontsize=fs - 0.5, va="top", bbox=dict(facecolor="black", alpha=0.55, pad=1.2, lw=0))
        if row.get("delta_marker"):
            axes[r, 4].text(0.97, 0.97, row["delta_marker"], transform=axes[r, 4].transAxes, color="white",
                            fontsize=fs - 0.5, va="top", ha="right",
                            bbox=dict(facecolor="black", alpha=0.55, pad=1.2, lw=0))
        if r == len(rows) - 1:  # one scale bar serves all rows (identical mm scale)
            b = d["scale_bar_mm"]
            x1, y0 = fov - 4, fov - 5
            axes[r, 4].plot([x1 - b, x1], [y0, y0], color="white", lw=1.4, solid_capstyle="butt")
            axes[r, 4].text(x1, y0 - 2.5, f"{b} mm", color="white", fontsize=fs - 0.5, ha="right", va="bottom")
    seg = [Patch(facecolor=hex_rgb(pal[n]), label=CLASSES[c]) for c, n in CLASSES.items()]
    chg = [
        Patch(facecolor=hex_rgb(pal["corrected"]), label="Corrected (D wrong, D+KD-band right)"),
        Patch(facecolor=hex_rgb(pal["new_error"]), label="New error (D right, D+KD-band wrong)"),
        Patch(facecolor=hex_rgb(pal["both_wrong"]), label="Both wrong"),
    ]
    leg_kw = dict(frameon=False, fontsize=fs - 0.5, handlelength=1.2, borderaxespad=0)
    fig.legend(handles=seg, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=1, title="Segmentation", title_fontsize=fs - 0.5, **leg_kw)
    fig.legend(handles=chg, loc="lower right", bbox_to_anchor=(0.99, 0.0), ncol=1, title="Change vs D", title_fontsize=fs - 0.5, **leg_kw)
    if out_stem is not None:
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_stem.with_suffix(".png"), dpi=d["preview_dpi"] if preview else d["final_png_dpi"])
        if not preview:
            fig.savefig(out_stem.with_suffix(".pdf"), dpi=d["pdf_embed_dpi"])  # embedded panel = upscaled array size
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0
    plt.close(fig)
    return arr


def comparison_sheet(arrays: dict[str, np.ndarray], out: Path, fs: float) -> None:
    kinds = ["normal", "protanopia", "deuteranopia", "tritanopia", "grayscale"]
    arr0 = next(iter(arrays.values()))
    cell_w = 3.0
    cell_h = cell_w * arr0.shape[0] / arr0.shape[1]
    fig, axes = plt.subplots(len(arrays), len(kinds), figsize=(cell_w * len(kinds) + 0.5, (cell_h + 0.35) * len(arrays)))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.01, wspace=0.03, hspace=0.12)
    for r, (name, arr) in enumerate(arrays.items()):
        for c, kind in enumerate(kinds):
            ax = axes[r, c]
            ax.imshow(simulate(arr, kind), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(kind, fontsize=fs + 2)
            if c == 0:
                ax.set_ylabel(f"Palette {name}", fontsize=fs + 2)
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--palette", default="both", choices=["A", "B", "both"])
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    cfg = yaml.safe_load(Path(args.config).read_text())
    P = {k: Path(v) for k, v in cfg["paths"].items()}
    d = cfg["display"]
    ids, masks = load_npz_index(P["npz_dir"])
    n = len(ids)
    print(f"NPZ samples: {n}")
    arms = [a["key"] for a in cfg["arms"]]
    seeds = cfg["seeds"]
    slice_d = {}
    for arm in arms:
        for s in seeds:
            pids, pred = load_preds(P["eval_root"], arm, s, n)
            assert pids == ids, f"{arm} seed {s}: sample order differs from NPZ"
            slice_d[(arm, s)] = np.stack([slice_dice(pred[i], masks[i]) for i in range(n)])
    patients = [parse_id(s)[0] for s in ids]
    run_val = {(a, s): run_value(np.nanmean(slice_d[(a, s)], axis=1), patients) for a in arms for s in seeds}
    seed_delta = {s: run_val[("kd_band", s)] - run_val[("decoder", s)] for s in seeds}
    seed = select_seed(seed_delta)
    assert seed == cfg["selection"]["expected_seed"], f"seed rule gives {seed}"
    print(f"paired seed: {seed} (delta {seed_delta[seed]:+.5f}; median {np.median(list(seed_delta.values())):+.5f})")

    delta_slice = np.nanmean(slice_d[("kd_band", seed)], axis=1) - np.nanmean(slice_d[("decoder", seed)], axis=1)
    chosen, n_elig, elig = select_cases(ids, masks, delta_slice, cfg["selection"]["case_percentiles"])
    print(f"eligible slices (GT has RV, MYO, LV): {n_elig}")
    assert [ids[i] for _, i, _ in chosen] == cfg["selection"]["expected_cases"]

    _, pD = load_preds(P["eval_root"], "decoder", seed, n)
    _, pK = load_preds(P["eval_root"], "kd_band", seed, n)
    P["out_tables"].mkdir(parents=True, exist_ok=True)

    # Field-of-view check: heart plus a 5 mm margin on each side, for every eligible slice.
    fov_rows = []
    for i in elig:
        pid, frame, _ = parse_id(ids[i])
        dy, dx = mm_per_pixel(P["raw_root"], pid, frame)
        fov_rows.append(dict(sample_id=ids[i], fov_needed_mm=round(fov_needed_mm(masks[i], dy, dx, 5.0), 2)))
    need = np.array([r["fov_needed_mm"] for r in fov_rows])
    print(f"FOV needed, {len(need)} eligible slices (heart + 5 mm each side): median {np.median(need):.1f} mm, "
          f"95th pct {np.percentile(need, 95):.1f} mm, max {need.max():.1f} mm")
    with (P["out_tables"] / "fov_check.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "fov_needed_mm"])
        w.writeheader()
        w.writerows(fov_rows)

    rows, table = [], []
    for p, i, target in chosen:
        pid, frame, z = parse_id(ids[i])
        phase = phase_of(P["raw_root"], pid, frame)
        dy, dx = mm_per_pixel(P["raw_root"], pid, frame)
        dD = slice_dice(pD[i], masks[i])
        dK = slice_dice(pK[i], masks[i])
        dice_D, dice_K = float(np.nanmean(dD)), float(np.nanmean(dK))
        rows.append(dict(
            img=np.load(P["npz_dir"] / f"{ids[i]}.npz")["image"][0], gt=masks[i], pD=pD[i], pK=pK[i],
            window=window(masks[i], dy, dx, float(d["fov_mm"])), dy=dy, dx=dx,
            dice=[float(np.nanmean(dD)), float(np.nanmean(dK))],
            label=f"{phase} {pid.replace('patient', 'p')} s{z}",
            delta_marker=f"ΔDSC {delta_slice[i]:+.3f}".replace("-", "−"),
        ))
        table.append(dict(
            row_percentile=p, sample_id=ids[i], patient=pid, frame=frame, phase=phase, slice=z, seed=seed,
            delta_target_at_percentile=round(target, 5), delta_mean_dice=round(float(delta_slice[i]), 5),
            D_mean_dice=round(float(np.nanmean(dD)), 6), D_RV=round(float(dD[0]), 6), D_MYO=round(float(dD[1]), 6), D_LV=round(float(dD[2]), 6),
            KD_mean_dice=round(float(np.nanmean(dK)), 6), KD_RV=round(float(dK[0]), 6), KD_MYO=round(float(dK[1]), 6), KD_LV=round(float(dK[2]), 6),
            mm_per_row=round(dy, 4), mm_per_col=round(dx, 4),
            figure_label_D=f"{dice_D:.3f}", figure_label_KD=f"{dice_K:.3f}", figure_label_delta=f"{delta_slice[i]:+.3f}",
        ))
        print(ids[i], phase, f"D {np.nanmean(dD):.4f}", f"KD {np.nanmean(dK):.4f}", f"dy {dy:.3f} dx {dx:.3f} mm")
    with (P["out_tables"] / "selected_cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    # Dimmed-MRI background levels in the Change vs D column (display luminance, valid field, all rows).
    lum = []
    for row in rows:
        top, left, nr, nc = row["window"]
        _, valid = crop_padded(row["gt"], top, left, nr, nc, 0)
        img_c, _ = crop_padded(row["img"], top, left, nr, nc, 0.0)
        lo, hi = np.percentile(img_c[valid], d["intensity_percentiles"])
        g = np.clip((img_c - lo) / (hi - lo + 1e-8), 0, 1)[valid] * d["diff_background_gain"]
        lum.append(g)
    lum = np.concatenate(lum)
    bg_levels = {f"MRI {q}th pct": float(np.percentile(lum, q)) for q in (5, 50, 95)}
    print("dimmed MRI luminance levels (Change vs D):", {k: round(v, 3) for k, v in bg_levels.items()})

    palettes = ["A", "B"] if args.palette == "both" else [args.palette]
    arrays = {}
    for name in palettes:
        pal = d["palettes"][name]
        seg = {k: hex_rgb(pal[k]) for k in ("RV", "MYO", "LV")}
        chg = {k: hex_rgb(pal[k]) for k in ("corrected", "new_error", "both_wrong")}
        labels = {"corrected": "Corrected", "new_error": "New error", "both_wrong": "Both wrong"}
        for kind in ["normal", "protanopia", "deuteranopia", "tritanopia"]:
            seg_min, seg_pair = min_delta_e(seg, kind)
            chg_min, chg_pair = min_delta_e(chg, kind)
            # each change colour against each dimmed-MRI level, under the same vision type
            vs_bg = []
            for k, col in chg.items():
                sim_c = simulate(col[None, None, :], kind)[0, 0]
                for bn, bl in bg_levels.items():
                    sim_b = simulate(np.full((1, 1, 3), bl), kind)[0, 0]
                    vs_bg.append((float(np.linalg.norm(lab(sim_c) - lab(sim_b))), f"{labels[k]}/{bn}"))
            bg_min, bg_pair = min(vs_bg)
            print(f"palette {name} {kind:12s} segmentation min dE76 {seg_min:5.1f} ({seg_pair}) | "
                  f"change min dE76 {chg_min:5.1f} ({chg_pair}) | change vs MRI min {bg_min:5.1f} ({bg_pair})")
        seg_g, _ = min_delta_e(seg, "grayscale")
        chg_g, _ = min_delta_e(chg, "grayscale")
        print(f"palette {name} grayscale (reference only): segmentation min dE76 {seg_g:5.1f}, change min dE76 {chg_g:5.1f}")
        stem = P["out_figures"] / f"acdc_qualitative_palette{name}"
        arrays[name] = render(rows, cfg, pal, stem, preview=args.preview)
        print("written:", stem.with_suffix(".png"), "(preview)" if args.preview else "(final, PDF+PNG)")
    if len(arrays) == 2:
        comparison_sheet(arrays, P["out_figures"] / "palette_comparison.png", d["font_size_pt"])
        print("written:", P["out_figures"] / "palette_comparison.png")


if __name__ == "__main__":
    sys.exit(main())
