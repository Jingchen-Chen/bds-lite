"""Main Figure 3: the two qualitative ACDC figures as one figure with panels a and b.

    PYTHONPATH=src:. .venv/bin/python analysis/revision/figure_qualitative_combined.py

Display only; no model is run and no case is reselected.
  a  U, D, BDS-Lite on the internal test patients (study 1): cases and seed of
     results/qualitative_bdslite/selected_cases.csv, predictions from predictions.npz there
     (written by figure_qualitative_bdslite.py).
  b  D, D+KD-band on the official test patients 101-150: cases and seed of
     results/tables/acdc_qualitative/selected_cases.csv (written by
     paper/figures/make_qualitative_figure.py), predictions from results/acdc_confirm_eval.
Each slice DSC and slice difference is recomputed and checked against those tables before
drawing. Panels, palette B, field of view, intensity window and scale bar come from the panel
code of paper/figures/make_qualitative_figure.py and paper/figures/config.yaml, so both
panels have the same column width and panel size. Panel b has five consecutive columns centred
under the six columns of panel a, with the same spacing between images.
The Segmentation and Change vs D legends are drawn once, at the bottom.
"""

from __future__ import annotations

import csv
import importlib.util
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "paper/figures/acdc_qualitative_combined"
PALETTE = "B"

_spec = importlib.util.spec_from_file_location(
    "make_qualitative_figure", ROOT / "paper/figures/make_qualitative_figure.py"
)
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def mean_dsc(pred: np.ndarray, gt: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(q.slice_dice(pred, gt)))


def make_row(sid, img, gt, preds, raw, fov, delta_check):
    """preds: column name -> prediction; the last two arms are D and the compared arm."""
    pid, frame, z = q.parse_id(sid)
    dy, dx = q.mm_per_pixel(raw, pid, frame)
    dice = {k: mean_dsc(p, gt) for k, p in preds.items()}
    names = list(preds)
    delta = dice[names[-1]] - dice["D"]
    if abs(delta - delta_check) > 6e-6:
        raise ValueError(f"{sid}: slice difference {delta:.6f} differs from table {delta_check}")
    return dict(
        preds=preds,
        img=img,
        gt=gt,
        window=q.window(gt, dy, dx, fov),
        dy=dy,
        dx=dx,
        dice=dice,
        label=f"{q.phase_of(raw, pid, frame)} {pid.replace('patient', 'p')} s{z}",
        delta_marker=f"ΔDSC {delta:+.3f}".replace("-", "−"),
    )


def rows_a(fov: float) -> list[dict]:
    out = ROOT / "results/qualitative_bdslite"
    raw = ROOT / "data/raw/acdc_internal_test_geometry"
    table = read_csv(out / "selected_cases.csv")
    with np.load(out / "predictions.npz") as z:
        ids = [str(x) for x in z["ids"]]
        preds = {k: z[k] for k in z.files if k != "ids"}
    rows = []
    for t in table:
        i, seed = ids.index(t["sample_id"]), int(t["seed"])
        with np.load(ROOT / "data/processed/acdc/test" / f"{t['sample_id']}.npz") as z:
            img, gt = z["image"][0].astype(np.float32), z["mask"].astype(np.uint8)
        arms = {a: preds[f"{a}_seed{seed}"][i] for a in ("U", "D", "BDS-Lite")}
        rows.append(
            make_row(t["sample_id"], img, gt, arms, raw, fov, float(t["delta_dsc_bdslite_minus_d"]))
        )
    return rows


def rows_b(cfg: dict, fov: float) -> list[dict]:
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    table = read_csv(ROOT / "results/tables/acdc_qualitative/selected_cases.csv")
    if [t["sample_id"] for t in table] != cfg["selection"]["expected_cases"]:
        raise ValueError("panel b cases differ from config.yaml")
    rows = []
    for t in table:
        seed, sid = int(t["seed"]), t["sample_id"]
        if seed != cfg["selection"]["expected_seed"]:
            raise ValueError("panel b seed differs from config.yaml")
        with np.load(paths["npz_dir"] / f"{sid}.npz", allow_pickle=True) as z:
            img, gt = z["image"][0], z["mask"]
        arms = {}
        for key, name in (("decoder", "D"), ("kd_band", "D+KD-band")):
            path = paths["eval_root"] / f"acdc_confirm_{key}_seed{seed}" / "preds_official_test.npz"
            with np.load(path, allow_pickle=True) as z:
                arms[name] = z["predictions"][[str(x) for x in z["ids"]].index(sid)]
        rows.append(
            make_row(sid, img, gt, arms, paths["raw_root"], fov, float(t["delta_mean_dice"]))
        )
    return rows


def render(panels: list[tuple[str, list[str], list[dict]]], cfg: dict) -> None:
    d = cfg["display"]
    pal = d["palettes"][PALETTE]
    fov, fs, k = float(d["fov_mm"]), d["font_size_pt"], d["pdf_upscale"]
    plt.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": d["font_family"], "font.size": fs}
    )
    # Shared panel size and spacing; centre each panel's consecutive columns.
    ncol = max(len(cols) for _, cols, _ in panels)
    w_in = d["width_mm"] / q.MM_PER_INCH * ncol / 5
    pw = w_in / (ncol + 0.03 * (ncol - 1))
    gap_x, gap_y = 0.03 * pw, 0.05 * pw
    head, between, legend = 0.30, 0.10, 0.62  # inches: column titles, panel gap, legend band
    nrows = sum(len(rows) for _, _, rows in panels)
    h_in = len(panels) * head + nrows * pw + (nrows - len(panels)) * gap_y
    h_in += (len(panels) - 1) * between + legend
    fig = plt.figure(figsize=(w_in, h_in))
    box = dict(facecolor="black", alpha=0.55, pad=1.2, lw=0)
    y = h_in  # top edge of the next band, inches from the bottom
    for letter, cols, rows in panels:
        panel_width = len(cols) * pw + (len(cols) - 1) * gap_x
        left = (w_in - panel_width) / 2
        fig.text(
            left / w_in,
            (y - 0.04) / h_in,
            letter,
            fontsize=fs + 1.5,
            fontweight="bold",
            va="top",
        )
        y -= head
        for r, row in enumerate(rows):
            names = [c for c in cols if c not in ("Image", "GT", "Change vs D")]
            first, second = row["preds"]["D"], row["preds"][names[-1]]
            image, gt, _, _, change = q.make_row_panels(dict(row, pD=first, pK=second), pal, d)
            arm_pan = {
                n: q.make_row_panels(dict(row, pD=row["preds"][n], pK=row["preds"][n]), pal, d)[2]
                for n in names
            }
            nr, nc = row["window"][2], row["window"][3]
            extent = (0, nc * row["dx"], nr * row["dy"], 0)
            bottom = y - pw
            for c, name in enumerate(cols):
                pan = {"Image": image, "GT": gt, "Change vs D": change}.get(name, arm_pan.get(name))
                pan = np.repeat(np.repeat(pan, k, axis=0), k, axis=1)
                ax = fig.add_axes(
                    [(left + c * (pw + gap_x)) / w_in, bottom / h_in, pw / w_in, pw / h_in]
                )
                ax.imshow(pan, extent=extent, interpolation="none")
                ax.set_xlim(0, fov)
                ax.set_ylim(fov, 0)
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if r == 0:
                    ax.set_title(name, fontsize=fs)
                if name in names:
                    ax.text(
                        0.03,
                        0.04,
                        f"DSC {row['dice'][name]:.3f}",
                        transform=ax.transAxes,
                        color="white",
                        fontsize=fs - 0.5,
                        va="bottom",
                        bbox=box,
                    )
                if name == "Image":
                    ax.text(
                        0.03,
                        0.97,
                        row["label"],
                        transform=ax.transAxes,
                        color="white",
                        fontsize=fs - 0.5,
                        va="top",
                        bbox=box,
                    )
                if name == "Change vs D":
                    ax.text(
                        0.97,
                        0.97,
                        row["delta_marker"],
                        transform=ax.transAxes,
                        color="white",
                        fontsize=fs - 0.5,
                        va="top",
                        ha="right",
                        bbox=box,
                    )
                    if r == len(rows) - 1:  # one scale bar per panel (identical mm scale)
                        b = d["scale_bar_mm"]
                        x1, y0 = fov - 4, fov - 5
                        ax.plot(
                            [x1 - b, x1], [y0, y0], color="white", lw=1.4, solid_capstyle="butt"
                        )
                        ax.text(
                            x1,
                            y0 - 2.5,
                            f"{b} mm",
                            color="white",
                            fontsize=fs - 0.5,
                            ha="right",
                            va="bottom",
                        )
            y = bottom - gap_y
        y += gap_y - between
    seg = [Patch(facecolor=q.hex_rgb(pal[n]), label=n) for n in q.CLASSES.values()]
    chg = [
        Patch(
            facecolor=q.hex_rgb(pal["corrected"]), label="Corrected (D wrong, compared arm right)"
        ),
        Patch(
            facecolor=q.hex_rgb(pal["new_error"]), label="New error (D right, compared arm wrong)"
        ),
        Patch(facecolor=q.hex_rgb(pal["both_wrong"]), label="Both wrong"),
    ]
    kw = dict(frameon=False, fontsize=fs - 0.5, handlelength=1.2, borderaxespad=0, ncol=1)
    fig.legend(
        handles=seg,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.0),
        title="Segmentation",
        title_fontsize=fs - 0.5,
        **kw,
    )
    fig.legend(
        handles=chg,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.0),
        title="Change vs D",
        title_fontsize=fs - 0.5,
        **kw,
    )
    fig.savefig(FIGURE.with_suffix(".png"), dpi=d["final_png_dpi"])
    fig.savefig(FIGURE.with_suffix(".pdf"), dpi=d["pdf_embed_dpi"])
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "paper/figures/config.yaml").read_text())
    fov = float(cfg["display"]["fov_mm"])
    panels = [
        ("a", ["Image", "GT", "U", "D", "BDS-Lite", "Change vs D"], rows_a(fov)),
        ("b", ["Image", "GT", "D", "D+KD-band", "Change vs D"], rows_b(cfg, fov)),
    ]
    render(panels, cfg)
    print("written:", FIGURE.with_suffix(".pdf"), FIGURE.with_suffix(".png"))


if __name__ == "__main__":
    main()
