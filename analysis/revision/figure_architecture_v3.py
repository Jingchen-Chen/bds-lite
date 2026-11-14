# ruff: noqa: E501
"""Fig. 1 (Arms and training signals), vector source of revision/figures/architecture_final.pdf.

History: the raster Fig. 1 (revision/figures/architecture.png) was redrawn in matplotlib and the
redraw was approved by the author on 2026-11-10. Revision of 2026-11-11 (layout and typesetting
only; no numbers, the figure carries none from result files):

- drawn at the PeerJ text width (\\the\\textwidth = 415.12572 pt, measured with tectonic and
  wlpeerj.cls) and included without scaling, so the printed type is the drawn type: 7-8.5 pt;
- STIXGeneral text with STIX mathtext (Times family, as the body text); TrueType embedding
  (pdf.fonttype = ps.fonttype = 42);
- Okabe-Ito palette with three classes only (kept at inference, training only, loss term);
  training-only boxes and paths are also dashed, so the classes survive greyscale printing;
- gate (a) drawn as in Eq. 1: q_s -> g = sigma(W_g q_s + b_g) -> 1 + alpha(g - 0.5) -> (x),
  F_s -> (x) -> F_s^m -> W_o -> z_s; the gate is a deployed (blue) component;
- skip connections drawn to both decoders: BoundaryDecoder.forward reads x1..x4 exactly as
  UNetDecoder.forward does (src/bds_lite/models/bds_lite.py:24-29, unet.py:35-40);
- surface-loss notation of Supplementary S2 (phi_nk, s_nk = max_u |phi_nk(u)|);
- objectives box removed (Eqs. 4-5 and Table 3 state the objectives); one line of symbols kept.

Output: revision/figures/architecture_final.{pdf,png}.

    .venv/bin/python analysis/revision/figure_architecture_v3.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "revision/figures"

TEXTWIDTH_TEX_PT = 415.12572  # \the\textwidth of wlpeerj.cls (TeX points, 72.27 per inch)
W_IN = TEXTWIDTH_TEX_PT / 72.27
W = W_IN * 72  # drawing units are PostScript points, so font sizes map 1:1
H = 313.0
Y0 = 15.0  # lowest drawn y (footer line)

# Okabe-Ito
BLUE, ORANGE, VERMILLION, BLACK = "#0072B2", "#E69F00", "#D55E00", "#000000"
FILL = {"deploy": "#D9EAF3", "train": "#FDF3DC", "loss": "#F2CFB2"}
EDGE = {"deploy": BLUE, "train": ORANGE, "loss": VERMILLION}
DASH = (0, (3.0, 1.6))
FS, FS_SMALL, FS_TITLE = 7.5, 7.0, 8.5
LW = 0.8
LANE = 6.0  # spacing of the skip arrows entering D_s and D_b (>= 2 mm = 5.67 pt)

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": FS,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def box(ax, x, y, w, h, text, kind="deploy", fs=FS, lw=LW, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=2.2",
            fc=FILL[kind],
            ec=EDGE[kind],
            lw=lw,
            ls=DASH if kind == "train" else "-",
            zorder=2,
        )
    )
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            linespacing=1.15,
            fontweight=weight,
            zorder=3,
        )


def badge(ax, x, y, w, h, letter):
    """Component letter of Table 3 on the top-right corner of a box."""
    cx, cy = x + w, y + h
    ax.add_patch(Circle((cx, cy), 4.6, fc="white", ec=BLACK, lw=0.7, zorder=6))
    ax.text(cx, cy - 0.2, letter, ha="center", va="center", fontsize=7, fontweight="bold", zorder=7)


def path(ax, pts, kind="deploy", arrow=True):
    color = BLUE if kind == "deploy" else ORANGE
    xs, ys = zip(*pts, strict=True)
    ax.add_line(
        Line2D(
            xs,
            ys,
            color=color,
            lw=LW if kind == "deploy" else LW + 0.15,
            ls="-" if kind == "deploy" else DASH,
            solid_capstyle="butt",
            dash_capstyle="butt",
            zorder=1,
        )
    )
    if arrow:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ax.add_patch(
            FancyArrowPatch(
                (x1 - 0.02 * (x1 - x0), y1 - 0.02 * (y1 - y0)),
                (x1, y1),
                arrowstyle="-|>,head_length=0.5,head_width=0.25",
                mutation_scale=7,
                color=color,
                lw=0,
                shrinkA=0,
                shrinkB=0,
                zorder=4,
            )
        )


def dot(ax, x, y, kind="deploy"):
    ax.add_patch(Circle((x, y), 1.2, color=BLUE if kind == "deploy" else ORANGE, zorder=4))


def label(ax, x, y, text, ha="center", va="bottom", fs=FS_SMALL):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, zorder=5)


def title(ax, x, y, text):
    ax.text(x, y, text, fontsize=FS_TITLE, fontweight="bold", va="center")


def legend(ax, x, y):
    items = [
        ("deploy", "kept at inference"),
        ("train", "training only"),
        ("loss", "loss term"),
    ]
    cx = x
    for kind, text in items:
        ax.add_patch(
            FancyBboxPatch(
                (cx, y - 3.2),
                11,
                6.4,
                boxstyle="round,pad=0,rounding_size=1.2",
                fc=FILL[kind],
                ec=EDGE[kind],
                lw=LW,
                ls=DASH if kind == "train" else "-",
            )
        )
        t = ax.text(cx + 14, y, text, va="center", fontsize=FS_SMALL)
        cx += 14 + {"kept at inference": 50, "training only": 39, "loss term": 27}[t.get_text()] + 9
    for kind, text, adv in (
        ("deploy", "inference path", 46),
        ("train", "training-only path", 56),
    ):
        path(ax, [(cx, y), (cx + 11, y)], kind)
        ax.text(cx + 14, y, text, va="center", fontsize=FS_SMALL)
        cx += 14 + adv + 9


def panel_a(ax):
    title(ax, 2, 322, "(a) BDS-Lite (arm D+F+G+S) during training")
    legend(ax, 2, 309)

    # input and shared encoder
    box(ax, 1, 225, 30, 22, "input $x$\n$H\\!\\times\\!W\\!\\times\\!C$", "deploy", fs=FS_SMALL)
    path(ax, [(31, 236), (36, 236)])
    box(ax, 36, 213, 141, 46, "", "deploy")
    ax.text(
        43,
        236,
        "encoder $E$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=FS,
        fontweight="bold",
    )
    stages = []
    for i, ch in enumerate(["$c$", "$2c$", "$4c$", "$8c$"]):
        sx = 50 + i * 32
        box(ax, sx, 220, 26, 32, f"stage {i + 1}\n{ch}\n$x_{i + 1}$", "deploy", fs=FS_SMALL)
        stages.append(sx + 13)
        if i < 3:
            path(ax, [(sx + 26, 236), (sx + 32, 236)])
    # skips x1..x3 to D_s (up) and D_b (down); the leftmost stage takes the outermost lane
    for i, sx in enumerate(stages[:3]):
        path(ax, [(sx, 252), (sx, 297 - LANE * i), (190, 297 - LANE * i)], "deploy")
        path(ax, [(sx, 220), (sx, 175 + LANE * i), (190, 175 + LANE * i)], "train")
    # bottleneck x4 to both decoders
    path(ax, [(176, 236), (183, 236)], "deploy", arrow=False)
    dot(ax, 183, 236)
    path(ax, [(183, 236), (183, 297 - 3 * LANE), (190, 297 - 3 * LANE)], "deploy")
    path(ax, [(183, 236), (183, 175 + 3 * LANE), (190, 175 + 3 * LANE)], "train")

    # segmentation path: D_s -> F_s -> (x) -> F_s^m -> W_o -> z_s
    box(ax, 190, 270, 58, 30, "segmentation\ndecoder $D_s$\n(3 up-blocks)", "deploy")
    path(ax, [(248, 285), (265, 285)], "deploy", arrow=False)
    label(ax, 256.5, 286.5, "$F_s$")
    dot(ax, 265, 285)
    path(ax, [(265, 285), (325.5, 285)], "deploy")
    ax.add_patch(Circle((331, 285), 5.5, fc="white", ec=BLUE, lw=LW + 0.2, zorder=3))
    ax.text(331, 285, "$\\times$", ha="center", va="center", fontsize=9, color=BLUE, zorder=4)
    path(ax, [(336.5, 285), (349, 285)], "deploy")
    label(ax, 342.75, 286.5, "$F_s^{\\mathrm{m}}$")
    box(ax, 349, 274, 29, 22, "$W_o$: 1$\\times$1\n$c\\to K$", "deploy", fs=FS_SMALL)
    path(ax, [(378, 285), (384, 285)], "deploy")
    box(ax, 384, 274, 26, 22, "logits\n$z_s$", "deploy", fs=FS_SMALL)

    # projection P and gate G (both deployed)
    path(ax, [(265, 285), (265, 259)], "deploy")
    box(ax, 247, 237, 36, 22, "$P$: 1$\\times$1\n$c\\to c_b$", "deploy", fs=FS_SMALL)
    path(ax, [(283, 248), (296, 248)], "deploy")
    dot(ax, 289, 248)
    label(ax, 289, 249.5, "$q_s$")
    box(ax, 296, 237, 70, 22, "$g=\\sigma(W_gq_s+b_g)$\n$\\alpha=0.25$", "deploy", lw=1.7)
    badge(ax, 296, 237, 70, 22, "G")
    path(ax, [(331, 259), (331, 279.5)], "deploy")
    label(ax, 328, 269, "$1+\\alpha(g-0.5)$", ha="right", va="center")

    # feature distillation F
    path(ax, [(289, 248), (289, 227)], "train")
    box(
        ax,
        276,
        205,
        90,
        22,
        "$\\lambda_dL_{\\mathrm{distill}}$\nSmoothL1$(q_s,\\mathrm{sg}(q_b))$",
        "loss",
        fs=FS_SMALL,
    )
    badge(ax, 276, 205, 90, 22, "F")

    # boundary path (training only)
    box(ax, 190, 172, 58, 30, "boundary\ndecoder $D_b$\n(3 up-blocks)", "train")
    badge(ax, 190, 172, 58, 30, "D")
    path(ax, [(248, 187), (262, 187)], "train", arrow=False)
    label(ax, 255, 188.5, "$q_b$")
    dot(ax, 262, 187, "train")
    path(ax, [(262, 187), (262, 216), (276, 216)], "train")
    path(ax, [(262, 187), (270, 187)], "train")
    box(ax, 270, 176, 36, 22, "1$\\times$1\n$c_b\\to1$", "train", fs=FS_SMALL)
    path(ax, [(306, 187), (322, 187)], "train")
    label(ax, 314, 188.5, "$z_b$")
    box(ax, 322, 176, 58, 22, "$\\lambda_bL_{\\mathrm{boundary}}$\nsee (b)", "loss", fs=FS_SMALL)


def panel_b(ax, x0):
    title(ax, x0, 151, "(b) Boundary supervision")
    box(ax, x0 + 4, 116, 44, 22, "$z_b$\nfrom (a)", "train", fs=FS_SMALL)
    path(ax, [(x0 + 26, 116), (x0 + 26, 100)], "train")
    box(ax, x0 + 4, 78, 44, 22, "$\\sigma(z_b)$", "train", fs=FS_SMALL)
    box(ax, x0 + 52, 116, 78, 22, "reference\nmask $y$", "train", fs=FS_SMALL)
    path(ax, [(x0 + 91, 116), (x0 + 91, 108)], "train")
    box(
        ax,
        x0 + 52,
        70,
        78,
        38,
        "$y_b$: dilation $-$ erosion\n(3$\\times$3 cross), union over\nforeground classes",
        "train",
        fs=FS_SMALL,
    )
    path(ax, [(x0 + 26, 78), (x0 + 26, 56)], "train")
    path(ax, [(x0 + 91, 70), (x0 + 91, 56)], "train")
    box(
        ax,
        x0 + 14,
        34,
        100,
        22,
        "$\\lambda_bL_{\\mathrm{boundary}}$\nBCE + Dice on $(z_b,\\,y_b)$",
        "loss",
        fs=FS_SMALL,
    )
    badge(ax, x0 + 14, 34, 100, 22, "D")


def panel_c(ax, x0):
    title(ax, x0, 151, "(c) Segmentation and surface losses")
    box(ax, x0 + 4, 116, 52, 22, "$z_s$\nfrom (a)", "deploy", fs=FS_SMALL)
    path(ax, [(x0 + 30, 116), (x0 + 30, 100)], "deploy")
    box(ax, x0 + 4, 78, 52, 22, "$p=\\mathrm{softmax}(z_s)$", "deploy", fs=FS_SMALL)
    box(ax, x0 + 62, 116, 66, 22, "reference\nmask $y$", "train", fs=FS_SMALL)
    path(ax, [(x0 + 95, 116), (x0 + 95, 108)], "train")
    box(
        ax,
        x0 + 62,
        70,
        66,
        38,
        "signed distance\n$\\phi_{nk}(u)$, $<0$ inside;\n$s_{nk}=\\mathrm{max}_u\\,|\\phi_{nk}(u)|$",
        "train",
        fs=FS_SMALL,
    )
    path(ax, [(x0 + 30, 78), (x0 + 30, 56)], "train")
    dot(ax, x0 + 30, 63, "train")
    path(ax, [(x0 + 30, 63), (x0 + 72, 63), (x0 + 72, 56)], "train")
    path(ax, [(x0 + 95, 70), (x0 + 95, 56)], "train")
    box(ax, x0 + 4, 34, 52, 22, "$L_{\\mathrm{seg}}(p,y)$\nCE + Dice", "loss", fs=FS_SMALL)
    box(
        ax,
        x0 + 62,
        34,
        66,
        22,
        "$\\lambda_{\\mathrm{sdf}}L_{\\mathrm{sdf}}$: mean of\n$p_{nk}(u)\\,\\phi_{nk}(u)/s_{nk}$",
        "loss",
        fs=FS_SMALL,
    )
    badge(ax, x0 + 62, 34, 66, 22, "S")


def panel_d(ax, x0):
    title(ax, x0, 151, "(d) External-teacher distillation")
    box(
        ax,
        x0 + 4,
        108,
        74,
        30,
        "teacher (frozen): U-Net,\nResNet-34 encoder,\nImageNet-pretrained",
        "train",
        fs=FS_SMALL,
    )
    path(ax, [(x0 + 78, 123), (x0 + 86, 123)], "train")
    box(ax, x0 + 86, 112, 36, 22, "cached\n$p^{t}$", "train", fs=FS_SMALL)
    path(ax, [(x0 + 104, 112), (x0 + 104, 104)], "train")
    box(
        ax,
        x0 + 22,
        74,
        100,
        30,
        "band $B$: dilation $-$ erosion\n(3$\\times$3 square) of $\\arg\\max\\,p^{t}$,\nunion over foreground classes",
        "train",
        fs=FS_SMALL,
    )
    path(ax, [(x0 + 122, 123), (x0 + 129, 123), (x0 + 129, 45), (x0 + 122, 45)], "train")
    path(ax, [(x0 + 90, 74), (x0 + 90, 56)], "train")
    box(ax, x0 + 4, 34, 40, 22, "student $p$\n(arm D)", "deploy", fs=FS_SMALL)
    path(ax, [(x0 + 44, 45), (x0 + 58, 45)], "train")
    box(
        ax,
        x0 + 58,
        34,
        64,
        22,
        "$\\lambda_{\\mathrm{kd}}L_{\\mathrm{kd}}$\nCE$(p^{t},p)$ in $B$",
        "loss",
        fs=FS_SMALL,
    )


def draw() -> plt.Figure:
    fig = plt.figure(figsize=(W_IN, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(Y0, H + Y0)
    ax.axis("off")
    panel_a(ax)
    grid = {"color": "#9a9a9a", "lw": 0.6, "ls": (0, (2, 2))}
    ax.plot([1, W - 1], [163, 163], **grid)
    x0s = (2, 2 + (W - 4 + 6) / 3, 2 + 2 * (W - 4 + 6) / 3)
    for xd in x0s[1:]:
        ax.plot([xd - 4, xd - 4], [33, 160], **grid)
    panel_b(ax, x0s[0])
    panel_c(ax, x0s[1])
    panel_d(ax, x0s[2])
    ax.text(
        2,
        Y0 + 2,
        "$c=32$: base channels;  $c_b=\\max(c/2,8)=16$;  $K$: number of classes;  sg: stop-gradient;  D, F, G, S: components of Table 3.",
        fontsize=FS_SMALL,
        va="bottom",
    )
    return fig


def main() -> None:
    fig = draw()
    fig.savefig(FINAL / "architecture_final.pdf", metadata={"CreationDate": None})
    fig.savefig(FINAL / "architecture_final.png", dpi=300)
    plt.close(fig)
    print("wrote", FINAL / "architecture_final.{pdf,png}", f"({W_IN:.4f} x {H / 72:.4f} in)")


if __name__ == "__main__":
    main()
