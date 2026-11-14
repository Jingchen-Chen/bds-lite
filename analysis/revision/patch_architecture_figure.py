"""Correct two errors in the authors' Figure 1 raster without touching the source file.

The source PNG drew the distillation target leaving the 1x1 boundary head instead of q_b, and its
gate formula omitted the bias b_g. This script reroutes the arrow from the q_b line, moves the
F_b = q_b label clear of the new line, and redraws the formula.

Input:  revision/manuscript/figure_architecture.png (unchanged)
Output: revision/manuscript/figure_architecture_v2.png
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "revision/manuscript/figure_architecture.png"
TARGET = ROOT / "revision/manuscript/figure_architecture_v2.png"
X0, Y0 = 2155, 801  # origin of the edited region in source pixels
WIDTH = 7


def formula(height: int, color: tuple[int, int, int]) -> Image.Image:
    plt.rcParams.update({"mathtext.fontset": "stix"})
    fig = plt.figure(figsize=(8, 1), dpi=300)
    ink = [c / 255 for c in color]
    text = r"$g=\sigma(W_g\,q_s+b_g),\ \alpha=0.25$"
    label = fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=20, color=ink)
    label.set_path_effects([patheffects.withStroke(linewidth=0.7, foreground=ink)])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    image = Image.open(buffer).convert("RGBA")
    return image.resize((round(image.width * height / image.height), height), Image.LANCZOS)


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    draw = ImageDraw.Draw(image)

    def at(x: int, y: int) -> tuple[int, int]:
        return X0 + x, Y0 + y

    white = image.getpixel(at(120, 330))
    gate_fill = image.getpixel(at(250, 66))
    orange = image.getpixel(at(70, 458))

    # Move the label below the q_b line.
    label = image.crop((*at(21, 350), *at(164, 438)))
    draw.rectangle((*at(21, 350), *at(164, 438)), fill=white)
    image.paste(label, at(23, 472))

    # Remove the old route out of the 1x1 head; the final arrow into L_distill stays.
    draw.rectangle((*at(579, 447), *at(684, 466)), fill=white)
    draw.rectangle((*at(660, 328), *at(735, 352)), fill=white)
    draw.rectangle((*at(660, 328), *at(684, 466)), fill=white)

    # New route: from the q_b line up and across to the existing arrow.
    branch = 92
    column = [y for y in range(320, 360) if image.getpixel(at(745, y))[2] < 150]
    level = (column[0] + column[-1]) // 2
    draw.line((*at(branch, 458), *at(branch, level)), fill=orange, width=WIDTH)
    draw.line((*at(branch - 3, level), *at(745, level)), fill=orange, width=WIDTH)
    x, y = at(branch, 458)
    draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=orange)

    # Redraw the gate formula with the bias.
    draw.rectangle((*at(288, 44), *at(690, 92)), fill=gate_fill)
    text = formula(44, (10, 10, 10))
    x = X0 + 482 - text.width // 2
    image.paste(text, (x, Y0 + 45), text)
    image.save(TARGET, optimize=True)


if __name__ == "__main__":
    main()
