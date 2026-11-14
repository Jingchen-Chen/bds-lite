"""Shared typesetting of the main-text figures (Figs. 1-3 of revision/manuscript/main.tex).

- width: \\the\\textwidth of wlpeerj.cls, 415.12572 TeX pt (measured with tectonic); figures are
  drawn at this width and included without scaling, so the drawn type size is the printed one;
- type: STIXGeneral text and STIX mathtext (Times family, as the body text), at least 7 pt;
  TrueType embedding (pdf.fonttype = ps.fonttype = 42);
- colour: Okabe-Ito (Okabe & Ito 2008, "Color Universal Design"), with black and greys for
  neutral marks; categories also differ in marker shape or fill, so they survive greyscale.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

TEXTWIDTH_TEX_PT = 415.12572
W_IN = TEXTWIDTH_TEX_PT / 72.27

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
INK = OKABE_ITO["black"]
MUTED = "#6b6b6b"  # neutral grey for secondary estimates and the zero line
BAND = "#e4e4e4"  # SESOI shading

RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply() -> None:
    plt.rcParams.update(RC)
