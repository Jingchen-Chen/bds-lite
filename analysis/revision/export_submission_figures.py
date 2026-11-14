# ruff: noqa: E501
"""Export the main-text figures as separate files for PeerJ submission.

PeerJ CS figure requirements (author instructions, archived 2025-11-07): one file per figure,
named Figure1, Figure2, ...; vector PDF or EPS for charts and diagrams; PNG for raster images
(minimum 900 px, ideally about 3000 px); minimal white space; no titles or captions in the image.

Order and sources follow main.tex. Charts drawn as vectors are exported as PDF cropped to their
content (Ghostscript bounding box); figures whose content is raster are exported as PNG with the
surrounding white space trimmed. Writes revision/submission/figures/ and a manifest CSV.
Usage: python analysis/revision/export_submission_figures.py
"""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "revision/figures"
OUT = ROOT / "revision/submission/figures"
# (figure number, source, export kind)
FIGURES = [
    (1, FIG / "architecture_final.pdf", "pdf"),
    (2, FIG / "central_question.pdf", "pdf"),
    (3, FIG / "kd_acdc.pdf", "pdf"),
    # The former Figure 4 (ISIC2018 contours) is now in Supplementary S12.
]
MARGIN_PT = 2


def trim_png(source: Path, target: Path, pad: int = 8) -> tuple[int, int]:
    image = Image.open(source).convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    box = ImageChops.difference(image, background).getbbox()
    left, top, right, bottom = box
    box = (
        max(0, left - pad),
        max(0, top - pad),
        min(image.width, right + pad),
        min(image.height, bottom + pad),
    )
    cropped = image.crop(box)
    # PeerJ resubmission checklist: 900 px minimum, 3000 px maximum on all sides.
    longest = max(cropped.size)
    if longest > 3000:
        scale = 3000 / longest
        cropped = cropped.resize(
            (round(cropped.width * scale), round(cropped.height * scale)), Image.LANCZOS
        )
    if min(cropped.size) < 900:
        raise ValueError(f"{source.name}: {cropped.size} is below 900 px on one side")
    cropped.save(target, dpi=(300, 300))
    return cropped.size


def crop_pdf(source: Path, target: Path) -> tuple[float, float]:
    out = subprocess.run(
        ["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=bbox", str(source)],
        capture_output=True,
        text=True,
        check=True,
    ).stderr
    x0, y0, x1, y1 = (
        float(v) for v in re.search(r"%%HiResBoundingBox: (\S+) (\S+) (\S+) (\S+)", out).groups()
    )
    x0, y0, x1, y1 = x0 - MARGIN_PT, y0 - MARGIN_PT, x1 + MARGIN_PT, y1 + MARGIN_PT
    width, height = x1 - x0, y1 - y0
    subprocess.run(
        [
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=pdfwrite",
            "-dFIXEDMEDIA",
            f"-dDEVICEWIDTHPOINTS={width:.2f}",
            f"-dDEVICEHEIGHTPOINTS={height:.2f}",
            f"-sOutputFile={target}",
            "-c",
            f"<</PageOffset [{-x0:.2f} {-y0:.2f}]>> setpagedevice",
            "-f",
            str(source),
        ],
        check=True,
    )
    return width, height


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, source, kind in FIGURES:
        target = OUT / f"Figure{number}.{kind}"
        if kind == "png":
            width, height = trim_png(source, target)
            if max(width, height) < 900:
                raise ValueError(f"{target.name} is below PeerJ's 900 px minimum")
            size = f"{width}x{height} px"
        else:
            width, height = crop_pdf(source, target)
            size = f"{width:.0f}x{height:.0f} pt (vector)"
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        rows.append([target.name, source.relative_to(ROOT).as_posix(), size, digest])
    with (OUT / "manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["file", "source", "size", "sha256"])
        writer.writerows(rows)
    for row in rows:
        print(*row[:3])
    # Remove superseded numbered exports only after every current export succeeds.
    # Source figures remain intact; the submission directory contains one file per figure.
    current = {row[0] for row in rows}
    for path in OUT.iterdir():
        if re.fullmatch(r"Figure\d+\.(?:png|pdf)", path.name) and path.name not in current:
            path.unlink()


if __name__ == "__main__":
    main()
