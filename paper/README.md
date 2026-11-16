# Paper

- `main.tex` / `main.pdf`: manuscript; `supplement.tex` / `supplement.pdf`: supplement
- `generated/`: number keys and tables written by the scripts in `analysis/revision/` (do not edit by hand)
- `figures/`: figures (the graphics path of both documents)
- `lit_numbers.csv`: published numbers for the literature-context table, with the SHA-256 of each source paper
- `wlpeerj.cls`, `references_zotero.bib`: template and bibliography

Run `bash paper/build.sh` from the repository root to regenerate the keys, tables and figures and compile
both PDFs (see the [reproduction guide](../docs/reproducing.md#building-the-paper) for its inputs).
It does not rerun frozen analyses or training.
`analysis/revision/verify_editorial_numbers.py` checks the central numbers against the per-case files.

The absolute means and SDs of the comparison table come from `analysis/revision/make_comparison_table.py`.
The public build reads checksum-verified case results directly from the provided evidence archive
into a temporary cache; it does not need a development checkout or third-party paper PDFs.
Published source values and citations are recorded in `lit_numbers.csv` and `literature_context.json`.

Figure 3 uses the validated `figures/acdc_qualitative_combined.pdf`. Panel 3a compares U, D and
BDS-Lite on internal ACDC test patients (seed 2; selection records in `results/qualitative_bdslite/`).
Panel 3b compares D and D+KD-band on official patients 101–150 (seed 206; records in
`results/tables/acdc_qualitative/`). The panels belong to different studies and test sets.

Normal builds verify the saved image assets and do not select cases or run inference.
`analysis/revision/figure_qualitative_combined.py` can redraw the recorded panels only when the
external raw/processed data and saved predictions are available at the paths in `figures/config.yaml`.
The supplemental figure layer verifies published assets by default; its explicit `--redraw --scratch DIR`
mode needs the corresponding external predictions and data. See the reproduction guide for details.
