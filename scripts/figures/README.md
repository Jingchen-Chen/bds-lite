# Figure generation

Scripts that regenerate manuscript figures from committed artifacts of record.
None of them recompute experimental results; they visualize the CSV/JSON
artifacts and prediction arrays already shipped (or distributed via Zenodo).

| Manuscript figure | Generator | Source data | Output |
|---|---|---|---|
| **Figure 1** (architecture / train-deploy decoupling) | `generate_architecture_figure.py` | none (schematic from code) | `figures/figure_architecture_schematic.{png,pdf}` |
| **Figure 2** (cluster-level effect-size heatmap) | `generate_effect_size_heatmap.py` | `analysis/outputs/cluster_level_statistics.csv` (= Table 3) | `figures/figure_cluster_effect_size_heatmap.{png,pdf}` |
| **Figure 3** (representative / failure cases) | `analysis/generate_rescue_analysis.py` (24-panel selection) | `analysis/outputs/failure_case_manifest.csv` + Zenodo prediction arrays + processed GT | `figures/failure_panels/*.png` |

## Reproducibility note (read before comparing to the typeset PDF)

The figures **typeset in the published manuscript** were finalized inside the
LaTeX project (e.g., the polished multi-panel Figure 1, and the combined
Figure 3 montage). The scripts here regenerate the **content** of those figures
from the artifacts of record — the numbers, panels, and structure are
reproducible — but they are not guaranteed to be pixel-identical to the
hand-finalized publication exports. See `docs/paper_mapping.md` for the full
figure/table → artifact crosswalk.

The earlier `figures/figure1..figure6_*.{png,pdf}` set corresponds to a previous
six-figure manuscript layout; it is retained for provenance. The final paper
uses three figures, mapped above.

## Running

```bash
# from the repository root, with the package importable
python scripts/figures/generate_architecture_figure.py
python scripts/figures/generate_effect_size_heatmap.py
# Figure 3 panels (needs Zenodo prediction arrays + processed data):
python analysis/generate_rescue_analysis.py
```
