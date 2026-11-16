# Results

`tables/` contains the saved numerical tables used by the manuscript and supplement.
`paper_inputs/` records the source of inventory configuration classifications.
`followup/results/contrasts.csv` stays unpacked for the independent archive verifier.

Case-level results, study reports and validation records are stored in the evidence archives.
Restore their original paths before running further analyses:

```bash
python evidence/restore.py --group results
```

The [evidence index](../evidence/README.md#study-index) maps each study to its result directory.
[archived_paths.csv](../evidence/archived_paths.csv) lists each file and its SHA-256.
The [reproduction guide](../docs/reproducing.md#archived-inputs) covers per-run training outputs
stored in the primary archive. Newly generated results and restored records are git-ignored.
