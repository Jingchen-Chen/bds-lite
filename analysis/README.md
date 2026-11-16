# Analysis

Run commands from the repository root. Start with the independent CPU verification:

```bash
python analysis/revision/verify_archived_statistics.py
```

For the remaining analyses, first restore saved supporting inputs:

```bash
python evidence/restore.py
```

| Directory | Scope |
| --- | --- |
| `revision/` | Component ablations, repeated-run statistics, historical corrections, diagnostics and paper assets |
| `kd/`, `kd3/` | Exploratory external-teacher distillation |
| `followup/` | ACDC controls, augmentation and post hoc analyses |
| `teacher_eval/` | Frozen-teacher evaluation on held-out ACDC patients |

The [study index](../evidence/README.md#study-index) links these analyses to the manuscript.
Some presentation generators import helpers from earlier generators; those dependencies are
retained. Registered model and training sources also retain their original bytes.
`python analysis/revision/build_paper_assets.py` regenerates presentation assets from saved evidence;
see [paper builds](../docs/reproducing.md#building-the-paper) for dependencies and full commands.
