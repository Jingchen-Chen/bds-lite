# Controlled migration — 2026-11-02

Plan recorded before research-file migration. Authority: `revision/plan.md`, copied byte-for-byte from `/home/jc/Downloads/plan.md`. No original research file will be overwritten.

| Source | Destination | Decision |
|---|---|---|
| `/home/jc/bds-lite` | existing paths | Retain all existing content; baseline SHA256 in audit/inventory.csv. |
| `/home/jc/bds-followup` | `recovered/followup/` | Preserve research protocols, failed/aborted runs, complete runs, predictions, configs, source, tests, splits, raw/derived data and licenses together. Separate scientific family; not automatically manuscript evidence. |
| `/home/jc/bds-lite-new` | `recovered/lite-new/` | Preserve differences and directory structure; exact duplicates become relative links into primary or an earlier recovered file. Not selected as active implementation. |
| `/home/jc/bds-latex` | `recovered/peerj-submission/` | Preserve typeset manuscript, bibliography, figures and submission ZIP; do not silently treat as revised manuscript. |
| followup Git refs | `revision/audit/followup-history.bundle` | Preserve all reachable history without changing source branches/remotes; untracked content separately covered by inventory/migration. |

Execution: `scripts/revision/migrate_sources.py`. Per-file source, destination, SHA256 and action will be written to `revision/audit/migration.csv`. New files are copied and destination-hashed; identical content is linked only to a path inside the primary repository (no hardlinks to originals). Existing destination paths cause failure. SHA256 identity verifies transfer integrity, not scientific validity.

Excluded: virtual environments, Git working metadata (history bundled separately), Python/test/lint/build caches, downloaded tool wheel/site directories, and regenerable LaTeX build intermediates. Research data/downloads and checkpoints are retained; duplicate bytes are not recopied. Existing source symlinks are inventoried but not automatically followed. Original source directories remain unchanged. Absolute paths embedded inside historical configs remain historical metadata; execution requires explicit path resolution.
