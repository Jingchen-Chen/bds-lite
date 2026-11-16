# Reproduction guide

Run commands from the repository root. The [README](../README.md) covers installation and a basic
training example. This guide covers the archived paper experiments and their dependencies.

## Environment

`pyproject.toml` declares the package dependencies and the `dev` and `data` extras.
[env_freeze.txt](env_freeze.txt) records a partial study environment inventory, including PyTorch
2.11.0 and CUDA 13.0. The external teacher uses `segmentation_models_pytorch==0.6.0`.
The recorded environment includes local overlays and has not been validated as a portable installation;
some study trainers explicitly check the recorded PyTorch/CUDA versions. The dated dependency
audit is preserved as `docs/REPRODUCIBILITY_STATUS_20261110.md` in the supporting records;
restore it with `python evidence/restore.py --group provenance`.

For statistical verification alone, a separate CPU environment with NumPy and SciPy is sufficient:

```bash
python3 -m venv /tmp/bds-lite-analysis
source /tmp/bds-lite-analysis/bin/activate
python -m pip install numpy scipy
python analysis/revision/verify_archived_statistics.py
python evidence/verify_snapshot.py
```

The first script recomputes 40 metric contrasts and 20 Holm-adjusted values from the two evidence
archives. The second verifies the primary archive's member checksums and independently recomputes
25 contrasts and 20 Holm-adjusted values. Neither script changes the saved results.

## Archived inputs

Supporting results, protocols and provenance are packaged to keep the public checkout compact.
Restore their exact original paths before running study analyses or training:

```bash
python evidence/restore.py
```

Use `--check` for a read-only integrity check, or `--group results`, `--group protocols` or
`--group provenance` to select records. Existing different files are never overwritten.
The [evidence index](../evidence/README.md) describes each archive and study;
the [restoration manifest](../evidence/archived_paths.csv) lists every destination and SHA-256.

The primary and supplementary evidence archives contain the case-level outputs used for reanalysis.
Archive digests are recorded in [BDS_LITE_EVIDENCE_FREEZE.md](../evidence/BDS_LITE_EVIDENCE_FREEZE.md);
the adjacent `.sha256` files list checksums of the files inside each archive.

```bash
sha256sum evidence/primary-evidence-20261110.tar.gz evidence/final-supplement-20261110.tar.gz
```

Extract the per-run case results, completion records and locked split manifests needed by the
paper build and study scripts:

```bash
tar -xzf evidence/primary-evidence-20261110.tar.gz --wildcards \
  'results/revision_runs/*/per_case_test.csv' 'results/revision_runs/*/final_metrics.json' \
  'recovered/followup/splits/*'
```

The separate `evidence/addendum-20261115.tar.gz` contains the post hoc gate-identity evaluation
of the **unweighted** ACDC reference. Its extracted numerical input is kept in
`results/gate_identity_unweighted_acdc/gate_identity_effects.csv`; `source.json` records the archive,
original member path and SHA-256. `make_gate_addendum_assets.py` writes independent `GIU.acdc.*`
keys. The earlier class-weighted follow-up result remains under `GID.acdc.*` and is identified
separately in Supplementary S19. Ordinary builds consume these saved results without gate inference.

Dataset images, complete model checkpoints and teacher caches must be obtained separately for
training and inference. The gate addendum includes selected prediction arrays, not a complete set of
model weights or predictions for the paper. Recorded hashes identify artifacts but do not reconstruct them.

## Paper experiments

The basic `scripts/train.py` / `scripts/evaluate.py` interface uses directory-based splits.
Paper studies use registered manifests, their own training adapters and study-specific aggregation
rules. Use the configuration and queue for the intended study rather than substituting a basic
training configuration.

| Study | Training / queue entry points | Specification |
| --- | --- | --- |
| Ablations | `scripts/revision/run_queue.py`, `scripts/revision/train_run.py` | `protocols/ablation/ABLATION_PROTOCOL.md` |
| Comparators | `scripts/comparators/run_queue_b.py`, `scripts/comparators/train_comparator.py` | `protocols/comparators/COMPARATOR_PROTOCOL.md` |
| ISIC2018 external teacher | `scripts/kd/run_queue_kd.py`, `scripts/kd/train_kd.py` | `protocols/kd_isic2018/KD_COMPARATOR_PROTOCOL.md` |
| ACDC / Synapse external teacher | `scripts/kd_mc/run_queue_mc.py`, `scripts/kd_mc/train_mc.py` | `protocols/kd_acdc_synapse/KD_MULTICLASS_PROTOCOL.md` |
| ACDC confirmation | `scripts/acdc_confirm/run_queue.py`, `scripts/acdc_confirm/launch.py` | `protocols/acdc_confirm/acdc_confirmation.md` |
| Follow-up controls | `scripts/kd_followup/run_queue.py` | `protocols/followup/followup_registration.md` |

Protocol paths become available after restoration. The [study index](../evidence/README.md#study-index)
covers additional studies; the complete protocol/queue index is restored to `protocols/README.md`.
Training requires the processed datasets and a GPU; teacher experiments also require the pretrained
weights and caches specified by their protocols.

### Paths in archived records

This release is a snapshot of the development repository. Frozen records retain original paths
such as `revision/queue_acdc_confirm.yaml` and commit identifiers from that repository.
[evidence/path_map.csv](../evidence/path_map.csv) maps the original paths to the public layout;
development commit identifiers cannot be resolved in this snapshot.

Before rerunning frozen study scripts, restore their original paths as ignored symbolic links:

```bash
python evidence/restore_original_layout.py
```

The helper leaves existing files untouched. Frozen protocols, queues and registered source files
should retain their recorded bytes so that checksum verification remains meaningful.

## Timestamp verification

After extracting the split manifests above, check the files listed in the timestamp manifests:

```bash
python evidence/verify_timestamp_manifests.py
```

With `opentimestamps-client` installed, inspect and verify the ACDC registration proof:

```bash
ots info protocols/acdc_confirm/acdc_confirmation.md.ots
sha256sum protocols/acdc_confirm/acdc_confirmation.md
ots verify protocols/acdc_confirm/acdc_confirmation.md.ots
```

`ots verify` requires a local Bitcoin node. The SHA-256 of the registration should match the digest
reported by `ots info`. The manifest proof at
`protocols/acdc_confirm/TIMESTAMP_MANIFEST.txt.ots` can be checked the same way.
The ACDC attestations reference Bitcoin block 974994. Follow-up proofs under `protocols/followup/`
contain pending calendar attestations only. Supplementary S18 and S21 describe their scope.

## Tests

Install the `dev` extra, restore supporting records, extract the split manifests and copy them to the location used by the split
integrity tests (for a fresh checkout):

```bash
mkdir -p data
python evidence/restore.py
cp -r recovered/followup/splits data/splits
PYTHONPATH=src:. python -m pytest -q
```

Additional requirements for the full suite:

- Comparator tests load LB-UNet. Run `python scripts/comparators/fetch_lbunet.py` to fetch its pinned,
  checksum-verified source into `third_party/`.
- `tests/revision/test_thermal_supervision.py` requires at least 20 GB free on the temporary
  filesystem. Use pytest's `--basetemp` option to select another disk.
- Two tests in `tests/revision/test_acdc_confirm_amendment1.py` depend on development commit
  `345cacd` and are skipped in this snapshot. Their source is retained as registered in Amendment 1.
- One test in `tests/revision/test_kd_multiclass.py` is skipped without `segmentation_models_pytorch`.
- The exploratory-contrast test is skipped if its per-run case CSVs have not been extracted from the primary archive.

## Building the paper

Prebuilt PDFs are in [paper/](../paper/README.md). To regenerate figures, tables and PDFs:

```bash
bash paper/build.sh
```

The build uses `.venv/bin/python` with Matplotlib installed, Tectonic (default `~/.local/bin/tectonic`,
or set `TECTONIC`), and `pdftotext` / `pdfinfo` from Poppler. It automatically restores supporting
results, protocols and provenance, then formats the saved structured evidence, checks published
figure assets, and compiles the documents. It does not train models, run inference, measure latency,
or recompute scientific contrast analyses. The supplemental labels layer verifies that numeric tokens
are preserved when applying the manuscript's terminology.

The required case CSVs, completion records and split manifests are read from the primary evidence
archive into a temporary cache, with member SHA-256 checks. Set `BDS_PAPER_CACHE` to choose a cache
location. No original-layout symbolic links or development checkout are required for this build.
Configuration weight classifications used by the inventory presentation are checked against the
original JSON members of the provided archives.

Published comparison values are read from `paper/lit_numbers.csv` and
`paper/literature_context.json`. These are saved source receipts; a normal build does not claim to
repeat full-text literature verification. Obtain the cited papers separately to recheck their tables;
when the optional four source PDFs are present in `paper/literature/`, the comparison helper also
checks their recorded hashes. Copyrighted papers and the historical submitted manuscript are not
needed by the public build. Historical submission word counts are omitted from this build.

Figure 3 and the supplemental data-dependent figures use validated published assets with checksums
in `paper/figures/published_assets.json`. Figure 3a uses internal ACDC patients (U, D, BDS-Lite);
Figure 3b uses official patients 101–150 (D, D+KD-band). Saved selection records accompany each panel.
Full redraw needs the corresponding dataset inputs, geometry and saved predictions; see
`paper/figures/config.yaml`, `figure_qualitative_combined.py` and the explicit `--redraw` mode of
`make_supplement_figures.py`. Neither redraw command is part of a normal paper build.
