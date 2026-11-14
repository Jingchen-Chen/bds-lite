# BDS-Lite: boundary distillation in compact 2D medical image segmentation

Code, configurations, split manifests, study protocols, frozen analysis outputs and manuscript sources
for

> J. Chen, C. Hong, Y. Wu, T. Yang. *When does boundary distillation help lightweight medical image
> segmentation?* PeerJ Computer Science, manuscript CS-2026:06:142937 (revised version).

The findings, with their uncertainty and limitations, are stated in the manuscript
(`revision/manuscript/main.pdf`) and its supplement (`revision/manuscript/supplement.pdf`). This
README makes no further claim about them.

## Release notes: a snapshot without development history

This repository is a **single-commit snapshot**. The authors' development repository is not public:
besides the code and records published here, it holds confidential peer-review material and, in
earlier commits, figures with dataset images whose licences do not permit republication (see
`revision/LICENSING_AUDIT.md`). The snapshot was made with `git archive` from one commit of that
repository; the commit message of the snapshot names that commit. The files listed under
"What is not included" were then removed, one log file was redacted (below), and this README was
moved to the root. Every other file is byte-identical to the development repository at that
commit. The earlier public history of this GitHub repository (July 2026) is not continued here.

### Commit identifiers

Commit identifiers that appear in files of this snapshot (for example `345cacd` in
`revision/acdc_confirm/TIMESTAMP_MANIFEST.txt`, `f2960239` in
`revision/followup/TIMESTAMP_MANIFEST.txt`, or the commits cited in protocols, amendments, deviation
logs and audit records) refer to the development repository. They cannot be resolved in this
snapshot. The development repository is available to the editor on request. Commit dates are local
records of that repository; the external time evidence is the OpenTimestamps proofs below, which
prove the content of files, not commits.

### Verifying frozen files: SHA-256 and OpenTimestamps

Each frozen file is identified by its SHA-256 digest. The digests are recorded in the timestamp
manifests, the study protocols and amendments, the evidence-archive manifests, and the run
configurations.

1. Files listed in a timestamp manifest (run from the repository root):

   ```bash
   grep -oE '[0-9a-f]{64}  \S+' revision/acdc_confirm/TIMESTAMP_MANIFEST.txt | sha256sum -c
   grep -oE '[0-9a-f]{64}  \S+' revision/followup/TIMESTAMP_MANIFEST.txt   | sha256sum -c
   ```

   Some listed files are git-ignored inputs (the split manifests under `recovered/followup/splits/`);
   extract them from the evidence archive first (see "Building").

2. Evidence archives:

   ```bash
   cd revision/evidence_freeze
   sha256sum primary-evidence-20261110.tar.gz final-supplement-20261110.tar.gz
   ```

   compare with the archive digests recorded in `revision/BDS_LITE_EVIDENCE_FREEZE.md`;
   `primary-evidence-20261110.sha256` and `final-supplement-20261110.sha256` list the digest of every
   file inside the archives.

3. OpenTimestamps proofs (`.ots`; client: `pip install opentimestamps-client`):

   ```bash
   ots info preregistration/acdc_confirmation.md.ots         # prints the file digest it commits to
   sha256sum preregistration/acdc_confirmation.md             # must equal that digest
   ots verify preregistration/acdc_confirmation.md.ots       # needs a local Bitcoin node
   ```

   The same applies to `revision/acdc_confirm/TIMESTAMP_MANIFEST.txt.ots`. Without a Bitcoin node,
   `ots info` prints the Bitcoin block of the attestation (block 974994); the merkle root it implies
   can be compared with that block's header from any block explorer. Supplementary S18 describes
   what the timestamp shows and what it does not. The proofs of the follow-up registration
   (`revision/followup_registration.md.ots`, `revision/followup/TIMESTAMP_MANIFEST.txt.ots`) hold
   only pending calendar attestations: the external timestamp of that plan was dropped before
   training (Supplementary S21). They are included as recorded.

## What is not included, and why

| Not included | Reason |
|---|---|
| The git history of the development repository | Holds confidential review material and, in earlier commits, figures with dataset images that may not be republished |
| Dataset images and annotations (ISIC 2018, ACDC, Synapse) | Third-party data; obtain them from the providers (`THIRD_PARTY_NOTICES.md`). The only dataset pixels in this repository are the three ACDC slices of Figure 3 (CC BY-NC-SA 4.0) and ISIC 2018 Task 1 reference-mask contours (CC0) |
| Files with verbatim reviewer comments: the earlier LaTeX response letters (`revision/response_to_reviewers.tex`, `revision/response_letter_v2.tex` and its copy in `revision/audit/manuscript_snapshot_20261107_pre_reorg/`) and the internal reviewer matrix (`revision/plan/reviewer_matrix.md`); also the superseded draft `revision/response_letter.tex` | Peer-review comments are confidential. The response letter (`revision/response.md`, `revision/response_to_reviewers.pdf`) answers each comment without quoting it |
| Model checkpoints and predicted label maps (including two prediction arrays of ACDC official test patients, `results/teacher_eval/*/preds_official_test.npz`) | Large binaries; the dataset terms do not address the redistribution of models or label maps derived from the images |
| Per-run result folders (`results/`, `outputs/`), apart from the files tracked here | Large; the per-case results the analyses need are in the evidence archives |
| Internal agent instructions, plans and status documents (`AGENTS.md`, `PLAN.md`, `README_revision.md`, the earlier root README, `archive/`, internal `docs/` reports, `results/final_manuscript_audit_v2/`, `revision/plan/`, `revision/STATUS.md`, and the copy of the agent instructions in `revision/evidence_freeze/history/`) | Working documents; several state withdrawn results |
| `revision/audit/git_whitespace_initial.log` | An 18 MB `git diff --check` log of the initial import that copies lines of many files, including local directory inventories; no analysis reads it |
| `revision/sources/isic_page.html`, `revision/sources/transunet/utils.py.txt`, the literature PDFs read by the table generator | Copies of third-party pages, code and papers; their SHA-256 digests stay in the receipts |
| `revision/release/README.md` | Superseded draft of this README |

No build step, test or reproduction step described here reads an excluded file. Some kept files
still name them, for example docstrings of `analysis/revision/make_final_assets.py` and `verify_final_assets.py`
and a comment in `revision/manuscript/main.tex` refer to `revision/plan/`, and
`revision/evidence_freeze/final_checksums.csv` lists five excluded files (`AGENTS.md`, `PLAN.md`, two
completion reports and the copy of the agent instructions), whose digests cannot be checked here.

`revision/audit/followup-git-log.txt` is included with one change: the 21 occurrences of a personal
e-mail address were replaced by `[email redacted]`.

## Building

`bash revision/build_revision.sh` regenerates the keys, tables and figures from the frozen outputs,
compiles the manuscript and supplement with `tectonic`, and renders the editor letter. It needs:

- **Python environment.** `revision/env_freeze.txt` is a partial inventory (PyTorch 2.11.0, CUDA
  13.0); `segmentation_models_pytorch` 0.6.0 is used by the teacher. The authors' `.venv` also
  resolves packages from external directories, so `pip install -e .` alone does not recreate it.
  A portable environment has not been validated (`revision/REPRODUCIBILITY_STATUS_20261110.md`).
- **Tools.** `tectonic`, `pdftotext` (poppler). `pandoc` is needed only for the response-letter
  step, which is skipped here.
- **Local inputs that are not tracked** (git-ignored):
  - per-case results and split manifests: 115 `results/revision_runs/*/per_case_test.csv`,
    48 `results/revision_runs/*/final_metrics.json` and 8 `recovered/followup/splits/*/*.json`.
    All are in the primary evidence archive, byte-identical:

    ```bash
    tar -xzf revision/evidence_freeze/primary-evidence-20261110.tar.gz --wildcards \
      'results/revision_runs/*/per_case_test.csv' 'results/revision_runs/*/final_metrics.json' \
      'recovered/followup/splits/*'
    ```

  - `revision/sources/literature/`: four published papers (TransUNet, Swin-Unet, EGE-UNet,
    LB-UNet) from which the published-comparison table reads its values; their SHA-256 digests are
    in `revision/lit_numbers.csv`. They are copyrighted and not redistributed.
  - `recovered/peerj-submission/main.tex`: the submitted manuscript source, used only to count the
    words of the submitted version. It is not included because it states the withdrawn ISIC2018
    results.
- **Response letter.** The letter steps need two local files with the reviewers' text, which are
  not distributed. Without them the build prints `SKIPPED` and leaves `revision/response.md`,
  `revision/PEERJ_REQUIREMENTS_CHECKLIST.md` and `revision/response_to_reviewers.pdf` as committed.

Regenerating the frozen outputs themselves (`analysis/revision/family_statistics.py` and related
scripts) needs the per-run results; retraining and inference need the datasets and a GPU. The
evidence archives support the statistical reanalysis described in Supplementary S16
(`revision/evidence_freeze/verify_snapshot.py`), not full retraining.

### Tests

```bash
PYTHONPATH=src:. python -m pytest -q
```

The tests read two git-ignored inputs:

- `tests/unit/test_split_integrity.py` reads the split manifests in `data/splits/`. They are
  byte-identical to `recovered/followup/splits/` from the primary evidence archive; after the
  extraction under "Building", run `mkdir -p data && cp -r recovered/followup/splits data/splits`.
  Without them, three of its tests fail.
- `tests/comparators/` loads LB-UNet, which `scripts/comparators/fetch_lbunet.py` downloads once at
  the pinned commit into `third_party/` and checks by SHA-256 (network access needed).

Run `ruff` inside a git checkout of this snapshot, so that `.gitignore` applies and the downloaded
`third_party/` code is not linted.

Two tests of `tests/revision/test_acdc_confirm_amendment1.py` are skipped with the reason
"requires the private development history": they read the frozen study files from commit `345cacd`
(`tests/revision/conftest.py`). The test file itself is unchanged; its SHA-256 is recorded in
`revision/acdc_confirm/amendment1/AMENDMENT_1.md`. One test of
`tests/revision/test_kd_multiclass.py` is skipped when `segmentation_models_pytorch` cannot be
imported; in the authors' current environment it is skipped for that reason.

### Scripts that read git history

None of these is used by the build or the tests above. Without the development history:

| Script | Git call | What happens in this snapshot |
|---|---|---|
| `scripts/acdc_confirm/amendment1/integrity_amended.py` | `git ls-tree` and `git show` on commit `345cacd` | Stops with a git error. Its record from the study, `revision/acdc_confirm/amendment1/integrity_amended*.json`, is included. The file is unchanged because its SHA-256 is registered in Amendment 1 |
| `analysis/revision/evaluate_official_isic.py` | `git show a6dc26c:analysis/revision/prereg_revision.md` | Stops with a git error before reading any data (it also needs the ISIC archives) |
| `analysis/followup/timestamp_manifest.py` | `git rev-parse`, `git ls-tree`, `git show` on the registration commit | Stops with a git error. It wrote `revision/followup/TIMESTAMP_MANIFEST.txt`, which is included |
| `analysis/revision/recover_predictions.py --execute` | `git show HEAD:<queue and protocol>` | Works in a git checkout of this snapshot (it compares the files with the snapshot commit); then needs checkpoints and a GPU |
| `analysis/revision/check_release_clean.py` | `git rev-list --objects`, `git hash-object` | Works on any commit, but compares against the restricted figures in `revision/restricted_local/`, which are not distributed; without them it reports that it cannot check content |

`scripts/revision/build_status.py` only prints a status text that mentions commit `a6dc26c`; it runs
no git command.

## Repository map

| Path | Contents |
|---|---|
| `src/bds_lite/` | Models, losses, data loading, training runner |
| `scripts/` | Trainers, queues and supervisors of each study (`revision/`, `expansion/`, `confirmation/`, `comparators/`, `kd/`, `kd_mc/`, `acdc_confirm/`, `kd_followup/`) |
| `analysis/` | Evaluation, statistics, tables, figures, letter tooling |
| `configs/` | Configuration of every run; their SHA-256 digests are registered in the queues and protocols |
| `splits/` | Locked split manifests (case identifiers only) |
| `preregistration/` | Analysis plan of the ACDC confirmatory study and its OpenTimestamps proof |
| `revision/` | Protocols, amendments, deviation logs, audits, frozen result tables, evidence archives, manuscript and letters |
| `revision/manuscript/` | Manuscript and supplement sources; every number is a generated key (`generated/`) |
| `tests/` | Unit and study tests |

## Licence and citation

The original code and text are under the MIT License (`LICENSE`). Third-party code, the ACDC images
of Figure 3 (CC BY-NC-SA 4.0) and the ISIC 2018 reference masks (CC0) keep their own terms:
`THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSES/`. Citation metadata: `CITATION.cff`.
