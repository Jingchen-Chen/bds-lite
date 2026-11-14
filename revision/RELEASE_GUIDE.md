# Release guide: public release of the PeerJ revision (v2)

> **Superseded procedure (2026-11-14).** The public release is now a single-commit snapshot made
> with `git archive` from branch `review-edits`, minus the paths in
> `revision/release/release_exclude.txt`, with `revision/release/PUBLIC_README.md` moved to
> `README.md` and the personal e-mail address in `revision/audit/followup-git-log.txt` redacted.
> `CITATION.cff`, `.zenodo.json`, `LICENSE`, `THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSES/`
> are now tracked at the repository root. What the snapshot leaves out, and why, is in that README.
> The steps below are the plan of 2026-11-08 and are kept as a record; the decisions in section 0
> still apply.

For the authors. Nothing here has been pushed or published; you do the push. Every step was
dry-run on 2026-11-08 in a scratch directory: the resulting single-commit tree (1,691 files,
167 MB) passed `check_release_clean.py`.

**Rule.** Build the release as a **new single-commit repository from the current tree**, never from
this repository's history. The branch `revision/peerj-major` contains removed dataset images in
earlier commits (`revision/LICENSING_AUDIT.md`), so it must never be pushed.

---

## 0. Decisions to make first

| # | Decision | Recommendation |
|---|---|---|
| D-a | **The existing public repository already exposes dataset images.** `github.com/Jingchen-Chen/bds-lite` at its current HEAD (`3104cb3`) contains `manuscript/BDS-Lite.pdf`, whose page 11 shows ISIC2018 photographs and ACDC and Synapse images with annotations. Its history (commits `1b0c4f6`, `6c4ba27`, `5319de3`) also contains `figures/failure_panels/*.png` and `figures/figure6_failure_cases.{png,pdf}` with the same kind of content. `revision/LICENSING_AUDIT.md` found no permission to republish these. | Publish v2 so that no reachable commit contains them: either replace the repository (delete and recreate it, then push the single commit), or force-push the single commit as the only branch and ask GitHub Support to purge cached views of the old commits. Pushing v2 on top of the old history would keep them reachable. `check_release_clean.py` now flags these paths (step 3). |
| D-b | Your uncommitted records: `revision/DEVIATIONS.md` (Experiment C entries), `revision/CONFIRMATION_SCOPE_AMENDMENT_20261107.md`, `revision/queue_plan.yaml`, `revision/queue_confirmation_acdc_synapse.yaml` | Commit them before step 1. Supplementary S15 describes the Experiment C stop that they record. |
| D-c | Experiment C test outputs (14 completed runs) | Keep them out of every public artifact. The results bundle in step 5 excludes their run folders; no analysis of them is frozen. |
| D-d | Model checkpoints and prediction arrays (`*.pt`, `preds_*.npz`) | Not included by default. The dataset terms do not address model weights or predicted label maps of ACDC and Synapse; add them to Zenodo only if you are satisfied that this is permitted. |
| D-e | `recovered/peerj-submission/main.tex` (the submitted source, with the withdrawn ISIC2018 results) | Included by default, because the word counts in the response letter are computed from it. Drop it if you prefer, and then also drop the word-count sentence from the letter. |
| D-f | Repository identity: same URL (`Jingchen-Chen/bds-lite`, as cited in the manuscript) or a new one | Same URL, recreated (D-a); otherwise update the Data and code availability statement. |

---

## 1. Prepare the source tree

```bash
cd /home/jc/bds-lite
git status --short          # only intended changes; commit your records (D-b) first
bash revision/build_revision.sh   # rebuilds keys, PDFs, pointers and letters; must end "build complete"
.venv/bin/ruff format --check . && .venv/bin/ruff check . \
  && .venv/bin/python -m pytest -q --basetemp="$HOME/.cache/bds_pytest"
```

## 2. Build the clean tree

```bash
SRC=/home/jc/bds-lite
REL=/home/jc/bds-lite-release-v3          # new, empty directory outside SRC
mkdir -p "$REL"
cd "$SRC"

# (a) tracked files, minus the release exclusions
git ls-files | grep -v -E -f <(grep -v '^#' revision/release/release_exclude.txt) > /tmp/release_files.txt
# (b) untracked files that the registered configs and the word count need
find recovered/followup/splits -name '*.json' | sort >> /tmp/release_files.txt
echo recovered/peerj-submission/main.tex >> /tmp/release_files.txt      # D-e
rsync -a --files-from=/tmp/release_files.txt "$SRC"/ "$REL"/

# (c) release front matter
cp revision/release/README.md     "$REL"/README.md
cp revision/release/CITATION.cff  "$REL"/CITATION.cff
cp /home/jc/bds-lite-release-v2/LICENSE "$REL"/LICENSE                    # MIT, as in the v1 release
cp -r /home/jc/bds-lite-release-v2/THIRD_PARTY_LICENSES "$REL"/
cp /home/jc/bds-lite-release-v2/THIRD_PARTY_NOTICES.md "$REL"/
```

**What goes in.**

| Included | Why |
|---|---|
| `src/`, `scripts/`, `analysis/`, `tests/`, `pyproject.toml`, `ruff.toml`, `.gitignore`, `.gitattributes` | Code and tests |
| `configs/` | Every run's configuration (their sha256 values are registered in the queues) |
| `splits/`, `recovered/followup/splits/**.json` | Split manifests (identifiers only). The second path is referenced by the configs and must keep its location. |
| `preregistration/acdc_confirmation.md`, `preregistration/acdc_confirmation.md.ots` | Pre-registration and its OpenTimestamps proof |
| `revision/` (tracked files) | Protocols and amendments; deviation logs; audits; validation records; frozen result tables (`revision/tables/`, `revision/kd_unified/`, `revision/acdc_confirm/results/`, `revision/prediction_recovery/`, ...); `revision/acdc_confirm/TIMESTAMP_MANIFEST.txt` and `.ots`; manuscript, supplement, figures and letters; claim boundary; this guide |
| `docs/devlog.md`, `docs/dataset_format.md` | Provenance of alpha and the loss weights; data format |
| `README.md`, `CITATION.cff` (from `revision/release/`), `LICENSE`, `THIRD_PARTY_LICENSES/`, `THIRD_PARTY_NOTICES.md` | Front matter |

**What stays out** (`revision/release/release_exclude.txt`, plus everything untracked).

| Excluded | Why |
|---|---|
| `AGENTS.md`, `PLAN.md`, `README_revision.md`, the old `README.md` | Internal agent instructions and plans; the old README states withdrawn results |
| `archive/` | The BSPC-era anchor carries the withdrawn ISIC2018 numbers |
| `docs/` except the two files above | Stale internal reports with withdrawn claims |
| `results/final_manuscript_audit_v2/` | Internal phase reports |
| `revision/sources/isic_page.html`, `revision/sources/transunet/utils.py.txt` | Third-party page and code copies; their hashes stay in the receipts |
| `revision/response_letter.tex`, `revision/response_letter_v2.tex` | Superseded drafts of the response letter |
| `data/`, `outputs/`, `results/` (except via the Zenodo bundle), `third_party/`, `revision/restricted_local/`, `revision/sources/literature/`, `*.pt`, `*.npz`, `*.npy` | Dataset content, unlicensed upstream code, restricted figures, copyrighted papers, binaries (D-d) |

**Update `THIRD_PARTY_NOTICES.md`** (it covers only the GSL port today). Add:
- *Boundary Loss components* (`scripts/comparators/components.py`): adapted from
  <https://github.com/LIVIAETS/boundary-loss> at `171c32d`, MIT License; add the upstream copyright
  notice and licence text to `THIRD_PARTY_LICENSES/`.
- *EGE-UNet* (`src/bds_lite/models/ege_unet.py`): ported from <https://github.com/JCruan519/EGE-UNet>
  at `f87e3df`, MIT License; add its notice and licence text likewise.
- *segmentation_models_pytorch* 0.6.0 (MIT) and the ResNet-34 ImageNet weights
  (`smp-hub/resnet34.imagenet`): used by the external teacher, installed or downloaded at run time,
  not redistributed.
- *LB-UNet*: no licence upstream; fetched at run time by `scripts/comparators/fetch_lbunet.py`, not
  redistributed.
- Replace the pointer to `docs/data_access.md` with a pointer to the README's data section, or copy
  that file over from the v1 release.

## 3. Commit once and check

```bash
cd "$REL"
git init -b main
git add -A
git commit -m "BDS-Lite revision release v2 (PeerJ CS-2026:06:142937)"
cd "$SRC"
.venv/bin/python analysis/revision/check_release_clean.py HEAD --repo "$REL"   # must print: HEAD: clean
```

`check_release_clean.py` scans every object reachable from the release commit, comparing contents
with `revision/restricted_local/` and paths with its marker list, which now includes the
public-history files of D-a. Also check that `git -C "$REL" log --oneline` shows one commit, and that
no file exceeds GitHub's 100 MB limit: `find "$REL" -size +95M`. The largest file is
`revision/error_analysis/per_image_class.csv`, at 28 MB.

## 4. Verify the clean tree

```bash
cd "$REL"
python -m venv .venv && .venv/bin/pip install -e '.[dev]'   # or reuse the frozen environment
.venv/bin/ruff check . && .venv/bin/python -m pytest -q
bash revision/build_revision.sh       # needs tectonic; must end "build complete"
git status --short                    # the rebuild must not change any tracked file except PDFs/logs
```

If `git status` shows changed generated files after the rebuild, stop: the release would not
reproduce its own manuscript.

## 5. Zenodo: results bundle (and, optionally, binaries)

Build the per-case results bundle. It is needed to recompute the frozen tables and excludes
Experiment C (D-c):

```bash
cd /home/jc/bds-lite
.venv/bin/python - <<'PY'
import glob, json, os, tarfile
st = json.load(open("revision/audit/experiment_c_blinding_check/experiment_c_status.json"))
expc = {r["run_id"] for r in st["runs"]}
files = []
for d in sorted(glob.glob("results/revision_runs/*/")):
    if os.path.basename(d.rstrip("/")) in expc:
        continue
    for pat in ("per_case_*.csv", "final_metrics.json", "config.json", "provenance.json", "history.json"):
        files += glob.glob(d + pat)
files += glob.glob("results/acdc_confirm_eval/**/per_case_*.csv", recursive=True)
files += ["results/acdc_confirm_eval/evaluation_record.json"]
files += glob.glob("results/runs/*p1b*/metrics.jsonl")
files += ["results/midterm_q2_rescue/analysis/per_case_metrics.csv"]
assert not any(r in f for f in files for r in expc)
with tarfile.open("/home/jc/bds-lite-results-v2.tar.gz", "w:gz") as tar:
    for f in sorted(files):
        tar.add(f)
print(len(files), "files")
PY
sha256sum /home/jc/bds-lite-results-v2.tar.gz
```

On 2026-11-08 this selection held 1,224 files (about 190 MB before compression).

## 6. GitHub release

1. Publish the single commit (D-a, D-f). For example, after recreating the repository empty:
   `git -C "$REL" remote add origin https://github.com/Jingchen-Chen/bds-lite.git && git -C "$REL" push -u origin main`.
2. Tag it: `git -C "$REL" tag -a v2.0.0 -m "Revision release for PeerJ CS-2026:06:142937" && git -C "$REL" push origin v2.0.0`.
3. On GitHub, open Releases, choose "Draft a new release" with tag `v2.0.0`, and give it a title and
   notes. The notes should cover: what changed since v1, the withdrawn ISIC2018 results, the
   pre-registration and its timestamp scope, and no dataset content. Attach
   `bds-lite-results-v2.tar.gz` and its sha256. Publish.

## 7. Zenodo version and DOI

- **If a Zenodo record of v1 exists:** open it, choose "New version", upload the GitHub release archive
  (Source code, tar.gz) and `bds-lite-results-v2.tar.gz`, set the version to `2.0.0`, update the
  description (as in the release notes) and the related identifier (the GitHub release URL), then
  publish. Use the **version DOI**, not the concept DOI.
- **If no record exists** (the v1 deposit was planned but may never have been minted): either create a
  new upload with the same files, or enable the GitHub–Zenodo integration *before* step 6.3. In that
  case Zenodo archives the release automatically, and you add the results bundle as a further
  version.

## 8. Put the DOI into the manuscript and rebuild

1. In `revision/manuscript/main.tex`, Data and code availability, replace
   `\authordecision{add the Zenodo DOI ...}` with `(doi:10.5281/zenodo.NNNNNNN)`.
2. In `revision/release/CITATION.cff` and `revision/release/README.md`, fill in the same DOI.
3. Run `bash revision/build_revision.sh` and rebuild the tracked-changes PDF:
   `.venv/bin/python analysis/revision/make_tracked_changes.py <path to latexdiff-so 1.3.4>`
   (download and checksum as described at the top of that script).
4. Commit in this repository, and copy the updated `README.md`, `CITATION.cff` and manuscript files
   into the release repository as a second commit. Run step 3's check again. Only if you want the
   DOI inside the archived snapshot itself, make a `v2.0.1` release and a new Zenodo version.
5. Submit to PeerJ: `main.pdf` with the LaTeX sources, `supplement.pdf`, `tracked_changes.pdf`,
   `response_to_reviewers.pdf`, and the editor letter. Before resubmitting, check the authors,
   affiliations, funding, and competing-interest entries in the submission system: PeerJ uses only
   those entries, and they cannot be changed after acceptance.
