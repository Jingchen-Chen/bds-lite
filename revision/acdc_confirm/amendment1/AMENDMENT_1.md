# Amendment 1 to `preregistration/acdc_confirmation.md`: integrity-check source-manifest pairing

- **Adopted:** 2026-11-08, after all 34 training runs finished and **before either test set
  was accessed**. No model had been evaluated on official patients 101–150 or on the internal
  20-patient test split. `results/acdc_confirm_eval/` did not exist when this amendment was written.
- **Authorized by:** the author, 2026-11-08 ("Option 1: a documented pre-test protocol amendment"),
  with eight safeguards. Each safeguard is mapped to the implementation below.
- **Pre-registration:** unchanged. Its sha256 is still `8ff4509d0f35cc83f9ab38f12bb9ab593b1a2f12847e34786bdf6db78db9c768`,
  frozen in commit `345cacd` (tag `acdc-confirm-prereg-v3`).
- **Scope:** this amendment changes only how step 7.1 (integrity) is checked. It affects nothing
  else.

## 1. What failed

`PYTHONPATH=src:. .venv/bin/python scripts/acdc_confirm/integrity.py` was run at
2026-11-08T13:43Z, after `queue_finished`. It failed. For each of the 34 runs it reported
exactly one problem, verbatim:

```
acdc_confirm_<arm>_seed<N>: source hashes differ from registration
```

It reported no other problem in any run, and no global problem. The original output is kept
byte for byte as `revision/acdc_confirm/amendment1/integrity_original_failed.json`
(sha256 `72df1270f4cf2045df0ec4aa3d1839bf5a6ae497036a589377c8c935adc05f88`).

## 2. Cause

- **What the check compares.** `analysis/kd3/integrity.py::check_run` (frozen, unchanged)
  requires the run's `provenance.json["source_hashes"]` to be exactly equal to the sources
  registered for that run in `revision/queue_acdc_confirm.yaml`.
- **What the registration holds.** `scripts/acdc_confirm/register.py` registered 74 sources by
  design: the parent study's `kd_sources()` (61 files) plus the 13 top-level
  `scripts/acdc_confirm/*.py` study files.
- **What provenance records.** The frozen trainer `scripts/kd_mc/train_mc.py` records only the
  61 `kd_sources()` in `provenance.json`.
- **Result.** The two lists can never be equal, so this check could never pass for any run of
  this study. It is a defect in how the study code was paired with the frozen check. It is not
  evidence of a source change.

Evidence collected before this amendment (read-only):

- **The 61 recorded hashes match.** In all 34 runs, the 61 recorded paths and hashes are
  identical to the registered entries. No recorded entry is missing, extra or different.
- **The missing entries are exactly the 13 study files.** These are the registered paths that
  provenance does not record.
- **The 13 study files are unchanged since the freeze.** They show no diff against freeze commit
  `345cacd`.
- **The queue runner checked every source at every launch.** `scripts/kd_mc/run_queue_mc.py`
  (frozen) checks all 74 registered source hashes against disk before every run, in every one of
  the 55 queue launches. It refused none.

**Correction.** The diagnosis reported to the author on 2026-11-08 said "75 registered" and
"62 recorded". The correct counts are 74 and 61, found by the regression test
`test_real_registration_matches_the_freeze_commit_split`. Only the counts were wrong. The
mismatch is the same: the same 13 `scripts/acdc_confirm/*.py` files. The amended check
enforces the correct counts exactly.

## 3. The amendment (exact rule)

Step 7.1 is checked by `scripts/acdc_confirm/amendment1/integrity_amended.py`.

**Run every original check.** It first runs the unchanged
`scripts/acdc_confirm/integrity.py::check` and keeps its full result verbatim.

**Only one problem can be tolerated.** That problem is the exact string
`"<run>: source hashes differ from registration"` for a registered run. It is accepted for that
run only if all four conditions hold:

1. **The registration has the expected shape.** It has exactly 74 sources: 13 under
   `scripts/acdc_confirm/` and 61 elsewhere.
2. **The provenance matches the 61 non-study entries exactly.** The run's provenance records
   exactly those 61 paths, each with its registered sha256. There is no missing, extra or
   differing entry.
3. **The 13 study files match the freeze commit.**
   - The 13 registered study paths are exactly the top-level `scripts/acdc_confirm/*.py` files
     of commit `345cacd02855bc93105dd2c2e0a1730c564ea953`.
   - For each file, the sha256 in the registration equals the sha256 of the committed blob.
4. **Every registered file is unchanged on disk.** All 74 registered files on disk still have
   their registered sha256. This is stricter than the original check, which does not re-hash
   sources at integrity time.

**Everything else fails, as before.** This includes:
- any other problem in any run;
- any global problem, such as seed completeness, the equivalence check or a test output;
- the tolerated string with any variation;
- the tolerated string for an unregistered run;
- any failed condition above.

**There is no general bypass.** No original check is removed or weakened.

**Evaluation gate.** The one-time evaluation is run by
`scripts/acdc_confirm/amendment1/evaluate_amended.py`. That script:
1. confirms that `evaluate_test.check` is still the original integrity check;
2. re-verifies the pre-amendment snapshot (§5);
3. rebinds `evaluate_test.check`, for that process only, to the amended check;
4. calls the unchanged `evaluate_test.main()`.

`evaluate_test.main()` keeps everything it already did:
- its refusal if `results/acdc_confirm_eval/` exists (one evaluation only);
- the NPZ hash checks on the official test set;
- the exploratory checkpoint hash checks;
- the job list;
- the frozen `train_run.py::evaluate`.

Step 7.3 (post-test integrity) becomes `integrity_amended.py --post-test`.

## 4. What does not change

- **Design:** hypotheses, the primary endpoint (test Boundary F1, official patients 101–150),
  secondary analyses (a)–(d), metrics and SESOIs.
- **Statistics:** the estimator (`expansion_statistics.contrast` with patient units, unpaired,
  Welch, two-sided α = 0.05) and the decision rules of §11.
- **Runs:** N = 17 per arm, seeds 201–217, the failed-run policy, achieved power, and
  `analyze.py`.
- **Frozen code and data:** all frozen training code, checkpoints, configs, the queue
  registration, the test data and its preprocessing.
- **The 13 registered study files:** none is modified. The amendment code lives in a new
  subdirectory, so the registered top-level file set is unchanged. A regression test enforces
  this.

## 5. Safeguards (author's items 1–8) and implementation

| # | Safeguard | Implementation |
|---|---|---|
| 1 | No change to frozen training code, checkpoints or registered settings | New files only (`scripts/acdc_confirm/amendment1/`, `tests/revision/test_acdc_confirm_amendment1.py`, this folder). `git diff 345cacd -- scripts/acdc_confirm/*.py scripts/kd scripts/kd_mc scripts/revision src configs revision/queue_acdc_confirm.yaml` is empty. |
| 2 | Preserve and hash artifacts, registration and original integrity output first | `amendment1/snapshot.py write` hashed 609 files: every file in the 34 replication and 6 exploratory run folders, the queue, the 34 configs, the pre-registration, all 74 registered sources, the equivalence record, the official-test manifest and preprocessing record, and the copied original integrity output. Record: `pre_amendment_snapshot.json` (sha256 `43973f9a8002231fb364fa89125661d5b93ea155cb87b416a83e75aa7b02b495`). `snapshot.py verify` runs before the evaluation and again after it. |
| 3 | Separate wrapper, all original checks, only the documented mismatch | `integrity_amended.py` (§3) |
| 4 | 61 recorded hashes equal registered; 13 study files checked against the registration and `345cacd` | Conditions 2 and 3 (and 4) |
| 5 | Reject any other discrepancy; no general bypass | `amend()`: an exact string match plus conditions 1–4, otherwise a failure |
| 6 | Regression tests, valid and corrupted manifests | `tests/revision/test_acdc_confirm_amendment1.py`, 27 tests: <br>• 1 valid manifest accepted; <br>• 13 corrupted manifests rejected (recorded hash differs, missing or extra; a study file recorded; a study or other file changed or missing on disk; a study file differing from the freeze commit; extra or missing study files in the commit or registration; a registered study hash differing); <br>• 8 other-problem strings rejected; <br>• real-registration checks against `345cacd`; <br>• no top-level study file added; <br>• the evaluation wrapper does not rebind on import. |
| 7 | Document before accessing either test set | This file and the code are committed before `evaluate_amended.py` runs. `PROTOCOL_LOG.md` records the order. |
| 8 | Then the pre-registered evaluation and analysis, unchanged | `integrity_amended.py`, then `evaluate_amended.py` (one evaluation), then `integrity_amended.py --post-test`, then the unchanged `analyze.py` |

**Implementation file hashes** (sha256 at adoption):

| File | sha256 |
|---|---|
| `scripts/acdc_confirm/amendment1/__init__.py` | `610b5f211dc4bf7fe237c032dcf00fb4903ba23c0dd614a1c7cb42782fc84039` |
| `scripts/acdc_confirm/amendment1/snapshot.py` | `0a34090deca389b74dcf963587a6b4afa1681a8d0ce22dcf1e7bb535b7316077` |
| `scripts/acdc_confirm/amendment1/integrity_amended.py` | `d14e0a4e3b6bda965dd57a7b2daf99ac25d10d86710d5ec3040ca9103284dc11` |
| `scripts/acdc_confirm/amendment1/evaluate_amended.py` | `a6d8f70308901a2bd6dd0ad9c698f420bf93fe3198e3a3fdf904dfc8a8e4ca45` |
| `tests/revision/test_acdc_confirm_amendment1.py` | `39f84bfeae9b420ef44181f9df2d229722927f8924eb23c5d5a3484c65edbdf0` |

**Checks at adoption:**
- `ruff format --check .` and `ruff check .` pass.
- `pytest tests/unit` (the commit hook): 59 passed.
- Full `pytest`:
  - With the default temporary directory: 135 passed, 1 skipped, 1 failed. The failure is
    `tests/revision/test_thermal_supervision.py::test_only_current_run_with_recent_temperature_and_checkpoint_can_resume`,
    which is unrelated to this study.
    - **Cause:** the function under test treats less than 20 GB free on the test's temporary
      volume as "low disk". `/tmp` is a tmpfs with 12 GB free.
    - **Status:** the test was not modified.
  - With `--basetemp` on the main disk (265 GB free): 136 passed, 1 skipped.

## 6. Commands after adoption (run once, in this order)

```bash
PYTHONPATH=src:. .venv/bin/python scripts/acdc_confirm/amendment1/integrity_amended.py
PYTHONPATH=src:. .venv/bin/python scripts/acdc_confirm/amendment1/evaluate_amended.py
PYTHONPATH=src:. .venv/bin/python scripts/acdc_confirm/amendment1/integrity_amended.py --post-test
PYTHONPATH=src:. .venv/bin/python scripts/acdc_confirm/amendment1/snapshot.py verify
PYTHONPATH=src:. .venv/bin/python scripts/acdc_confirm/analyze.py
```

## 7. Reporting

This amendment is reported as a deviation in `revision/acdc_confirm/DEVIATION_CHECKLIST.md` and
in the study's final documentation.

**Separate open item: the OpenTimestamps proofs are not yet verified.** Both proofs carry a
`BitcoinBlockHeaderAttestation(974994)`, but they have not been checked against a Bitcoin block
header, because no node or author-chosen header source is available. Until that check is done,
they must not be described as independently verified.
