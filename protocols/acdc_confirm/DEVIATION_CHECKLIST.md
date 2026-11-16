# Deviation checklist: held-out ACDC replication (pre-registration §7.6)

Checked against `preregistration/acdc_confirmation.md` (sha256 `8ff4509d…c768`), frozen at
`345cacd` (tag `acdc-confirm-prereg-v3`). Amendment 1, adopted before any test-set access, is
committed as `885ce41`. Compiled 2026-11-08 after the analysis. Each event below has a UTC entry
in `revision/acdc_confirm/PROTOCOL_LOG.md`.

**Status key:**
- **✓**: as pre-registered.
- **DEV**: a deviation (listed in §A below).
- **NOTE**: no deviation, but worth disclosing.

## A. Deviations

| ID | Section | What happened | Effect on the result |
|---|---|---|---|
| **D1** | §7.1, §7.2, §7.3 | **Amendment 1: the integrity check was paired with the registration incorrectly.** The pre-test `integrity.py` failed for all 34 runs, and only on `source hashes differ from registration`. **Cause:** the queue registers 74 sources (61 `kd_sources()` plus the 13 `scripts/acdc_confirm/*.py`), while the frozen trainer records only the 61 in `provenance.json`; the frozen `check_run` requires exact equality, so it could never pass. **What was done:** the author authorized a pre-test amendment, `revision/acdc_confirm/amendment1/AMENDMENT_1.md`, committed in `885ce41` before either test set was touched. **The wrapper:** it runs every original check and accepts only that exact problem. It accepts it only if: the 61 recorded hashes equal the registration; the 13 study files equal the registration and commit `345cacd`; all 74 registered files are unchanged on disk. Any other discrepancy fails. 27 regression tests cover it. **What was preserved first:** a snapshot of 609 files, including every run artifact, and a byte copy of the original failed output. **Outcome:** the amended check passed before and after the test evaluation; the original check, run inside the wrapper, reproduced its output byte for byte. The evaluation itself was the unchanged `evaluate_test.main`, run once. | None on any number. The hypotheses, endpoint, estimator, metrics and decision rules are unchanged. The check is stricter, not looser: the original does not re-hash sources at integrity time. |
| **D2** | Reporting (D1) | **A count was misreported.** The diagnosis first given to the author said "75 registered / 62 recorded". The correct counts are 74 / 61, found by a regression test before the amendment was committed. The mismatch is the same 13 files. Corrected in `PROTOCOL_LOG.md` (2026-11-08T13:58:09Z), `PLAN.md` and `AMENDMENT_1.md`. | None. |
| **D3** | §4(c) | **The cross-arm seed correlation was missing from the outputs.** `analyze.py::paired` computes it, but `analyze.py::row` does not write it to `contrasts.csv` or `results.json`. **What was done:** `scripts/acdc_confirm/post/seed_correlation.py` (new; `analyze.py` unchanged) calls the unchanged `analyze.paired` on the same inputs. It checks that Δ, the interval bounds and p equal the written seed-paired rows (to 1e-12), and writes `revision/acdc_confirm/seed_correlation_supplement.json`. | None. It is a descriptive value only. |

## B. Item-by-item check

| § | Pre-registered item | Status | Evidence |
|---|---|---|---|
| Header | Freeze commit; the file is never edited after the freeze | ✓ / NOTE | Freeze `345cacd`. The file's sha256 is unchanged; it is hash-checked at every launch and by the snapshot. **NOTE:** its first line still reads "FINAL DRAFT v3 … Not yet committed". That line was written just before the freeze and, by rule, was not edited afterwards. |
| Header | Every deviation logged with UTC time and reason | ✓ | `PROTOCOL_LOG.md` |
| §1 | Selection rationale stated; the pooled cross-dataset interval is not used | ✓ | It is not used anywhere in the analysis or the report. |
| §2 | Official patients 101–150 from CREATIS; integrity PASSED; preprocessing gate passed | ✓ | `official_test_integrity.json`, `official_test_preprocessing.json` (all before the freeze) |
| §2 | No model evaluated on 101–150 before the freeze | ✓ | The first model access to the test sets was the one evaluation, at 2026-11-08T13:58:47Z. |
| §2 | Per-NPZ hashes checked again at evaluation | ✓ | `evaluate_test.official_loader` (unchanged) checked all 1,076 NPZ files. |
| §2 | Fallback | ✓ (not used) | `PRIMARY_SET = "official_test"` |
| §3 | One primary endpoint (Boundary F1, 50 official patients, seeds 201–217), frozen unpaired estimator, two-sided α 0.05 plus Δ > 0 | ✓ | `analyze.py` unchanged. `expansion_statistics.py` sha256 `17099233…` equals the registration. |
| §4 | Secondary analyses (a)–(d), per-structure results and the "also reported" items | ✓ / DEV D3 | `results/contrasts.csv`, `per_class.csv`, `per_seed.csv`, `results.json`, plus the seed-correlation supplement |
| §4 | No secondary result changes or replaces the primary decision | ✓ | |
| §5 | N = 17 per arm, seeds 201–217; seeds 1–3, 4–10 and 100 not used | ✓ | 34 runs; registered configs only |
| §5 | Pairing (same initial weights and batch order per seed) | ✓ | Frozen launcher path; equivalence and unit tests (§9) |
| §5 | Run order seed by seed, decoder-only then D+KD-band | ✓ | The order in which runs completed matches the registered order exactly. |
| §5 | Failed-run policy | ✓ | 54 thermal epoch-stops (55 queue launches), each resumed automatically from its saved epoch. No non-thermal halt, no relaunch, no run declared failed (`failed_runs.json` absent). |
| §6 | N justified; cost about 15.5 GPU-hours | ✓ / NOTE | Wall clock 2026-11-07T22:09:31Z to 2026-11-08T13:43:02Z, i.e. 15.56 h. Summed epoch time: 7.63 h (D+KD-band) + 7.52 h (decoder-only). |
| §7.1 | Integrity checks | DEV D1 | The original check failed on one defect only, and every other check passed. The amended check passed (`amendment1/integrity_amended.json`). Feasibility: L_kd decreased in every D+KD-band run. |
| §7.2 | One test evaluation; the frozen evaluator and loader; outputs only under `results/acdc_confirm_eval/` | ✓ (via D1) | 74 jobs: 34 + 6 official, 34 internal. `evaluation_record.json`. The snapshot verified that the run folders were unchanged after the evaluation. |
| §7.3 | Post-test integrity | ✓ (via D1) | `amendment1/integrity_amended_post_test.json`: passed. |
| §7.4 | Analysis once; achieved power | ✓ / NOTE | Achieved power equals the planned power, since no run failed: 0.803 at +0.015 and the 90% bound. **NOTE:** the registered output folder `revision/acdc_confirm/results/` matches the repository `.gitignore` pattern `results/`, so its four files were added with `git add -f`. |
| §7.5 | Readings against the SESOI | ✓ | The `reading` column, from the frozen `ex.classify` |
| §7.6 | This checklist | ✓ | |
| §8 | Frozen items unchanged | ✓ | `git diff 345cacd` is empty for `scripts/kd*`, `scripts/revision`, `src`, `configs`, `analysis`, the 13 study files and the queue. Teacher `best.pt` `735747…dafa0` and cache `42266e…077e` were re-hashed on 2026-11-08 and match. No augmentation was used (enforced by `require_protocol`). |
| §9 | Implementation as registered; no existing file edited | ✓ / NOTE | Files added after the freeze: `scripts/acdc_confirm/amendment1/*` (D1), `scripts/acdc_confirm/post/seed_correlation.py` (D3), `tests/revision/test_acdc_confirm_amendment1.py`. |
| §10 | Monitoring limited to training losses, validation and telemetry; no test data during training | ✓ | |
| §11 | Decision wording fixed in advance | ✓ | Success, "of unclear size": the 95% lower bound is +0.0098, which is above 0 and below +0.01. See `STUDY_REPORT.md`. |
| §12.1 | No evaluation on 101–150 before the freeze | ✓ | |
| §12.2 | No test look before integrity; one evaluation | ✓ | `evaluate_test.main` refuses a second run. |
| §12.3 | No seed changes or interim analysis | ✓ | |
| §12.4 | Endpoint, set, estimator, α, sidedness and thresholds unchanged | ✓ | |
| §12.5 | §8 items unchanged | ✓ | |
| §12.6 | Teacher and cache not retrained, modified or evaluated | ✓ | Hashes as above; the teacher is not in any evaluation job. |
| §12.7 | No exclusions | ✓ | |
| §12.8 | No KD-full; ISIC2018 and Synapse not extended | ✓ | |
| §12.9 | Experiment C (ACDC seeds 4–10) not read | ✓ | |
| §12.10 | Pooled cross-dataset interval not used as evidence | ✓ | |
| §13 | Limitations | ✓ | Restated in `STUDY_REPORT.md` |

## C. Open provenance item (separate from the protocol)

- **O1: the OpenTimestamps proofs are not yet verified.** Both proofs carry
  `BitcoinBlockHeaderAttestation(974994)`. They have **not** been verified against a Bitcoin
  block header: no node or author-chosen header source was available, and `ots verify` failed
  with "Could not connect to Bitcoin node". Until the merkle root has been checked against the
  header of block 974994, the timestamp must not be described as independently verified.
- **Context:** using OpenTimestamps in place of the push the author first asked for was the
  author's decision (2026-11-07). The pre-registration itself requires neither.

## D. Environment note (not a protocol item)

- **One unrelated test fails with the default temporary directory.** It is
  `tests/revision/test_thermal_supervision.py` (Experiment C). The code under test treats less
  than 20 GB free on the test's temporary volume as "low disk", and `/tmp` is a tmpfs with
  12 GB free.
- **Workaround and status:** with `--basetemp` on the main disk, the full suite passes. The test
  was not modified.
