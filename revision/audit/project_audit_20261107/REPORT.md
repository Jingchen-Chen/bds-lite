# BDS-Lite project audit and proposed storyline

Date: 2026-11-07. Author-facing working document. Nothing was committed, pushed or published. No training was started or interrupted.

**Correction, same day.** Sections 6.1, 6.4 and 6.5 first gave run-level Welch intervals with no test-unit error. They now use the frozen expansion estimator, as the manuscript does. This widened the intervals and changed two readings: B-C and C-M2 on development images now include zero, so the gate-rescue reading in 6.4 is weaker than first stated, and BDS-Lite below decoder on validation excludes zero but is not Holm-significant.

**Status.** The author chose to run the manuscript reorganization and the historical re-evaluation in parallel with Experiment C. See `revision/HISTORICAL_REEVALUATION_PROTOCOL.md`.

**Key to names used throughout.** Family O (or "April grid"): the 13-arm grid behind the submitted models. S0/S6: its U-Net and BDS-Lite. M0-M7: the retrained arms (M0 U-Net, M1 surface only, M2 decoder only, M3 BDS-Lite without surface, M4 without distillation, M5 without gate, M6 full, M7 decoder + surface). October arms: U, S, B, F (= M0, M1, M3, M6 on ISIC seeds 1-3), G (gate only), C (gate + decoder, no distillation), D (gate + feature matching, no boundary supervision), UA (U-Net + D4 augmentation). Stage 1: the 40-run expansion. Stage 2: Experiment C, 70 runs, in progress. BF1: Boundary F1. SESOI: the registered thresholds (0.01 DSC/IoU/BF1, 2 px HD95, 0.5 px ASSD).

---

## 1. Bottom line

1. **Scale.** 536 run records: 425 completed training runs, which reduce to about 379 distinct models by tensor hash and about 358 independent training realisations. Another 70 are Stage 2 (7 finished, 1 running, 62 queued, all unread), 24 were withdrawn before starting, 12 are short reproduction gates, and 5 are smoke, failed, aborted or verification runs. About 40 further studies were evaluation-only.

2. **The ISIC leak is now traced to its root, and a clean estimate existed all along.** On 21 May, `scripts/reeval_main_seeds.py` hard-coded ISIC to the 519-image `val` set to apply a genuine BF1 bug fix. That replaced a valid 390-image held-out test with a set that was 69% training images. The favourable HD95 result in the submission came entirely from those training images: +1.76 px on the 358 training images, +0.09 px on the 77 held-out ones, and about zero on the clean 390 (Section 5.1).

3. **The confirmed core stands.** I recomputed all six primary Stage 1 contrasts from raw per-image files and they match to four decimals. BDS-Lite below decoder (-0.032) and GSL below decoder (-0.031) on ISIC are confirmed. BDS-Lite below decoder also has the same direction on validation images, with an interval excluding zero, though not Holm-significant there.

4. **Four things in the 7 November manuscript should change.**
   - "Clear ordering" with the decoder on top is stronger than the evidence. The decoder gain is +0.021 on the official test but +0.003 on in-distribution validation (interval inside SESOI), and it has the opposite sign in the April grid.
   - "Fixed before the experiments" for alpha and the loss weights is contradicted by the devlog. The original submission's wording, "before the matched evaluation", was accurate.
   - "No 2.5D" is not quite true: a three-slice ACDC control exists.
   - The low retrained U-Net affects Synapse as well as ACDC, and the text only says ACDC.

5. **Recommended storyline.** Keep the title and make the paper literally answer it. Organise around the conditions under which boundary distillation appears to help: relative to what, in which recipe, on which task, on which error, and under what standard of evidence. **The answer: we could not establish a condition in which distillation adds to simple training-only supervision, and where it appeared to help, that depended on the reference, the evaluation images or the number of trainings.** The mechanism explains why: the transfer succeeds in a projection and leaves the features that make the prediction unchanged.

6. **Timing.** Stage 2 should finish in about 39 hours at the observed 37 minutes per run, longer with thermal pauses. It directly tests the decoder claim, so I recommend fixing the outline now and rewriting after unblinding. Section 11 gives the branches, fixed by the protocol before any result is known.

---

## 2. Method and guardrails

**Read:** `AGENTS.md`, root `PLAN.md`, `revision/plan.md`, all revision protocols and audits, `docs/` (devlog, project state, ISIC blocker, publishability review), `archive/` status and manuscript drafts, the October research tree (38 decisions, 43 protocols), the decision letter, the submitted manuscript, and the current `main.tex`.

**Computed:** a run ledger from the filesystem; re-derivation of frozen contrasts; new post hoc analyses from existing per-image files. Scripts and outputs are in this folder.

**Stage 2 blind kept.** Stage 2 run IDs were taken from `queue_confirmation.yaml` and recorded by file existence only. No Stage 2 metric, history or per-case file was opened. I also checked that the refresh chain, which rewrote `numbers.tex` at 08:32, reads only seeds 1-3 and the seed-1 diagnostics.

**Evidence classes used below**

| Class | Meaning | May support conclusions? |
|---|---|---|
| V | Completed, registered or frozen, independently recomputed | Yes |
| D | Completed and re-inferred, but on development images only | Yes, for mechanism and scope, not test claims |
| P | New post hoc analysis of V or D runs | Descriptive; must be labelled post hoc |
| H | Completed historically, not validated under the current evaluator | No. May raise questions only |
| X | Invalid or withdrawn | No |
| B | Blind, in progress | Not read |

**Governance gaps.** `manuscript_v2/ANCHOR_ABSTRACT.md` is not at its documented path. The only copy is `archive/manuscript_v2_bspc/`, it is the BSPC-era anchor, and it carries the contaminated 0.5520/0.5597 numbers, so it cannot serve as a claim boundary. `archive/training_logs_april/`, named in `AGENTS.md`, does not exist. Root `PLAN.md` has no phase for this work, so it was not edited. I did not overwrite the `results/final_manuscript_audit_v2/` reports that `AGENTS.md` asks for at the end of multi-phase tasks, because they hold earlier phases' records.

---

## 3. Experimental inventory

Full run-level record: `experiment_ledger.csv` (536 rows: path, era, family, dataset, arm, seed, epochs, status, duplicate target, partition, evaluation, evidence class, manuscript use).

### 3.1 Totals

| Category | Records |
|---|---:|
| Completed training runs | 425 |
| Stage 2, blind (7 finished, 1 running, 62 queued) | 70 |
| Withdrawn before start | 24 |
| Reproduction gates (4 epochs, no test evaluation) | 12 |
| Smoke runs | 2 |
| Failed, aborted or verification-only | 3 |
| **Total** | **536** |

From 425 completed records to independent realisations:

| Step | Removed | Left |
|---|---:|---:|
| p1 and p1b variants that are the same checkpoint evaluated differently (30, of which 2 are in fact distinct) | 28 | 397 |
| April aliases: `unet_boundary_loss` = `bds_lite_boundary_only` (9), ISIC `gate_surface_class_weighted` = `gate_surface` (3) | 12 | 385 |
| October U and F = p1b ISIC models | 6 | **379** distinct by tensor hash |
| `bds_lite_surface_only` = `unet_surface_loss` in every metric (functional duplicate) | 9 | 370 |
| November GSL seeds 1-3 reproduce Phase 16 exactly | 9 | 361 |
| WP5 diagnostics replay seed-1 runs | 3 | **about 358** |

### 3.2 Training families

| # | Family | Dates (2026) | Runs | Design | Partition and evaluation | Motivation | Outcome | Class | In 7 Nov manuscript |
|---|---|---|---:|---|---|---|---|---|---|
| 1 | Single-seed exploration | Apr 26-27 | 37 | Mostly Synapse; seed 2026, plus 2027/2028 for two arms | ISIC 2,075/519; validation only | Find a recipe that beats U-Net on HD95/ASSD | Fixed lambda_b 0.05, lambda_d 0.05, lambda_s 1, alpha 0.25. Synapse distances never fixed; HR rescue rejected | Exploratory, tuned on validation | No, and it is the true provenance of the hyperparameters |
| 2 | April grid (family O) | Apr 28 - May 3 | 117 | 13 arms x 3 datasets x 3 seeds | **ISIC 1,815/389/390**; ACDC, Synapse locked; trainer commit 5914d35 lost | "Reproducibility grid" | U-Net and BDS-Lite became the submitted models. The other 11 arms were never reported | S0/S6 V after re-evaluation; rest H | S0/S6 only |
| 3 | p1 gate pilot | May 7 | 15 dirs | Gate and stop-gradient variants, single seed | Locked | Is the gate needed? | Negligible effect of removal | Exploratory | No |
| 4 | p1b | May 7-8 | 36 dirs (18 U/full + 18 variants) | 3 seeds | Locked; ACDC full class-weighted | Multi-seed gate check | Basis of the retrained set | V | Yes (M0, M6, R6) |
| 5 | p2 baselines | May 8 | 18 | GSL, Boundary DoU | Locked | Recent loss baselines | GSL strong; DoU ACDC flagged; trajectories differ from Phase 16 | H, superseded | No |
| 6 | p2b | May 8-9 | 9 | BDS-Lite (no gate) + GSL | Locked | Is the path complementary to GSL? | No gain over GSL (BF1 -0.002, +0.006, -0.004) | H | No |
| 7 | EGE-UNet | May 22 | 9 | 3 x 3, about 55K parameters | ISIC on 519 | Modern compact baseline | Far below U-Net on ACDC and Synapse | H, dropped | No |
| 8 | Phase 16 GSL | Jun 9-10 | 9 | 3 x 3 | Locked; ISIC reported on selection set | Matched comparator after OOM blocker | Submitted rows; checkpoints lost; later reproduced exactly | V via retrain | Via retrain |
| 9 | October ISIC attribution | Oct 17-18 | 24 | U, S, B, F, G, C, D, UA x 3 | Locked; validation only, by protocol | Which component matters? | Frozen screens failed; B-C positive; D4 raises means | D | U, S, B, F only |
| 10 | October ACDC controls | Oct 19-20 | 15 | Index-grid vs paper-GSL; three-slice context; loss reduction unit | Recovered 80-case data; 10 val patients; test untouched | Stronger baselines | All closed by frozen screens, with directional gains retained | D | No |
| 11 | Registered revision runs | Nov 2-5 | 96 | M1-M5, C1, C3, C4, GSL, WP5, alpha, lambda | Locked; official ISIC test | Reviewer ablations and comparators | Main three-seed evidence | V | Yes |
| 12 | Stage 1 expansion | Nov 5-7 | 40 | ACDC M2 fix (3), M7 (9), ISIC seeds 4-10 x 4 arms (28) | Locked; official test; frozen protocol | Resolve ISIC BF1; fill the factorial | Tier 1 not met; two confirmed contrasts | V | Yes |
| 13 | Stage 2 Experiment C | Nov 7- | 70 | ACDC, Synapse seeds 4-10 x 4 arms; ISIC U-Net, decoder seeds 11-17 | Frozen protocol | Confirm the decoder; test BDS-Lite vs decoder elsewhere | **Blind** | B | No, by instruction |

Full classification of the 425 completed runs:

| Type | Runs |
|---|---:|
| Exploratory or development | 91 (families 1, 3, 9, 10) |
| Matched and reported in the original submission | 27 (S0/S6, Phase 16) |
| Completed but never reported | 135 (99 April arms, 9 EGE, 18 p2, 9 p2b) |
| Variants and references for the retrained set | 36 (p1b) |
| Registered revision runs | 96 |
| Frozen expansion, of which 28 are fresh confirmation seeds | 40 |

### 3.3 Evaluation-only studies (no training)

| When | Study | Scale | Result | Class |
|---|---|---|---|---|
| May 13 | Post-recovery evaluation | 48 checkpoints | Reproducibility check; p=0.026 unsupported | Historical |
| May 21 | BF1 fix and re-evaluation | 27 checkpoints | **Fixed a real bug; introduced the ISIC contamination** | X for ISIC |
| May 21 | Seed-1 per-image audit | 3 datasets | "First significant positive" on Synapse: image-level, one seed | H, superseded |
| Jun 6 | Midterm per-case analysis | 2,453 cases | Subgroups, failure panels; ISIC part contaminated | X for ISIC |
| Oct 18 | BF1 tolerance audit (1/2/4 px) | 10,899 masks | B-C positive at all tolerances; others vary | D |
| Oct 18 | Operating point, complete misses, TTA | 6 runs | Simple offset route closed; TTA rescued no empty prediction | D |
| Oct 18 | ISIC identity audit | 2,594 images | 16 known components cross the old split (29 train, 17 val images); 2,204 patients unknown | D |
| Oct 18-20 | About 20 ACDC, annotation, loss and quality audits | Various | Closed routes; physical geometry unverified | D |
| Nov 2 | Official ISIC evaluation | 18 cohorts, 9,900 maps | Replaces the 519-image evaluation | V |
| Nov 2 | Prediction recovery | 73 cohorts, 45,042 maps | Independently validated | V |
| Nov 2 | Volume, error, surface CDF, gate identity | 20, 33, 21 cohorts; 202,444 rows | Generated analyses | V |
| Nov 3-5 | Resource profile; gradient residual audit | Idle GPU | Latency +8%, not +24% | V |
| Nov 5-7 | Transfer probe, gate structure, precision/recall, seed subsets | Seeds 1-3 and 1-10 | Mechanism and repeatability | P |

### 3.4 Failed, abandoned and duplicated work

| Item | What happened | How to treat it |
|---|---|---|
| E1 long-schedule GSL (May 10) | GSL term inactive; stopped; CE+Dice artifact remains | Invalid. Do not use |
| GSL rerun OOM (June) | Pairwise distance fallback needed 8.66 GiB | Fixed by exact SciPy EDT; legitimate |
| First October launch | AMP overflow check too strict; 0 runs finished | Fixed by amendment; reruns from scratch |
| ACDC M2 seeds 1-3 | Class-weighted by mistake | Superseded by unweighted runs; no Holm decision flipped |
| `acdc_m1_sdf_only_seed1` | Class-weighted, ran before withdrawal | Sensitivity only |
| 24 withdrawn registrations | ACDC class-weighted arms, alpha/lambda trimming | Never run; keep the record |
| MedSAM, MedSAM2 | Weights downloaded; inference never run | Not an experiment |
| Thermal stops, unrelated GPU job | Resumed from saved epochs | Not failures |
| April rescue attempts (loss-only, LCC, init-from-U-Net, two-stage, HR) | All failed the Synapse distance gate | Real negative history, single-seed, validation |
| `docs/gate_analysis_writeup.md` | Calls p1b gate-removed models separately trained | Wrong: same checkpoints. Never repeat this |

---

## 4. Research timeline

| Date | Event | Why | Result | Consequence |
|---|---|---|---|---|
| Apr 26 | Repository starts; baselines | Month 1 plan | U-Net references on validation | Single-seed habit starts |
| Apr 26-27 | 37-run exploration, Synapse first | v1 BDS-Lite lost to U-Net on Synapse | Sequential tuning on validation: alpha 0.5 to 0.25, lambda_b 0.1 to 0.05, surface 0.05 to 1 | Recipe chosen on validation; ISIC called "clean positive" on one seed |
| Apr 28 - May 3 | 13-arm grid | Reproducibility | Run on an older ISIC partition, with its own 390-image test | **Partition mismatch born** |
| May 7-9 | p1, p1b, p2, p2b | Gate and baseline checks | GSL competitive; gate removable | GSL becomes the constraint on any claim |
| May 10 | BSPC plan, Framing 3 | Method paper | E1 blocked; p=0.026 unsupported | Anchor put on hold |
| May 13-19 | Recovery; p=0.026 withdrawn | Impossible at n=3 (minimum 0.25) | 0.4826/0.4900 restored from a stash | Valid numbers, but called a "lost" split |
| **May 21** | **BF1 bug fix and re-evaluation** | Disjoint boundaries scored 1.0 | Script mapped ISIC to `val`; **390 replaced by 519** | **Contamination enters the results** |
| May 21-22 | Per-image audit; EGE-UNet | Publishability review | Saw 313/390 overlap, misread | Symptom seen, cause missed |
| Jun 6 | Midterm rescue | No performance win available | Repositioned as a controlled study | "When does..." framing appears |
| Jun 9-10 | Phase 15 OOM, Phase 16 GSL | Matched comparator | 9 runs; noted "obsolete 1,815/389/390 family" | Third missed signal |
| June (from the manuscript ID) | PeerJ submission 142937 | | ISIC on contaminated 519 | Title fixed |
| Oct 17-21 | October follow-up | Attribution before new methods | 39 runs, 38 decisions, all stopped | "No independent contribution established" |
| Undated | Decision letter | | Major revision | Ablation, comparators, statistics, qualitative results |
| **Nov 2** | **Revision audit** | Trace every number | **Step counts expose the mismatch**; official test acquired; p1b replacement proposal withdrawn | 519-image ISIC results withdrawn |
| Nov 3-5 | 96 runs complete | Reviewer requests | No component gain established; GSL reproduced exactly | Rewrites around corrected evidence |
| Nov 5 | Narrative assessment; Stage 1 frozen | Decoder effect unresolved | Narrative 2 chosen | 40 runs registered |
| Nov 6 | Tier addendum | Author asked for a stronger narrative if earned | Criteria fixed before results | |
| Nov 7 | Stage 1 validated | | Tier 1 not met; BDS-Lite and GSL below decoder confirmed | Experiment C frozen and launched; manuscript rewritten without it |

**How the framing moved**

| Stage | Claim |
|---|---|
| April | Method paper: lower HD95 at almost no cost; ISIC and ACDC positive, Synapse a limitation |
| May, BSPC | Method with honest scope; p=0.026 |
| May 21 | Dataset-dependent effect; "defensible but risky" |
| June, PeerJ | Controlled study: helps on single-contour tasks, metric conflict on complex anatomy |
| Oct | Stops short of any method claim |
| Nov 5 | A method paper that tests its own design assumption |
| Nov 7 | Does distillation add to training-only supervision? |

---

## 5. Reconciliation with later audits

### 5.1 The ISIC leak

**Root cause.** Family O trained on 1,815 images, selected on 389, and tested on 390. Those splits are exactly disjoint (verified). `scripts/reeval_main_seeds.py` then set `"isic2018": "val"` with the comment "no test partition; historical 'test' JSON used val predictions". That comment was wrong. The `val` resolved to the *locked* 519 images of a different partition. Evidence on disk: U-Net and BDS-Lite `_test.json` now say `split: val, n=519`, while the other 11 arms still say `split: test, n=390`, and every per-case `_test.csv` still holds the 390 held-out images.

**Signals that were missed**

| Date | Signal | How it was read |
|---|---|---|
| May 14 | 390 prediction IDs match no locked split | "Test split lost" |
| May 21 | 313/390 in locked train | "Cannot be a held-out test". Direction reversed: the 390 were held out from the models' own training |
| June | Split file records 1,815/389/390 | "Obsolete", not recognised as what family O used |
| All along | `config_resolved.yaml` points to that file | Not checked until Nov 2 |

**Where the submitted result came from (class P; reproduces submitted Table 3 exactly).** Seed-averaged paired differences, positive favours BDS-Lite:

| Images | n | HD95 (px) | ASSD (px) | BF1 | DSC |
|---|---:|---|---|---|---|
| All 519 (as submitted) | 519 | +1.04 [+0.24, +1.81] | +0.22 [-0.02, +0.45] | +0.0077 [+0.0008, +0.0147] | -0.0003 |
| In the models' training set | 358 | **+1.76 [+0.74, +2.78]** | +0.43 [+0.16, +0.69] | +0.0110 [+0.0023, +0.0196] | +0.0019 |
| In their selection set | 84 | -1.20 [-3.16, +0.44] | -0.59 [-1.47, +0.06] | -0.0007 | -0.0053 |
| Held out | 77 | +0.09 [-0.99, +1.17] | +0.12 [-0.13, +0.37] | +0.0016 | -0.0046 |
| **Clean 390 test (original)** | 390 | **-0.03 [-0.86, +0.82]** | -0.07 [-0.35, +0.19] | +0.0074 [-0.0007, +0.0159] | -0.0026 |

The HD95 benefit was a training-set effect. A held-out evaluation showing no benefit already existed. The 390-image BF1 used the historical implementation (see 6.3).

**ISIC BF1 across every evaluation**

| Evaluation | Images | Runs per arm | U-Net to BDS-Lite | Status |
|---|---|---:|---|---|
| April exploration | val 519 | 1 | 0.534 to 0.565 | Single seed, selection set |
| Family O, original test | clean 390 | 3 | 0.4826 to 0.4900 (+0.007) | Clean, legacy BF1 |
| Family O, as submitted | 519 | 3 | 0.552 to 0.560 (+0.008) | **Invalid** |
| Family O, official test | 1,000 | 3 | 0.362 to 0.392 (+0.031) | Clean; fixed models, other partition and trainer |
| Retrained, official test | 1,000 | 3 | 0.363 to 0.367 (+0.004) | Clean |
| Retrained, official test | 1,000 | 10 | 0.373 to 0.362 (-0.011) | Clean, frozen |
| Retrained, validation | 519 | 10 | 0.534 to 0.528 (-0.007) | Clean, post hoc |

### 5.2 Historical claims and their status

| Claim | Source | Status now |
|---|---|---|
| ISIC "clean positive" (Month 4) | One seed, validation, tuned on it | Withdrawn |
| p=0.026 | No artifact; impossible at n=3 | Withdrawn |
| ISIC Holm-favourable HD95 and BF1 (Table 3) | 519, contaminated | Invalid (5.1) |
| "U-Net+GSL weaker on ISIC" | Clean GSL vs contaminated arms | Withdrawn |
| Helps on single-contour tasks | Above | Withdrawn; the data do not support it |
| Synapse overlap-distance conflict | Not significant at submission | Not established |
| ACDC BDS-Lite worsens distances | Primary scoring only | Scoring-dependent: -1.45 to +1.99 px |
| Synapse "first significant positive" (May) | One seed, image-level | Not a recipe-level result |
| Gate removable | p1b and family O | **Holds** (under 0.1% of pixels change). Same checkpoints, inference-time |
| Gate "active and modulating" | p1b activations | True but misleading: near-constant channel scale |
| +24% batch-1 latency | Hook-based count | Not reproduced; +8% |
| GSL lead (15/15 cells, 8 Holm) | Three seeds | Vanished on ISIC at ten seeds |
| 1,072 extra parameters | Architecture | **Holds**; not evidence of effectiveness |

### 5.3 Other corrections already made

ACDC M2 arm read from the wrong directory (fixed); ACDC class-weights confusion (retraction); checkpoint hash mismatch was a CRLF error; gradient residual immaterial (6.1e-4 against a 0.139 gap); both-empty convention disclosed; Figure 1 arrow and gate bias corrected.

---

## 6. New findings from this audit

None of these is in the 7 November manuscript. Each carries its evidence class.

### 6.1 The decoder's gain is much smaller on validation (P)

Ten runs per arm on the 519 validation images, with the frozen estimator (`revision/tables/families/audit_validation_contrasts.csv`). Metric parity between sources was verified exactly (seed-1 full recipe matches on DSC, BF1 and HD95).

| Contrast | Validation BF1 | Official test BF1 (frozen) |
|---|---|---|
| Decoder - U-Net | +0.003 [-0.004, +0.010] | +0.021 [+0.001, +0.040] |
| BDS-Lite - decoder | **-0.010 [-0.017, -0.003]** (Holm p=0.067) | -0.032 [-0.052, -0.012] |
| BDS-Lite - U-Net | -0.007 [-0.015, +0.001] | -0.011 [-0.032, +0.010] |
| GSL - decoder | -0.006 [-0.015, +0.003] | -0.031 [-0.052, -0.009] |

- The ordering is the same on both sets: decoder, U-Net, GSL, BDS-Lite.
- **BDS-Lite below decoder has the same direction on separate images, with an interval excluding zero.** It does not pass Holm correction there, so this supports the confirmed claim without replicating it formally.
- Decoder vs U-Net on validation lies just inside SESOI (upper limit 0.00997), which is "no relevant effect" by the frozen labels. The test gain may be specific to the official test distribution, where BF1 is about 0.37 instead of 0.53 and run SD is 2.5-4 times larger. This is a hypothesis, not a finding.
- Stage 2 re-uses the same official test images, so it cannot separate "the decoder helps" from "the decoder helps on this distribution". Report validation alongside it.
- Caveat: validation is the checkpoint-selection set, though selection used DSC and not BF1.

### 6.2 Same recipes, different pipelines: seeds are not the only source of variation (V and P)

- S0 vs M0: ACDC DSC 0.765 vs 0.743, BF1 0.811 vs 0.786. **Synapse DSC 0.537 vs 0.509, BF1 0.560 vs 0.530.** The manuscript flags ACDC only. The Synapse gap (0.028) is within its wide margin but larger than ACDC's.
- M2 is *below* S0 on both datasets (ACDC 0.802 vs 0.811; Synapse 0.545 vs 0.560). So the "decoder gain" on these datasets is measured against a U-Net that an earlier pipeline beat without any decoder.
- Retrained runs peak far earlier: ACDC U-Net selected epochs 16/35/18 vs 105/46/45; Synapse 19/32/49 vs 61/51/89.
- The April trainer ended with higher training loss on all three datasets (for example Synapse final Dice loss 0.059 vs 0.044).
- BDS-Lite S6 (0.392) sits level with the ten-run decoder (0.394) on the official test, while retrained BDS-Lite sits at 0.362. So "BDS-Lite below decoder" is a statement about the retrained pipeline.
- The cause cannot be established because the trainer is lost. No augmentation appears in the configs.

### 6.3 The April grid disagrees with the November ordering (H; raises questions, supports no conclusion)

April arms map onto the current cells and include one cell November lacks: decoder + distillation with no gate. Recomputed under the current both-empty and unit convention (DSC matches S0: 0.7647 vs 0.7641). Legacy BF1 is a sound proxy for current BF1 on the same models (r = 0.997-0.999 over 31,000+ ACDC and Synapse rows; mean absolute difference 0.013-0.017). The cause of that offset is unknown, because the historical implementation is lost. Three seeds per arm; run-level intervals.

| Contrast (BF1) | Realisation | ISIC | ACDC | Synapse | Pooled |
|---|---|---:|---:|---:|---|
| Decoder - U-Net | April (ISIC clean 390) | -0.002 | -0.005 | -0.010 | -0.005 [-0.015, +0.005] |
| | November (seeds 1-3) | +0.016 | +0.016 | +0.015 | +0.016 [+0.005, +0.027] |
| Transfer, no surface | April (distillation, no gate) | +0.008 | -0.002 | +0.023 | **+0.009** [-0.003, +0.020] |
| | November (distillation + gate) | -0.005 | +0.011 | +0.022 | **+0.009** [-0.001, +0.019] |
| Transfer, with surface | April | +0.004 | +0.001 | +0.013 | +0.004 [-0.006, +0.013] |
| | November | -0.013 | +0.003 | +0.006 | +0.001 [-0.010, +0.013] |
| Full - decoder | April | +0.009 | +0.003 | +0.011 | +0.007 [-0.005, +0.018] |
| | November (seeds 1-3; ISIC ten-seed value is -0.032) | -0.012 | +0.000 | -0.008 | not computed |

November cells are the frozen estimates, except Full - decoder, which is the difference of run means in `run_level_noise.csv`. April intervals are run-level only and omit test-unit error, so they are somewhat too narrow.

Three readings, none of them a conclusion:

1. **"Decoder pointed the same way on every dataset" holds only in the retrained set.** In the April grid the sign is negative on all three, and the two pooled intervals narrowly fail to overlap (allowing for the omitted test-unit error, they may touch).
2. **The transfer estimate without the surface loss is nearly identical in two independent pipelines (about +0.009), with ISIC near zero and Synapse near +0.02 in both.** That is more favourable to distillation than the current wording suggests, and it runs *opposite* to the withdrawn single-contour conclusion. It remains undetermined, and on Synapse the aggregation sensitivity already reported applies.
3. **Stage 2 does not include M3.** The contrast that leans positive stays at three seeds after Stage 2.

### 6.4 Distillation may be rescuing the gate, not adding to the decoder (D and P)

October's B-C (+0.019 [-0.001, +0.039]; positive in all three seeds and at all three tolerances) looks like the project's clearest pro-distillation result. Set beside the November decoder arm it reads differently:

| ISIC validation BF1, seeds 1-3 | |
|---|---:|
| M2 decoder, no gate | 0.5429 |
| B: gate + distillation | 0.5422 |
| C: gate, **no** distillation | 0.5230 |
| D: gate + matching, no boundary supervision | 0.5290 |
| G: gate only | 0.5351 |
| U: U-Net | 0.5364 |

C is below M2 by 0.020 [-0.043, +0.003], or 0.015 against all ten decoder runs, and B matches M2 (-0.001 [-0.020, +0.019]). Both intervals include zero, so this is a pattern and not a result. A possible reading: an unconstrained `q_s` driving the gate costs boundary agreement, and distillation removes that cost. On the official test the with-surface cells point the same way (M4 0.361 vs M7 0.380). D below B suggests the target's boundary content matters. This fits the static-gate finding but is three seeds, development images, and post hoc. It is a testable hypothesis that would explain the apparent conflict between B-C and M3-M2, and the re-evaluation on the official test checks it.

### 6.5 Context the paper currently lacks (D)

- **D4 augmentation on ISIC:** U-Net DSC +0.017 [+0.009, +0.025], BF1 +0.032 [+0.008, +0.055], about ten times the decoder's validation effect, with empty predictions rising from 5 to 14 of 1,557. The limitation "no augmentation" can be quantified. Whether any boundary effect survives augmentation is untested, which is the most serious external-validity gap.
- **Three-slice ACDC input:** DSC +0.2 to +0.8 points, ASSD -15% to -42%, HD95 -24% to -61% in all seeds, on 10 validation patients, recovered data. Not a matched result, but "nor 2.5D" (`main.tex:354`) should be reworded, and it bears on R3-M10.
- **Deep supervision (H):** the April grid has a generic auxiliary control. November has none, so whether any decoder effect is boundary-specific is unresolved.

### 6.6 Hyperparameter provenance is misstated (V, documentary)

`docs/devlog.md` shows gate scale 0.5 reduced to 0.25 on 26 April because the full-scale gate "hurt labels 2, 5, and 6" on Synapse validation, with lambda_b and surface weight tuned the same way. `main.tex:84` and `:111` and letter line 171 say these were "fixed before the experiments", and `REVIEWER_COVERAGE.md` says "a-priori bound". Reviewer 2 asked for exactly this justification. Honest answer: chosen by single-seed validation tuning, frozen before the matched evaluation, and the sensitivity analysis shows no endpoint moved beyond threshold. No test data were involved, so no result changes. This also closes open item 6 in `HUMAN_DECISIONS.md`.

---

## 7. Where the evidence stands

Only classes V and D, with P labelled.

| Status | Finding | Estimate | Evidence |
|---|---|---|---|
| **Confirmed** | BDS-Lite below decoder, ISIC BF1 | -0.032 [-0.052, -0.012]; fresh Holm p=0.031 | Claim row BDS-below-decoder; also on validation (P) |
| **Confirmed** | GSL below decoder, ISIC BF1 | -0.031 [-0.052, -0.009]; fresh Holm p=0.014 | Claim row GSL-below-decoder |
| **Measured mechanism** | Distillation raises boundary signal in `q_s` | AP 0.123/0.191/0.104 vs 0.035/0.070/0.012 | Transfer-mechanism (P) |
| **Measured mechanism** | No detectable change in `F_s` | +0.009, +0.001, -0.006; intervals include zero | Transfer-mechanism (P) |
| **Measured mechanism** | Gate is a near-constant channel scale, not band-selective | Under 0.1% of pixels change | Gate-static (P) |
| **Evidence of no relevant effect** | BDS-Lite vs U-Net on ISIC DSC and HD95 | DSC -0.003; HD95 0.07 px higher | BDS-vs-UNet |
| **Evidence of no relevant effect** | Transfer path on pooled DSC, HD95 | All inside SESOI | Transfer-overlap-distance |
| **Evidence of no relevant effect** | Surface loss on HD95 | +0.07 [-0.36, +0.50] px pooled, positive favours the term | Surface-HD95 (additive) |
| **Supported, not confirmed** | Decoder above U-Net, ISIC BF1 | +0.021 [+0.001, +0.040]; fresh Holm p=0.26 | Decoder-vs-UNet; not on validation (P) |
| **Undetermined** | Transfer adds to decoder, no surface | +0.009 [-0.001, +0.019] | Transfer-no-added-value |
| **Undetermined** | Transfer adds, with surface | +0.001 [-0.010, +0.013] | Same row |
| **Undetermined** | BDS-Lite vs U-Net on ISIC BF1 | -0.011 [-0.032, +0.010] | BDS-vs-UNet |
| **Undetermined** | Interaction; task dependence | Pooled -0.008 [-0.025, +0.009] | Three datasets confounded by modality and size |
| **Harm, size unclear** | Surface alone on ISIC BF1 | -0.041 [-0.077, -0.006] | Surface-BF1; not seen in pooled direct cells |
| **Measurement** | Identical recipes reach p<0.05 in about 82% of three-run splits | 79-85% by recipe | Same-recipe (P) |
| **Measurement** | Three-run draws of BDS-Lite vs U-Net reverse sign | 15% significant in the opposite direction | Seed subset (P) |
| **Withdrawn** | All of 5.2 | | |

---

## 8. Reassessing the contributions

| Candidate | Verdict | Reason |
|---|---|---|
| BDS-Lite as a method | **Drop as a contribution** | No gain over U-Net or decoder. STDC and BPKD pre-empt the pattern |
| Parameter efficiency | Drop | 1,072 parameters follows from deleting a decoder. Decoder-only and GSL deploy with zero |
| Task-dependence finding | Drop as a claim | The original version is withdrawn; three datasets cannot establish it |
| **Matched reference for distillation** | **Lead** | The contrast everyone needs and few report. Yields a confirmed result and a usable upper bound |
| **Feature-level transfer check** | **Promote** | Most original result, and it transfers to any feature-distillation method. Currently under-displayed |
| **Recipe-level evaluation** | **Keep, widen** | Add pipeline as a second layer of variation |
| **Corrected ISIC evaluation** | **Keep; sharpen** | Root cause and membership decomposition turn an embarrassment into a checkable case |
| Error geometry | Keep, secondary | Explains why boundary signals move BF1 and not much else |

BDS-Lite becomes **the instrument**: a recipe built so each signal can be removed. That also answers Reviewer 3 on novelty without overclaiming.

---

## 9. Proposed storyline

### 9.1 Thesis

> In a compact U-Net, we could not establish any condition under which boundary distillation adds to simple training-only boundary supervision. Where it appeared to help, the appearance depended on the reference model, the evaluation images, or the number of trainings. Distillation does move boundary signal, but into a projection and not into the features that make the prediction.

### 9.2 Why this answers the title

The 7 November text concedes the title asks its question "loosely" and answers it in one line of Discussion. The proposal makes the title's *when* the paper's organising device, by testing five conditions and reporting each outcome:

| When... | Test | Answer |
|---|---|---|
| **Relative to what?** | Decoder-only reference | Not beyond the decoder. Against U-Net alone, a recipe inherits whatever the decoder gives |
| **In which recipe?** | With and without surface and gate | Leans positive without the surface loss (+0.009, undetermined), about zero with it. Gate alone may cost (development) |
| **On which task?** | Three tasks | No established dependence. The single-contour answer is withdrawn; ISIC is not where any gain appears |
| **On which error?** | BF1, DSC, HD95, error location | Only BF1 moves. No relevant effect on DSC or HD95. About 20% of ISIC error pixels are near the contour against 70% on ACDC |
| **Under what evidence?** | Test partition, seeds, pipelines, test unit | This condition changed the answer most |

The thesis stays true whichever way Stage 2 falls, because it concerns conditions tested, not one contrast (Section 11).

### 9.3 Logic of the paper

1. **The question needs a reference.** "Help" has to mean help beyond the same network with the same boundary decoder and no transfer.
2. **The ruler comes first.** Retraining moves results as much as the effects of interest, so inference is at run level.
3. **The answer, condition by condition.** Confirmed, supported, undetermined and no-relevant-effect results kept distinct.
4. **The mechanism.** The transfer works by its own objective and leaves the prediction unchanged.
5. **What follows.** The earlier answer is withdrawn, there is a practical order of work, and there are checks others can adopt.

### 9.4 Balance

| Positive, reported as such | Negative, reported as such | Inconclusive, reported as such |
|---|---|---|
| Distillation transfers signal (AP up 2.7-8.7 times) | BDS-Lite below decoder (confirmed) | Decoder vs U-Net |
| Target content appears to matter (D control) | Surface loss did not reduce HD95 | Transfer on BF1 |
| Decoder is free at inference, leads ten-run ISIC arms | HD95 benefit was an artifact | Interaction; task dependence |
| Gate removable; graph collapses to U-Net | Training cost +44% time, +46% memory | Whether results survive augmentation |
| No relevant harm to DSC or HD95 vs U-Net | Latency +8% to +15% | |

### 9.5 Alternatives considered

| Storyline | Decision | Why |
|---|---|---|
| Method paper (Tier 1-3) | Rejected | Fails the frozen Tier 1 test |
| "Distillation does not help" | Rejected | Converts undetermined into absence. Pooled upper bound near +0.02; two pipelines lean positive |
| Reproducibility paper | Rejected | Loses the title, and reviewers asked about the method |
| Decoder as the positive finding | Rejected for now | Unconfirmed, absent on validation, opposite in April. Revisit after Stage 2 |
| **Conditions paper** | **Recommended** | Every headline statement matches a result, and it needs no result to go one particular way |

### 9.6 Skeptical objections

| Objection | Response |
|---|---|
| "A negative paper under a question title." | It reports one confirmed contrast, an upper bound, a mechanism and a reversal of the authors' earlier answer. PeerJ judges soundness |
| "The central contrast rests on three seeds." | True. Say so in the abstract, and note Stage 2 does not include M3 |
| "The confirmed deficit does not isolate distillation." | Correct. Attribute it to the recipe, as the manuscript does |
| "One backbone cannot speak for distillation." | Keep every claim to this design. The reference and feature check are the general parts |
| "No augmentation makes this artificial." | Quantify with D4 and state that survival under augmentation is untested |
| "Fresh seeds re-use the same test images." | True. Report validation too |
| "You ran many arms and show few." | Publish the ledger as a supplementary table. This pre-empts the charge |

---

## 10. Section outline with justification

Target: as short as the requested additions allow. By one counter (body text without floats or equations) the submission has 5,529 words and the 7 November draft 8,042; earlier documents gave 7,100 and 8,100 from another count. Reviewer 2 asked for shorter prose.

**Abstract.** Question, why a decoder reference, design, ISIC correction, then results by status, mechanism, implication.
*Why:* R3-A1 and A2 want novelty and numbers. Keep the correction, per the addendum. Do not lead with +0.031.

**1. Introduction** (five paragraphs, the order R3-I4 asks for)
1. Motivation: why contours matter. Keep the one-pixel DSC opening.
2. Background: losses, boundary branches, distillation.
3. Gap: "help relative to what?" and "how would we know?"
4. Approach: BDS-Lite as an instrument; three tasks; five conditions.
5. Contributions, re-ordered: (1) matched-reference test, (2) feature-level check, (3) recipe-level evaluation, (4) corrected ISIC evaluation.

*Change:* replace "clear ordering" with the condition-by-condition answer. *Why:* the current ending claims more than Section 7 allows.

**2. Related work.** Keep, compressed. Add a sentence that the literature rarely reports a decoder-matched reference, once it is verified **[CLAIM-CHECK]**.

**3. Materials and methods**
- 3.1 BDS-Lite: the instrument. Equations, deployed graph, exact provenance of alpha and weights. *R2-4, R2-5, E-2.*
- 3.2 How boundary information can reach the prediction: three routes, written as testable expectations. *R1-2, R3-I2, I6.*
- 3.3 Datasets. *R3-I1, I3, I8.*
- 3.4 Correction of the ISIC evaluation: add root cause, clean-390 existence, membership table (or Supplement). *Why here:* readers must see it before results.
- 3.5 Arms and evidence sets: one factorial diagram mapping each arm to a condition, replacing prose. Add an evidence-class line for retrained, original, and development sets.
- 3.6 Metrics.
- 3.7 Statistics: two questions, status rules, SESOI, and what is post hoc. *R3-S, R3-R14.*

**4. Results** (one claim per subsection, ordered by the logic in 9.3)
- 4.1 How large an effect can be resolved? Keep first. Add pipeline variation (6.2). *Why first:* every later interval depends on it.
- 4.2 Relative to what? Decoder vs U-Net, transfer vs decoder, ISIC ten-run table and figure, with validation beside test (6.1). Central display: Table "the five conditions".
- 4.3 In which recipe? Surface, gate, interaction, additive model labelled exploratory.
- 4.4 On which task? Per-dataset contrasts, and the single-contour reversal stated plainly.
- 4.5 Why the transfer changed little. **New figure.** `q_s` vs `F_s`, gate structure, plus development controls C and D (6.4). *Why:* the most original result has no display.
- 4.6 Which errors change? Error location, spurious components, missed structures, all-case sensitivity, qualitative panel. *R3-R13, R3-M11.*
- 4.7 Other boundary methods and published results. GSL with full weight, Boundary Loss, Attention U-Net, LB-UNet, compact literature table. *R1-3, R3-R15.*
- 4.8 Cost. Last, so it is not read as evidence of effectiveness. *R3-S.*

**5. Discussion**
- 5.1 When does boundary distillation help? The direct answer, using the five conditions.
- 5.2 Why a working transfer left the prediction unchanged.
- 5.3 What changed from the original submission. A short table; state the reversal openly.
- 5.4 Reliability as part of the answer. Partition, seeds, pipelines, scoring of undefined cases.
- 5.5 Practical guidance by evidence strength. A decoder is the cheapest first step, not a proven one, until Stage 2 says otherwise.
- 5.6 Clinical relevance. Keep short. *E-4, R2-2.*
- 5.7 Limitations. Add the augmentation number, pipeline loss, and M3 at three seeds. Reword 2.5D.

**6. Conclusions.** One paragraph: the answer, the mechanism, what to do.

**Supplement (additions)**
- S-A Full experiment ledger, grouped by family.
- S-B April grid as a labelled historical cross-check, only after re-evaluation.
- S-C October controls (G, C, D, UA, three-slice) with development-only labels.
- S-D ISIC membership decomposition and BF1 tolerance sensitivity.

**Main displays (9):** Fig 1 architecture; Fig 2 datasets; Table 1 datasets; Fig 3 ten-run plot; Table 2 frozen contrasts; Table 3 five conditions; Fig 4 central forest plot; Fig 5 mechanism (new); Fig 6 qualitative. Deployment table moves to a compact row or the Supplement.

### Reviewer coverage under this structure

| Cluster | Answered in | Remaining gap |
|---|---|---|
| Ablation (R1-4, E-1, R3-M9) | 4.2-4.3 | Interaction unresolved at three seeds |
| Theory (R1-2, E-2) | 3.2, 4.5 | Post hoc, stated |
| Comparators (R1-3, R3-R15) | 4.7 | nnU-Net, Swin UNETR not run |
| Statistics (R3-S, R14) | 3.7, 4.1 | None |
| Qualitative (R3-R13) | 4.6 | ACDC and Synapse label maps only |
| Compact focus (R2-1, E-3) | 1, 3.1 | Argued, not tested |
| Clinical relevance (E-4, R2-2) | 5.6 | EF, mass and mm not computed |
| 2.5D, small targets (R3-M10, M11) | 4.6, 5.7 | Three-slice context only |
| Novelty (R3-A1, R3-O) | 1, 8 | Most exposed point; answered by reframing |

---

## 11. Stage 2 branches

No prediction is implied. Rows follow `CONFIRMATION_PROTOCOL.md` Section 6.

| Outcome | Sentences that change | What stays |
|---|---|---|
| Primary significant, positive | Decoder becomes the positive finding; Abstract, 4.2, 5.1, 5.5 | Five conditions; thesis about distillation |
| ...with uneven datasets | Same, stated as uneven | Same |
| Primary not significant | Drop any recommendation relying on the decoder; 5.5 becomes "check on your task" | All else |
| Primary significant, negative | Earlier estimates reported as not replicated | All else |
| BDS-Lite vs decoder confirmed on ACDC or Synapse | Task condition gains a confirmed entry. In BDS-Lite's favour, this is the first positive condition for the title, on that dataset only | Tiers 1-3 stay unmet |
| Not established there | ISIC deficit not extended | All else |

---

## 12. Corrections needed in the current draft

| Location | Problem | Fix |
|---|---|---|
| `main.tex:84`, `:111`; letter `:171`; `REVIEWER_COVERAGE.md:23` | "Fixed before the experiments", "a-priori" contradicted by the devlog | State validation tuning, frozen before the matched evaluation |
| `main.tex:55`, `:216`, `:326` | "Clear ordering", "pointed the same way on every dataset" | Restrict to retrained set; add validation result |
| `claim_evidence_table.csv` row Decoder-direction | Same | Scope it |
| `main.tex:269` | Low U-Net stated for ACDC only | Add Synapse |
| `main.tex:354` | "nor 2.5D" | "No matched 2.5D"; cite the development control |
| ISIC correction section | No root cause; no mention that a clean evaluation existed | Add both |
| Confirmed contrasts | Scope not stated | "In the retrained pipeline" |
| `main.tex:49`, `:330` | "Reports usually compare..." uncited | **[CLAIM-CHECK]** or cite |
| Response letter | 27 of 86 locations stale | Rewrite after the manuscript; list ACDC M2 and these corrections |
| Stage 2 omission | Conflicts with addendum rule 5 | Author decision (below) |

---

## 13. Decisions and next steps

No item below has been started.

| # | Decision | Recommendation | Cost |
|---|---|---|---|
| 1 | Rewrite now or after Stage 2? | **Wait.** About 39 hours or more; it tests the claim most exposed in 6.1-6.3, and rule 5 otherwise needs a disclosure | Time only |
| 2 | Correct the provenance text (12) | Do regardless | None |
| 3 | Disclose all families in a supplementary ledger? | **Yes.** Protects against a selective-reporting charge | Text only |
| 4 | Re-evaluate April decoder, distillation and gate arms, and October G, C, D, on the official test with the current evaluator? | **Yes, if a short protocol is frozen first.** Inference only, on CPU, so it does not touch the GPU queue. New Holm family, labelled post hoc. Turns 6.3 and 6.4 from H/D into validated evidence | 36 model-dataset evaluations (27 April, 9 October) |
| 5 | BF1 tolerance sensitivity on frozen predictions | Yes, same protocol. BF1 is the primary endpoint | CPU |
| 6 | Add M3 at fresh seeds after Stage 2? | The highest-value single experiment for the title: about 21 runs, roughly 13 GPU hours. Would need registration under the addendum | Needs your authorisation |
| 7 | Add an augmented arm? | Only if a reviewer presses. It changes the locked protocol | 12+ runs |

**Suggested order:** items 2 and 3 now; freeze a protocol for 4 and 5; let Stage 2 finish and validate; unblind; rewrite to this outline; rewrite the letter.

---

## Appendix: files in this folder

| File | Content |
|---|---|
| `experiment_ledger.csv` | 536 run records |
| `april_grid_runs_current_convention.csv` | April arms, run level |
| `isic_val_test_runs.csv` | ISIC validation and test per run, Stage 2 excluded |
| `scripts/build_ledger.py` | Ledger builder (Stage 2 by existence only) |
| `scripts/rederive_stage1.py` | Re-derivation of frozen contrasts |
| `scripts/isic_leak_decomposition.py` | Section 5.1 |
| `scripts/isic_val_test_table.py`, `isic_val_contrasts.py` | Sections 6.1, 6.4 |
| `scripts/april_grid.py`, `legacy_bf1_parity.py` | Section 6.3 |
| `scripts/selection_epochs.py`, `training_trajectories.py` | Section 6.2 |

Re-derivation result: all BF1 and DSC contrasts match the frozen file exactly, in all-seed and fresh-seed sets. HD95 differs by at most 0.07 px through handling of undefined pairs, against a 2 px threshold.
