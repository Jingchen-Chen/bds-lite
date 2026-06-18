# BDS-Lite Public Release — Stage 2 Proposal (tree + dispositions)

**Target paper:** `results/submission_candidate_manuscript_v1/` (the candidate).
**Release root:** `<release-root>/` (sibling; confirmed).
**Disposition keys:** **COPIED** (verbatim from a source path) · **GENERATED**
(produced at build time from source data) · **AUTHORED** (newly written by me) ·
**+EDIT** (copied then minimally modified — scope/scrub only, diff-reported).

Scope rules applied: GSL **IN**; EGE-UNet / MedSAM / Boundary DoU / MobileViT /
Attention-UNet **OUT**; `manuscript_v2/` **excluded**; internal phase-named dirs mapped
to clean public paths; `cluster_level_statistics.csv` and `failure_case_manifest.csv`
are **artifacts of record — copied verbatim, never regenerated**.

---

## 1. Proposed tree

### Top level
| Path | Disposition | Rationale |
|---|---|---|
| `README.md` | AUTHORED | Honest conditional summary; install; "reproduce" map for candidate Tables 1–6 / Figures 1–6; data/code availability; scope/limits; BibTeX. |
| `LICENSE` | AUTHORED | MIT for original code (dataset licenses separate). |
| `THIRD_PARTY_NOTICES.md` | AUTHORED | GSL port attribution + upstream license; U-Net architecture citation (see §2a). |
| `CITATION.cff` | AUTHORED | Paper + repo metadata; Zenodo DOI placeholder. |
| `requirements.txt` | GENERATED | Exact pins from `.venv`: `numpy==2.4.4 scipy==1.17.1 torch==2.11.0 torchvision==0.26.0 scikit-image==0.26.0 pandas==3.0.2 Pillow==12.2.0 PyYAML==6.0.3 einops==0.8.2 tqdm==4.67.3 matplotlib==3.10.9 h5py==3.16.0`; `nibabel` = TODO (not in current venv; needed for ACDC raw conversion). |
| `environment.yml` | GENERATED | Conda env: Python 3.14.4 + same pins. |
| `.gitignore` | AUTHORED | Exclude `data/`, `*.pt *.pth *.npy *.npz`, `*.log`, caches, `.venv`. |
| `reproduce.sh` | AUTHORED | End-to-end: convert → train → evaluate → **single full-script** `analysis/generate_rescue_analysis.py` (RNG note) → tables/figures; flags steps needing Zenodo/raw data. |
| `ZENODO_MANIFEST.csv` | GENERATED | Bundle/file/bytes/sha256 (§2b), built in Stage 3. |
| `audit/AUDIT.md` | COPIED | The re-anchored Stage-1 audit (moved here). |
| `audit/STAGE2_PROPOSAL.md` | COPIED | This document. |

### `manuscript/`
| Path | Disposition | From / rationale |
|---|---|---|
| `BDS_Lite_submission_candidate_v1.md` | COPIED | `results/submission_candidate_manuscript_v1/manuscript/` — the paper. |
| `BDS_Lite_submission_candidate_v1.docx` | COPIED (optional) | same dir — convenience conversion (label as non-authoritative). |
| `tables/table{1..6}_*.{csv,md}` | COPIED | `…/tables/` — paper tables, each row carries `source_artifact`. |
| `integration_audit.md` | COPIED | `…/00_integration_audit.md` — Phase16→17 integration record. |
| `human_tasks_before_submission.md` | COPIED | `…/` — outstanding human/editorial tasks. |

### `src/bds_lite/` (in-scope modules only)
| Path | Disposition | Rationale |
|---|---|---|
| `__init__.py`, `utils/{__init__,config,seed}.py` | COPIED | Package + config/seed utils. |
| `models/unet.py` | COPIED (+EDIT?) | U-Net backbone (1,927,042 params). Contains `AttentionUNet` — strip class or leave dormant (surface). |
| `models/bds_lite.py`, `models/blocks.py` | COPIED | h_φ decoder, bounded gate α=0.25, shared blocks. |
| `models/__init__.py` | COPIED **+EDIT** | Remove `EGEUNet`/`MobileViTUNet`/`AttentionUNet` from imports + `__all__`. |
| `losses/gsl.py` | COPIED | GSL comparator (third-party port — §2a). |
| `losses/__init__.py` | COPIED **+EDIT** | Remove `boundary_dou` import. |
| `losses/__init__`…`training/losses.py` | COPIED | Composite BDS-Lite loss (BCE+Dice boundary, SmoothL1 stop-grad distill, signed-distance surface, GSL hook). |
| `training/{runner,__init__}.py` | COPIED | Training loop; max-val-DSC checkpoint. |
| `training/builders.py` | COPIED **+EDIT** | Drop `ege_unet`/`mobilevit_unet`/`attention` build branches. |
| `evaluation/{metrics,profiling,__init__}.py` | COPIED | Corrected Boundary F1, DSC/IoU/HD95/ASSD, resource profiler. |
| `data/{datasets,boundary,converters,schema,__init__}.py` | COPIED | Pipeline + dataset converters + boundary targets. |
| ~~`models/ege_unet.py`, `models/mobilevit_unet.py`, `losses/boundary_dou.py`, `*.egg-info`~~ | **DROP** | Out of scope / build cruft. |

### `configs/`
| Path | Disposition | Rationale |
|---|---|---|
| `base.yaml`, `datasets/{isic2018,acdc,synapse}.yaml` | COPIED | Locked common + dataset configs. |
| `gsl/phase16_matched_gsl_{ds}_seed{1,2,3}.yaml` (9) | COPIED **+SCRUB** | From `…/gsl_memory_bounded/configs/`; remap `output_dir` + `split_file` to public paths (§2c). |
| `run_resolved/{ds}_{unet,bds_lite_full}_seed{1,2,3}.yaml` (18) | COPIED **+SCRUB** | **Config of record** — `outputs/runs/*/config_resolved.yaml` (fully expanded; α=0.25, λ's). |
| `reconstructed/{ds}_{unet,bds_lite_full}.yaml` | AUTHORED (optional) | Clean top-level YAML, **labeled reconstruction, diff-verified** to the resolved snapshot (surface — decision #2). |
| ~~`configs/experiments/*` (50)~~ | **DROP** | EGE/MobileViT/Attention/ablation/p2/e1 archaeology; superseded by run_resolved + recipe. |

### `splits/`
| Path | Disposition | Rationale |
|---|---|---|
| `{isic2018,acdc,synapse}/{train,val,test}.json` | COPIED | From `data/splits/` — seed 2026, `sha256`, subject-disjoint. |
| ~~top-level `splits/*_split.json`~~ | **DROP** | Stale seed-1 set (ISIC 1815/389/390). |

### `scripts/`
| Path | Disposition | Rationale |
|---|---|---|
| `train.py`, `evaluate.py`, `reeval_main_seeds.py` | COPIED | Train / evaluate / regenerate main seed evals. |
| `convert_{isic2018,acdc,synapse}.py`, `prepare_boundary_targets.py` | COPIED | Rebuild `data/processed` + boundary targets from raw. |
| `profile_model.py` | COPIED | Resource/inference profile. |
| `gsl/{aggregate_phase16_gsl,gsl_memory_bounded,train_memory_bounded_gsl,profile_memory_bounded_gsl,prepare_phase16_gsl}.py`, `gsl/run_phase16_gsl.sh` | COPIED **+SCRUB** | From `…/submission_blocker_resolution_v1/scripts/` — GSL matched rerun + profiling. |
| ~~`adapt_legacy_*`, `train_step3_10ep`, `generate_*` (manuscript_v2 tables/figs), `prepare_*` reports, `per_image_*`, `eval_gsl_baselines`, `visualize_*`, `profile_boundary_branch_ablation`~~ | **DROP** | manuscript_v2-specific or project archaeology (surface any you want kept). |

### `analysis/` (cluster-aware backbone, mapped from `midterm_q2_rescue/`)
| Path | Disposition | Rationale |
|---|---|---|
| `generate_rescue_analysis.py` | COPIED **+EDIT(paths)** | Producer of cluster stats + 24-panel selection; remap input constants (§2c). |
| `outputs/cluster_level_statistics.csv` | COPIED | **Artifact of record (Table 3) — do NOT regenerate.** |
| `outputs/failure_case_manifest.csv` | COPIED | 24-panel deterministic selection record. |
| `outputs/{per_case_distribution_summary,outlier_sensitivity,tradeoff_pattern_counts,subgroup_summary,per_class_summary,per_case_metrics,per_class_case_metrics,top_case_changes,boundary_distance_correlations,property_delta_correlations,gate_removal_summary,deployment_resource_comparison}.csv` | COPIED | Supporting per-case/subgroup/outlier outputs (all verbatim-verified). |
| `outputs/analysis_run_summary.json` | COPIED | Run provenance (seed, n_clusters). |
| `inputs/` | COPIED | Upstream summaries the script consumes that survive (gate-removal `p1b_gate_comparison/summary.csv`, `resource_profile_isic2018_comparison.csv`). Superseded-GSL inputs → surface (§2c). |
| `README.md` | AUTHORED | **Bootstrap-RNG caveat**, artifacts-of-record policy, single-full-script re-run guidance, pinned numpy/scipy. |
| narrative `*.md` (`per_case_analysis.md`, `subgroup_analysis.md`, `failure_case_analysis.md`, `statistical_robustness_audit.md`, …) | COPIED → `analysis/notes/` (optional) | Human-readable commentary; surface keep/drop. |

### `results/` (candidate evidence, clean names)
| Path | Disposition | Rationale |
|---|---|---|
| `main_test_means.csv` | GENERATED | U-Net + BDS-Lite rows of `recomputed_main_test_means.csv` (drop EGE rows) — filter only, no recompute (surface vs copy+note). |
| `matched_gsl_summary.csv`, `matched_gsl_seed_results.csv` | COPIED | `…/gsl_memory_bounded/` — GSL Table 2 rows. |
| `seed_level_wilcoxon_summary.csv` | COPIED | `…/final_manuscript_audit_v2/` — all-1.0 seed-level Holm. |
| `training_cost_combined_summary.csv`, `gsl_training_cost_summary.csv`, `profiling_environment.json` | COPIED | `…/submission_blocker_resolution_v1/profiling/` — Table 4 source + env. |
| `resource_profile_isic2018_comparison.csv` | COPIED | `results/tables/` — deploy params/+1,072. |
| `gsl/evaluations/phase16_matched_gsl_*_{val,test}.csv` | COPIED | GSL per-seed eval. |
| `gsl/runs/phase16_matched_gsl_*/{config.json,metrics.jsonl}` | COPIED | GSL run provenance (drop `train.log`). |

### `figures/`
| Path | Disposition | Rationale |
|---|---|---|
| `figure1_architecture_training_inference_decoupling.{pdf,png}` | COPIED | `…/submission_critical_evidence_v1/figures/`. |
| `figure2_matched_main_results.{pdf,png}` | COPIED | `…/submission_candidate_manuscript_v1/figures/` (journal version). |
| `figure3_per_case_delta_distributions.{pdf,png}` … `figure6_failure_cases.{pdf,png}` | COPIED | `…/submission_critical_evidence_v1/figures/`. |
| `failure_panels/*.png` (24) | COPIED | `…/midterm_q2_rescue/figures/failure_cases/` — transparency for the 24-panel claim. |
| `captions.md` | COPIED | `…/submission_candidate_manuscript_v1/figures/final_caption_set.md`. |
| ~~`figure2_main_results_separated_families.*`~~ | **DROP** | Superseded Phase-15 figure. |

### `docs/`
| Path | Disposition | Rationale |
|---|---|---|
| `data_access.md` | AUTHORED | ISIC2018/ACDC/Synapse download + DOIs/citations + license/ethics; rebuild `data/processed` + splits via converters. |
| `mechanism.md` | AUTHORED | Task-dependence: why ISIC helps, ACDC null, Synapse distance-negative. |
| `naming.md` | AUTHORED | The `_test`→validation misnomer **and the inversion**: aggregates `isic2018_*_test.json` = 519 real val; prediction dirs `isic2018_*_val/` = 519 real val; stale 389/390 dropped. |
| `compute.md` | AUTHORED | RTX 5060 Laptop GPU, Linux 7.0.0, Python 3.14.4, torch 2.11.0+cu130, CUDA 13.0, pinned libs. |
| `statistics.md` | AUTHORED | Cluster methods (bootstrap CI, Holm signed-rank, sign effect, LOCO) + **bootstrap-RNG reproducibility caveat** + exact numpy/scipy pins. |
| `config_of_record.md` | AUTHORED | run_resolved snapshots + GSL configs; the missing top-level YAMLs. |

### `tests/`
| Path | Disposition | Rationale |
|---|---|---|
| `unit/test_split_integrity.py` | COPIED **+EDIT** | Disjointness guard; point at `splits/`. |
| `unit/{test_metrics,test_losses,test_boundary,test_profiling}.py` | COPIED | In-scope unit tests. |
| ~~`test_converters,test_cross_dataset_report,test_final_ablation_report,test_seed_averaged_results,test_recent_boundary_losses`~~ | DROP (surface) | Reference dropped code/reports. |

---

## 2. Items to resolve before Stage 3

### (a) Third-party code + license inventory
| Component | File (in release) | Origin | Upstream license | Action needed |
|---|---|---|---|---|
| **GSL** (Generalized Surface Loss) | `src/bds_lite/losses/gsl.py` | Celaya, Riviere, Fuentes 2024; ref impl `github.com/aecelaya/gen-surf-loss` @`2115cca` (arXiv:2302.03868) | **NOT stated in our file** | **CONFIRM upstream license** (do not assume); carry its LICENSE text + citation/persistent ID in `THIRD_PARTY_NOTICES.md` per PeerJ. **This is the one true blocker.** |
| U-Net backbone | `src/bds_lite/models/unet.py` | Ronneberger et al. 2015 (architecture); our reimplementation | original code | MIT (ours) + cite paper. |
| Metrics / losses / data / profiling | `src/bds_lite/{evaluation,training,data}/…` | our implementation (standard HD95/ASSD/Dice defs) | original code | MIT (ours). |
| EGE-UNet (DROPPED) | — | `JCruan519/EGE-UNet` @`f87e3df`, MIT | MIT | excluded; no obligation. |
| Boundary DoU (DROPPED) | — | `sunfan-bvb/BoundaryDoULoss` | unconfirmed | excluded. |
| Datasets | not shipped | ISIC2018 / ACDC / Synapse | dataset-specific | download instructions + DOIs only; no redistribution/relicense. |

### (b) Zenodo manifest plan (built in Stage 3)
| Bundle | Contents | Approx size / count | sha256 |
|---|---|---|---|
| `checkpoints_unet_bdslite.tar` | `outputs/runs/{ds}_{unet,bds_lite_full}_seed{1,2,3}/best.pt` | ~0.6 GB / 18 | per-file + tarball |
| `checkpoints_gsl.tar` | `…/gsl_memory_bounded/checkpoints/phase16_matched_gsl_*/best.pt` | ~0.2 GB / 9 | per-file + tarball |
| `predictions.tar` | `outputs/evaluations/predictions/{ds}_{unet,bds_lite_full}_seed{1,2,3}_{split}` (ISIC **val**=519, ACDC test=366, Synapse test=1568) | ~1.2 GB / ~13.5k npy | per-file + tarball |
- `ZENODO_MANIFEST.csv`: `bundle,relative_path,bytes,sha256`; tarball-level sha256 alongside.
- Each `results/*.csv` already names its on-disk `source_artifact`; the manifest maps those to archived bundles, so a reviewer can trace table → CSV → (Zenodo) array/checkpoint.
- **Decisions:** best-only vs include `last.pt`/epoch checkpoints? one tarball vs per-dataset? Hashing ~13.5k files is a Stage-3 build step (not run now).

### (c) Residual path / secret scrubs + edits
1. **Exclude all `*.log`** (the only absolute-path leaks in in-scope dirs: `midterm_q2_rescue/analysis/generate_rescue_analysis.log`, GSL `logs/*.log`, `runs/*/train.log`). Plus exclude `manuscript_v2/build/`, `docs/devlog.md` (carried from Stage 1).
2. **GSL configs** (`configs/gsl/*`): remap `output_dir` and `split_file` (currently `results/submission_blocker_resolution_v1/gsl_memory_bounded/…`) to public `splits/` + a public results dir; verify the GSL locked split == `data/splits/` (both seed 2026).
3. **`analysis/generate_rescue_analysis.py` input constants** reference the original layout incl. `manuscript_v2/supplementary/holm_family.csv` (dropped) and superseded GSL dirs. Plan: vendor the **used** inputs (gate-removal summary, resource profile) into `analysis/inputs/` and patch those path constants; mark the superseded-GSL branch (its `gsl_*_summary.csv` outputs are not used by the candidate — Table 2 GSL comes from the phase-16 matched rerun). Re-running still yields RNG-similar (not bit-identical) CIs; the committed CSV stays the record.
4. **Ship `config_resolved.yaml` (expanded), not `config_source.yaml`** for run-of-record (the latter `extends:` the missing `configs/models/bds_lite.yaml`).
5. **Final full-tree secret/path scan** before the Stage-3 commit.

### Other decisions to confirm
- `main_test_means.csv`: ship **filtered** (U-Net/BDS only, GENERATED) vs copy full + note EGE unused?
- Produce the optional `configs/reconstructed/*` clean YAMLs (labeled, diff-verified), or rely on `run_resolved/` only?
- Keep the `.docx` convenience copy and the `analysis/notes/*.md` narratives, or md/CSV only?
- `nibabel` pin for `requirements.txt` (absent from current `.venv`; needed only for ACDC raw conversion) — confirm a version or mark optional-extra TODO.

---

*Stage 2 proposal. No release files beyond `AUDIT.md` + this proposal have been created.
Awaiting your resolution of §2(a)/(b)/(c) and the "other decisions" before Stage 3 build.*
