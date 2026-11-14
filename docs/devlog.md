# Development Log

Use this file as the compact research log for implementation and experiment work. Keep entries factual and reproducible.

## Template

```text
## YYYY-MM-DD - branch-name

Goal:
- 

Changed:
- 

Commands:
- 

Result:
- 

Problems:
- 

Next:
- 
```

## 2026-04-26 - initial-scaffold

Goal:
- Establish the initial BDS-Lite research codebase from `plan.md`.

Changed:
- Added source package layout, configs, docs, scripts, paper folders, and tests.
- Added boundary/SDF generation, baseline U-Net, initial BDS-Lite U-Net, losses, and metrics.

Commands:
- `python3 -m compileall src scripts tests`
- `PYTHONPATH=src python3 -m pytest tests/unit`

Result:
- Unit tests passed: `6 passed`.

Problems:
- No processed datasets are present yet, so training was not run.
- This workspace did not contain `.git` at the time of scaffold validation.

Next:
- Convert Synapse, ACDC, and ISIC2018 into the documented `.npz` format.
- Generate boundary targets.
- Run the first U-Net baseline.

## 2026-04-26 - cleanup-review

Goal:
- Check scaffold quality against `plan.md` and remove obvious wiring drift.

Changed:
- Made model construction fall back to dataset `in_channels` and `num_classes`.
- Added unit tests for model-building defaults.
- Corrected README status for the existing ISIC2018 converter.
- Removed empty/local-only repo placeholders: `.codex`, notebooks, tools, experiments, and custom `.githooks`.

Commands:
- `python3 -m pytest tests/unit`

Result:
- Unit tests passed: `8 passed`.

Problems:
- No raw or processed datasets are present, so conversion and training cannot run yet.

Next:
- Add or point to raw dataset files.
- Convert at least one dataset to `.npz`.
- Generate boundary targets and run the first U-Net baseline.

## 2026-04-26 - audit-cleanup

Goal:
- Audit the scaffold for academic rigor before Phase 2; close gaps in the training
  loop, evaluation, and reproducibility plumbing without expanding scope.

Changed:
- `src/bds_lite/training/runner.py`: migrated to non-deprecated `torch.amp.GradScaler`
  / `torch.amp.autocast`, added optional gradient clipping and LR scheduler step,
  and accumulated mean foreground Dice/IoU during validation.
- `scripts/train.py`: added cosine LR scheduler, deterministic DataLoader workers
  via `worker_init_fn` and a seeded `Generator`, structured logging to both stdout
  and `results/runs/<experiment>/train.log`, JSONL metrics dump, and `best.pt` /
  `last.pt` checkpoint tracking based on validation DSC.
- `scripts/evaluate.py`: new script that loads a checkpoint and reports per-class
  Dice, IoU, HD95, ASSD, and Boundary F1 on a chosen split, writing a CSV next to
  the checkpoint.
- `configs/base.yaml`: surfaced `trainer.scheduler` and `trainer.grad_clip`.
- `pyproject.toml`: bumped `torch>=2.4` / `torchvision>=0.19` so the new AMP API
  is guaranteed to be present.
- `.gitignore`: added `wandb/`, `mlruns/`, `runs/`, `lightning_logs/`, `*.log`,
  `.ipynb_checkpoints/`, and `.env*` patterns.
- `src/bds_lite/training/losses.py`: documented that `LossWeights.surface` is
  reserved and not yet wired into `BDSLiteLoss`.
- `README.md`: refreshed the status section and documented the new evaluation
  workflow and per-run output layout.

Commands:
- `ruff format .`
- `ruff check .`
- `python3 -m pytest tests/unit`

Result:
- Unit tests passed: `8 passed`. Ruff format and lint both clean.
- Both ISIC experiment configs build their models and pick up the new
  scheduler/grad_clip defaults; new `torch.amp` API verified against the
  installed PyTorch (2.11).

Problems:
- A full smoke-test epoch was not run because no `val/` split exists under
  `data/processed/isic2018/` yet (only `train/`).

Next:
- Generate ISIC2018 `val/` split (`--val-fraction 0.2 --seed 2026`) and run a
  short baseline to confirm the new training loop end-to-end.
- Wire surface-loss term into `BDSLiteLoss` once D2/D5 decisions are taken.
- Add Synapse and ACDC dataset converters.

## 2026-04-26 - isic-configs

Goal:
- Add first ISIC2018 experiment configs for Month 1 baseline work.

Changed:
- Added U-Net and BDS-Lite ISIC2018 experiment configs.

Commands:
- `PYTHONPATH=src python3 -c "from bds_lite.utils import load_config; from bds_lite.training import build_model; paths=['configs/experiments/isic_unet_baseline.yaml','configs/experiments/isic_bdslite_unet.yaml']; for p in paths: c=load_config(p); m=build_model(c); print(p, c['dataset']['in_channels'], c['dataset']['num_classes'], type(m).__name__)"`
- `python3 -m pytest tests/unit`

Result:
- ISIC configs build expected models: `UNet` and `BDSLiteUNet`, both using 3 input channels and 2 classes.
- Unit tests passed: `8 passed`.
- Raw ISIC2018 files are not present under `data/raw`, so conversion could not be run.

Problems:
- Need ISIC2018 image and mask folders before data conversion.

Next:
- Place ISIC2018 Task 1 images and masks under `data/raw/isic2018`.
- Convert ISIC2018 to `.npz`, generate boundary targets, and run the U-Net baseline.

## 2026-04-26 - isic-data-and-smoke-test

Goal:
- Convert all ISIC2018 training images with a proper train/val split, generate
  boundary targets, and verify the training pipeline runs end-to-end.

Changed:
- `data/processed/isic2018/`: regenerated from all 2594 ISIC2018 Task 1 pairs
  (224×224, float32) with 80/20 train/val split (seed 2026); train=2075, val=519.
- `data/processed/isic2018/train/`: boundary (`radius=1`) and SDF targets added
  in-place with `prepare_boundary_targets.py`.
- `configs/base.yaml`: removed `model.in_channels: 1` and `model.num_classes: 2`
  from the `model:` block — those are dataset-specific; the builder already falls
  back to `dataset.in_channels` / `dataset.num_classes`.

Commands:
- `PYTHONPATH=src python3 scripts/convert_isic2018.py --image-dir data/raw/isic2018/ISIC2018_Task1-2_Training_Input --mask-dir data/raw/isic2018/ISIC2018_Task1_Training_GroundTruth --output-root data/processed/isic2018 --image-size 224x224 --val-fraction 0.2 --seed 2026`
- `PYTHONPATH=src python3 scripts/prepare_boundary_targets.py --dataset isic2018 --split train --radius 1 --num-classes 2`
- `PYTHONPATH=src python3 -m pytest tests/unit -q`

Result:
- Unit tests: `8 passed`.
- 3-epoch smoke test on ISIC2018 U-Net baseline (GPU, AMP, batch=8, num_workers=0):
  - epoch 1: train=0.8035, val_loss=0.6508, val_dsc=0.7426, val_iou=0.6468
  - epoch 2: train=0.5511, val_loss=0.5451, val_dsc=0.7761, val_iou=0.6884
  - epoch 3: train=0.4578, val_loss=0.4687, val_dsc=0.8088, val_iou=0.7247
- Pipeline is end-to-end verified: data → train → val metrics.

Problems:
- `num_workers > 0` does not work when training is invoked from a Python heredoc
  (no importable `__main__`); use `scripts/train.py` for actual runs.
- No `test/` split exists yet; evaluate.py defaults to `--split test`. Use
  `--split val` until a held-out test split is created.

Next:
- Run full 150-epoch U-Net baseline: `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/isic_unet_baseline.yaml`
- Add Synapse and ACDC dataset converters (Month 1 deliverable).
- Create a held-out test split for ISIC2018 (extend converter or copy val).

## 2026-04-26 - synapse-converter

Goal:
- Write Synapse and ACDC dataset converters; convert all available Synapse data.

Changed:
- `src/bds_lite/data/converters.py`: added `convert_synapse` (reads HiFormer-layout
  `train_npz/*.npz` + `test_vol_h5/*.h5`, case-level train/val split, resizes,
  saves as BDS-Lite NPZ with `image [1,H,W]` and `mask [H,W]`) and `convert_acdc`
  (reads standard ACDC NIfTI layout with patient-level train/val/test split,
  per-volume min-max normalisation).
- `scripts/convert_synapse.py`: CLI wrapper for `convert_synapse`.
- `scripts/convert_acdc.py`: CLI wrapper for `convert_acdc`.
- `configs/experiments/unet_baseline.yaml` / `bdslite_unet.yaml`: removed redundant
  `model.in_channels` and `model.num_classes`; added explicit `trainer:` block
  (batch=16, lr=1e-4 for multi-organ task).
- `pyproject.toml`: added `[project.optional-dependencies] data` group with `h5py`
  and `nibabel`.

Commands:
- `PYTHONPATH=src python3 scripts/convert_synapse.py --src /home/jingchen/vit-refined/HiFormer/data/Synapse --output-root data/processed/synapse --image-size 224x224 --val-fraction 0.2 --seed 2026`
- `PYTHONPATH=src python3 scripts/prepare_boundary_targets.py --dataset synapse --split train --radius 1 --num-classes 9`
- `PYTHONPATH=src python3 -m pytest tests/unit -q`

Result:
- Synapse: train=1738, val=473, test=1568 slices; all 224×224, `image [1,H,W]`.
- Boundary + SDF targets generated for synapse/train.
- Unit tests: `8 passed`.
- ISIC2018 150-epoch U-Net baseline running (epoch 30: val_dsc=0.849).

Problems:
- ACDC raw data is not present; `convert_acdc` is implemented but untested.
  Need to download ACDC challenge data from creatis.insa-lyon.fr.

Next:
- Download ACDC data and run `convert_acdc.py`.
- After ISIC training finishes, run `evaluate.py --checkpoint results/checkpoints/isic2018_unet_baseline/best.pt --split val`.
- Start Synapse U-Net baseline training.
- Consider running `isic_bdslite_unet` experiment in parallel.

## 2026-04-26 - acdc-converter

Goal:
- Add TransUNet-format ACDC converter and complete all three dataset conversions.

Changed:
- `src/bds_lite/data/converters.py`: added `convert_acdc_transunet()` reading flat
  directory of `patient{NNN}_frame{NN}_slice_{N}.h5` files (TransUNet pre-sliced
  format); patient-level 70/10/20 train/val/test split; resizes to target size.
- `scripts/convert_acdc.py`: added `--format` flag (`transunet` default, `nifti`
  for official NIfTI layout); routes to correct converter.

Commands:
- `PYTHONPATH=src python3 scripts/convert_acdc.py --src data/raw/acdc/ACDC/ACDC_training_slices --output-root data/processed/acdc --image-size 224x224 --val-fraction 0.1 --test-fraction 0.2 --seed 2026`
- `PYTHONPATH=src python3 scripts/prepare_boundary_targets.py --dataset acdc --split train --radius 1 --num-classes 4`

Result:
- ACDC: train=1350, val=186, test=366 slices; image [1,224,224] z-score normalised float32, labels 0-3.
- Boundary + SDF targets being generated for acdc/train.
- All three Month 1 datasets now converted: ISIC2018 ✓, Synapse ✓, ACDC ✓.

Problems:
- None.

Next:
- After ISIC training finishes (~epoch 150): evaluate best checkpoint on val split.
- Start Synapse U-Net baseline training.
- Add ACDC experiment configs (unet + bdslite_unet).

## 2026-04-26 - isic-baseline-eval

Goal:
- Finish the ISIC2018 U-Net baseline evaluation and make Month 1 baseline
  tracking explicit.

Changed:
- `src/bds_lite/evaluation/metrics.py`: undefined HD95/ASSD now return `nan`
  instead of `inf` when either predicted or target surface is empty.
- `scripts/evaluate.py`: aggregates per-sample and per-label metrics with
  `nanmean`, so undefined surface-distance samples do not poison the reported
  mean.
- `tests/unit/test_metrics.py`: added a regression test for empty-prediction
  surface-distance metrics.
- `configs/experiments/acdc_unet_baseline.yaml` and
  `configs/experiments/acdc_bdslite_unet.yaml`: added ACDC experiment configs.
- `results/tables/baseline_results_month1.csv`: recorded the completed ISIC2018
  U-Net baseline and left Synapse/ACDC baseline rows pending.

Commands:
- `python3 -m pytest tests/unit/test_metrics.py -q`
- `.venv/bin/ruff check scripts/evaluate.py tests/unit/test_metrics.py`
- `.venv/bin/ruff format --check scripts/evaluate.py tests/unit/test_metrics.py`
- `python3 -m pytest tests/unit -q`
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/isic2018_unet_baseline/best.pt --split val`

Result:
- Unit tests passed: `9 passed`.
- Ruff check and format check passed for the changed Python files.
- ISIC2018 U-Net validation metrics on 519 slices:
  - DSC: 0.8633
  - IoU: 0.7866
  - HD95: 21.0114
  - ASSD: 7.1368
  - Boundary F1: 0.5343

Problems:
- This shell cannot access the GPU (`nvidia-smi` cannot communicate with the
  driver; `torch.cuda.is_available()` is `False`), so starting the 150-epoch
  Synapse baseline here would fall back to CPU and is not practical.

Next:
- Start the Synapse U-Net baseline on a CUDA-visible shell:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/unet_baseline.yaml`

## 2026-04-26 - synapse-cuda-preflight

Goal:
- Start the Synapse U-Net baseline and prevent accidental CPU fallback from this
  non-CUDA shell.

Changed:
- `scripts/train.py`: added explicit `--device {auto,cuda,cpu}` support and
  config-driven device selection; requesting CUDA now raises immediately when
  `torch.cuda.is_available()` is false.
- `configs/base.yaml`: documented the default `trainer.device: auto`.
- `configs/experiments/unet_baseline.yaml`: set `trainer.device: cuda` for the
  Synapse 150-epoch baseline.
- `README.md`: refreshed the Month 1 status and removed stale text saying
  Synapse/ACDC converters are not implemented.

Commands:
- `nvidia-smi`
- `python3 -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"`
- `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/unet_baseline.yaml`

Result:
- CUDA is not visible in this shell: `nvidia-smi` cannot communicate with the
  NVIDIA driver; PyTorch reports `cuda_available False` and `device_count 0`.
- The Synapse U-Net baseline command now fails fast with
  `RuntimeError: CUDA was requested, but torch.cuda.is_available() is False.`
  instead of silently starting a CPU run.

Problems:
- The actual 150-epoch Synapse baseline still requires a CUDA-visible shell.

Next:
- From a CUDA-visible shell, run:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/unet_baseline.yaml`

## 2026-04-26 - resource-profiling

Goal:
- Add Month 1 resource reporting required by `plan.md` so baseline tables can
  track efficiency as well as segmentation quality.

Changed:
- `src/bds_lite/evaluation/profiling.py`: added reusable parameter counting,
  hook-based Conv/ConvTranspose/Linear FLOP estimation, inference FPS/latency,
  and peak CUDA-memory profiling.
- `scripts/profile_model.py`: added CLI profiling from any experiment config,
  including optional retained-boundary-branch profiling for BDS-Lite variants.
- `tests/unit/test_profiling.py`: added fast unit coverage for parameter counting
  and CPU profiling.
- `README.md`: documented the new profiling command and updated status.
- `results/tables/resource_profile_isic2018_unet_baseline.csv`: recorded the
  first ISIC2018 U-Net resource profile from this CPU-only shell.

Commands:
- `python3 -m pytest tests/unit/test_profiling.py -q`
- `.venv/bin/ruff check src/bds_lite/evaluation/profiling.py scripts/profile_model.py tests/unit/test_profiling.py`
- `.venv/bin/ruff format src/bds_lite/evaluation/profiling.py scripts/profile_model.py tests/unit/test_profiling.py`
- `PYTHONPATH=src python3 scripts/profile_model.py --config configs/experiments/isic_unet_baseline.yaml --device cpu --warmup 1 --iterations 3`
- `python3 -m pytest tests/unit -q`

Result:
- Unit tests passed: `11 passed`.
- ISIC2018 U-Net profile, batch=1, CPU:
  - Params: 1,927,042
  - FLOPs: 15,969,615,872
  - FPS: 5.4674
  - Latency: 182.9037 ms
  - Peak GPU memory: not available on CPU

Problems:
- CUDA is still unavailable in this shell, so FPS and peak-memory numbers are
  CPU-only placeholders and should not be used in the final paper table.

Next:
- Rerun `scripts/profile_model.py` with `--device cuda` from a CUDA-visible shell
  for final FPS and peak-memory values.
- Start Synapse U-Net baseline from a CUDA-visible shell.

## 2026-04-26 - synapse-baseline-eval

Goal:
- Record the completed Synapse U-Net baseline in the Month 1 tracking table.

Changed:
- `results/checkpoints/synapse_unet_baseline/eval_val.csv`: evaluated the best
  Synapse U-Net checkpoint on the validation split.
- `results/tables/baseline_results_month1.csv`: filled the Synapse U-Net row.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_unet_baseline/best.pt --split val`
- `python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"`

Result:
- Synapse U-Net best checkpoint was selected at epoch 36 by validation DSC.
- Validation metrics over 473 slices:
  - DSC: 0.8448
  - IoU: 0.8255
  - HD95: 10.4754
  - ASSD: 3.6543
  - Boundary F1: 0.8548

Problems:
- CUDA is still not visible in this shell (`torch.cuda.is_available() == False`),
  so ACDC baseline training was not started here.

Next:
- Start the ACDC U-Net baseline from a CUDA-visible shell:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/acdc_unet_baseline.yaml --device cuda`

## 2026-04-26 - acdc-baseline-eval

Goal:
- Record the completed ACDC U-Net baseline and close the first three-dataset
  U-Net baseline pass.

Changed:
- `results/checkpoints/acdc_unet_baseline/eval_val.csv`: evaluated the best ACDC
  U-Net checkpoint on the validation split.
- `results/tables/baseline_results_month1.csv`: filled the ACDC U-Net row.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/acdc_unet_baseline/best.pt --split val`
- `python3 -m pytest tests/unit -q`

Result:
- ACDC U-Net best checkpoint was selected at epoch 25 by validation DSC.
- Validation metrics over 186 slices:
  - DSC: 0.7919
  - IoU: 0.7194
  - HD95: 8.3924
  - ASSD: 2.6993
  - Boundary F1: 0.8555

Problems:
- Final epoch DSC dropped to 0.7639 while best epoch was 25, suggesting the
  baseline starts overfitting early on this split.

Next:
- Run or profile baseline resource metrics on a CUDA-visible shell.
- Start the first BDS-Lite U-Net baseline comparison, beginning with Synapse:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/bdslite_unet.yaml --device cuda`

## 2026-04-26 - synapse-bdslite-v1-eval

Goal:
- Evaluate the completed Synapse BDS-Lite U-Net v1 run and compare it with the
  Synapse U-Net baseline.

Changed:
- `results/checkpoints/synapse_bdslite_unet/eval_val.csv`: evaluated the best
  BDS-Lite checkpoint on the validation split.
- `results/tables/synapse_bdslite_v1.csv`: added the first direct Synapse
  U-Net vs BDS-Lite v1 comparison table.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_unet/best.pt --split val`
- `PYTHONPATH=src python3 -c "import torch; ckpt=torch.load('results/checkpoints/synapse_bdslite_unet/best.pt', map_location='cpu'); print({k: ckpt.get(k) for k in ['epoch','best_metric','metric_name']})"`

Result:
- Synapse BDS-Lite best checkpoint was selected at epoch 51 by validation DSC.
- Validation metrics over 473 slices:
  - DSC: 0.8362
  - IoU: 0.8175
  - HD95: 11.4823
  - ASSD: 4.7322
  - Boundary F1: 0.8507
- Compared with Synapse U-Net baseline, BDS-Lite v1 is worse:
  - DSC: -0.0087
  - IoU: -0.0081
  - HD95: +1.0069
  - ASSD: +1.0779
  - Boundary F1: -0.0041

Problems:
- The full boundary+distillation v1 setting does not support the core paper
  claim on Synapse. It improves neither overlap nor boundary-sensitive metrics.

Next:
- Run a narrower Synapse ablation before training other BDS-Lite datasets:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/synapse_bdslite_boundary_low.yaml --device cuda`

## 2026-04-26 - synapse-boundary-low-config

Goal:
- Prepare the immediate follow-up Synapse ablation after BDS-Lite v1 underperformed
  the U-Net baseline.

Changed:
- `configs/experiments/synapse_bdslite_boundary_low.yaml`: added a BDS-Lite
  ablation with lower boundary supervision and no feature distillation
  (`boundary_weight: 0.1`, `distill_weight: 0.0`).

Commands:
- `PYTHONPATH=src python3 -c "from bds_lite.utils import load_config; from bds_lite.training import build_model; c=load_config('configs/experiments/synapse_bdslite_boundary_low.yaml'); m=build_model(c); print(c['experiment']['name'], c['loss']['boundary_weight'], c['loss']['distill_weight'], type(m).__name__)"`

Result:
- Config loads and builds `BDSLiteUNet` with `boundary_weight: 0.1` and
  `distill_weight: 0.0`.

Problems:
- Training was not started in this shell because long Synapse runs require
  CUDA-visible execution.

Next:
- Run the new Synapse ablation:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/synapse_bdslite_boundary_low.yaml --device cuda`

## 2026-04-26 - synapse-boundary-low-eval

Goal:
- Evaluate the low-boundary/no-distillation Synapse ablation and decide the next
  method adjustment.

Changed:
- `results/checkpoints/synapse_bdslite_boundary_low/eval_val.csv`: evaluated the
  best checkpoint on Synapse validation.
- `results/tables/synapse_bdslite_v1.csv`: added the low-boundary ablation row.
- `src/bds_lite/training/losses.py`: wired `surface_weight` through
  `BDSLiteLoss` using an SDF-weighted foreground probability error.
- `src/bds_lite/data/datasets.py` and `scripts/train.py`: added fail-fast SDF
  requirements when `surface_weight > 0`.
- `configs/experiments/synapse_bdslite_boundary_surface_low.yaml`: added the
  next Synapse ablation with boundary supervision plus low surface loss.
- `tests/unit/test_losses.py`: added coverage for the surface-loss path.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_low/best.pt --split val`

Result:
- Synapse BDS-Lite boundary-low best checkpoint was selected at epoch 32 by
  validation DSC.
- Validation metrics over 473 slices:
  - DSC: 0.8489
  - IoU: 0.8302
  - HD95: 12.5559
  - ASSD: 5.0946
  - Boundary F1: 0.8621
- Compared with Synapse U-Net baseline, boundary-low improves overlap and local
  boundary agreement but worsens distance metrics:
  - DSC: +0.0040
  - IoU: +0.0047
  - HD95: +2.0806
  - ASSD: +1.4403
  - Boundary F1: +0.0073

Problems:
- The ablation is promising for DSC/Boundary F1 but still fails the paper's
  HD95/ASSD claim.

Next:
- Run the surface-loss ablation:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/synapse_bdslite_boundary_surface_low.yaml --device cuda`

## 2026-04-26 - synapse-surface-low-eval

Goal:
- Evaluate the first surface-loss Synapse ablation and adjust the next distance
  metric experiment.

Changed:
- `results/checkpoints/synapse_bdslite_boundary_surface_low/eval_val.csv`:
  evaluated the best checkpoint on Synapse validation.
- `results/tables/synapse_bdslite_v1.csv`: added the weak absolute surface-loss
  ablation row.
- `src/bds_lite/training/losses.py`: replaced the weak absolute-error surface
  term with a signed normalized SDF surface loss over valid foreground channels.
- `configs/experiments/synapse_bdslite_boundary_surface_signed.yaml`: added the
  next Synapse ablation with signed surface loss.
- `tests/unit/test_losses.py`: updated surface-loss tests for signed loss and
  absent SDF channels.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_surface_low/best.pt --split val`

Result:
- Synapse BDS-Lite weak surface-loss best checkpoint was selected at epoch 36 by
  validation DSC.
- Validation metrics over 473 slices:
  - DSC: 0.8470
  - IoU: 0.8270
  - HD95: 11.2127
  - ASSD: 4.3639
  - Boundary F1: 0.8573
- Compared with the boundary-low ablation, weak surface loss improved HD95/ASSD
  but reduced DSC and Boundary F1.
- Compared with U-Net, it still misses the core distance claim:
  - DSC: +0.0022
  - IoU: +0.0015
  - HD95: +0.7373
  - ASSD: +0.7096
  - Boundary F1: +0.0025

Problems:
- The logged surface scalar was around `1e-6`, so the first surface loss was too
  weak to materially shape training.

Next:
- Run the signed surface-loss ablation:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/synapse_bdslite_boundary_surface_signed.yaml --device cuda`

## 2026-04-26 - synapse-surface-signed-eval

Goal:
- Evaluate the signed surface-loss Synapse ablation and decide whether the next
  step is checkpoint selection or loss-strength adjustment.

Changed:
- `results/checkpoints/synapse_bdslite_boundary_surface_signed/eval_val.csv`:
  evaluated the best checkpoint on Synapse validation.
- `results/checkpoints/synapse_bdslite_boundary_surface_signed/eval_test.csv`:
  evaluated the best checkpoint on Synapse test.
- `results/tables/synapse_bdslite_v1.csv`: added the signed-surface ablation
  row.
- `configs/experiments/synapse_bdslite_boundary_surface_strong.yaml`: added the
  next Synapse ablation with stronger signed surface supervision.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_surface_signed/best.pt --split val`
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_surface_signed/best.pt --split test`
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_surface_signed/epoch_0025.pt --split val --output results/checkpoints/synapse_bdslite_boundary_surface_signed/eval_val_epoch_0025.csv`
- Repeated the same validation sweep for saved epochs 50, 75, 100, 125, and
  150.
- `PYTHONPATH=src python3 -c "from bds_lite.utils import load_config; from bds_lite.training import build_model; c=load_config('configs/experiments/synapse_bdslite_boundary_surface_strong.yaml'); m=build_model(c); print(c['experiment']['name'], c['loss']['surface_weight'], type(m).__name__)"`
- `PYTHONPATH=src python3 -m pytest tests/unit/test_losses.py`
- `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/synapse_bdslite_boundary_surface_strong.yaml --device cuda`

Result:
- Synapse BDS-Lite signed-surface best checkpoint was selected at epoch 25 by
  validation DSC.
- Validation metrics over 473 slices:
  - DSC: 0.8448
  - IoU: 0.8262
  - HD95: 11.8402
  - ASSD: 4.3969
  - Boundary F1: 0.8572
- Test metrics:
  - DSC: 0.8506
  - IoU: 0.8330
  - HD95: 10.9248
  - ASSD: 4.3173
  - Boundary F1: 0.8628
- The saved-epoch sweep showed epoch 25 was also the best saved HD95 checkpoint
  for this run, so the remaining HD95/ASSD gap is not just a checkpoint
  selection artifact.
- The stronger surface config loads and builds `BDSLiteUNet` with
  `surface_weight: 1.0`.
- Loss tests passed: `2 passed`.

Problems:
- Signed surface supervision preserved baseline-like DSC but still worsened
  distance metrics relative to the Synapse U-Net baseline.
- The signed surface scalar quickly shrank to around `-8e-4`, so
  `surface_weight: 0.05` was likely still too weak to materially shape the
  learned boundary geometry.
- CUDA is not visible in this shell, so the stronger surface training command
  failed fast with `RuntimeError: CUDA was requested, but
  torch.cuda.is_available() is False.`

Next:
- Run the stronger signed surface-loss ablation:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/synapse_bdslite_boundary_surface_strong.yaml --device cuda`

## 2026-04-26 - synapse-surface-strong-eval

Goal:
- Evaluate the stronger signed surface-loss Synapse ablation and decide whether
  more surface weighting is justified.

Changed:
- `results/checkpoints/synapse_bdslite_boundary_surface_strong/eval_val.csv`:
  evaluated the best checkpoint on Synapse validation.
- `results/tables/synapse_bdslite_v1.csv`: added the strong signed-surface
  ablation row.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_surface_strong/best.pt --split val`
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_surface_strong/epoch_0025.pt --split val --output results/checkpoints/synapse_bdslite_boundary_surface_strong/eval_val_epoch_0025.csv`
- Repeated the same validation sweep for saved epochs 50, 75, 100, 125, and
  150.

Result:
- Synapse BDS-Lite strong signed-surface best checkpoint was selected at epoch
  25 by validation DSC.
- Validation metrics over 473 slices:
  - DSC: 0.8405
  - IoU: 0.8232
  - HD95: 11.1886
  - ASSD: 4.2412
  - Boundary F1: 0.8499
- Compared with weaker signed surface, strong surface improved HD95 and ASSD but
  reduced DSC and Boundary F1.
- Compared with U-Net, it still misses the distance claim:
  - DSC: -0.0043
  - IoU: -0.0023
  - HD95: +0.7132
  - ASSD: +0.5869
  - Boundary F1: -0.0049
- The saved-epoch sweep did not find a better HD95 checkpoint than `best.pt`.

Problems:
- Increasing surface weight trades away overlap/local boundary quality without
  closing the HD95/ASSD gap.
- Remaining HD95 is concentrated in the harder foreground classes, so the next
  useful diagnostic is whether distance errors come from small disconnected
  false-positive components.

Next:
- Add largest-connected-component evaluation as a diagnostic postprocess and
  compare U-Net vs BDS-Lite validation metrics with the same postprocess.

## 2026-04-26 - synapse-lcc-diagnostic

Goal:
- Check whether the Synapse HD95/ASSD gap is caused mostly by small disconnected
  false-positive components.

Changed:
- `scripts/evaluate.py`: added `--keep-largest-component`, which keeps only the
  largest predicted connected component for each foreground class and writes to
  `eval_<split>_lcc.csv` by default.
- `results/checkpoints/synapse_unet_baseline/eval_val_lcc.csv`: evaluated U-Net
  with the LCC postprocess.
- `results/checkpoints/synapse_bdslite_boundary_surface_strong/eval_val_lcc.csv`:
  evaluated strong-surface BDS-Lite with the same LCC postprocess.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_unet_baseline/best.pt --split val --keep-largest-component`
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_surface_strong/best.pt --split val --keep-largest-component`
- `.venv/bin/ruff check scripts/evaluate.py`
- `.venv/bin/ruff format --check scripts/evaluate.py`
- `PYTHONPATH=src python3 -m pytest tests/unit/test_losses.py tests/unit/test_metrics.py`

Result:
- U-Net with LCC:
  - DSC: 0.8432
  - IoU: 0.8244
  - HD95: 11.0697
  - ASSD: 4.2826
  - Boundary F1: 0.8575
- Strong-surface BDS-Lite with LCC:
  - DSC: 0.8396
  - IoU: 0.8228
  - HD95: 11.6565
  - ASSD: 4.7629
  - Boundary F1: 0.8515
- LCC worsened HD95/ASSD for both methods, so the distance errors are not mainly
  removable small-component false positives.
- Ruff passed and focused tests passed: `5 passed`.

Problems:
- Postprocessing is not the right fix for the current Synapse distance gap.

Next:
- Try a loss/architecture adjustment rather than inference cleanup. The most
  direct next ablation is stronger signed surface supervision only if we accept
  further DSC tradeoff; otherwise pivot to a boundary feature/gating change.

## 2026-04-26 - synapse-strong-surface-tiny-boundary-config

Goal:
- Prepare the next targeted Synapse loss ablation after strong surface improved
  ASSD but reduced DSC and still missed U-Net HD95.

Changed:
- `configs/experiments/synapse_bdslite_boundary_surface_strong_tiny_boundary.yaml`:
  added a CUDA-ready ablation with `boundary_weight: 0.05`,
  `distill_weight: 0.0`, and `surface_weight: 1.0`.

Commands:
- `PYTHONPATH=src python3 -c "from bds_lite.utils import load_config; from bds_lite.training import build_model; c=load_config('configs/experiments/synapse_bdslite_boundary_surface_strong_tiny_boundary.yaml'); m=build_model(c); print(c['experiment']['name'], c['loss']['boundary_weight'], c['loss']['surface_weight'], type(m).__name__)"`

Result:
- Config loads and builds `BDSLiteUNet` with `boundary_weight: 0.05` and
  `surface_weight: 1.0`.

Problems:
- Training was not started in this shell because CUDA is not visible here.

Next:
- Run the tiny-boundary strong-surface ablation from a CUDA-visible shell:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/synapse_bdslite_boundary_surface_strong_tiny_boundary.yaml --device cuda`

## 2026-04-26 - synapse-strong-surface-tiny-boundary-eval

Goal:
- Evaluate the completed tiny-boundary strong-surface Synapse ablation and
  update the ablation table.

Changed:
- `results/checkpoints/synapse_bdslite_boundary_surface_strong_tiny_boundary/eval_val.csv`:
  evaluated the best checkpoint on Synapse validation.
- `results/tables/synapse_bdslite_v1.csv`: added the tiny-boundary
  strong-surface ablation row.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_surface_strong_tiny_boundary/best.pt --split val`

Result:
- Synapse BDS-Lite tiny-boundary strong-surface best checkpoint was selected at
  epoch 47 by validation DSC.
- Validation metrics over 473 slices:
  - DSC: 0.8458
  - IoU: 0.8254
  - HD95: 11.0865
  - ASSD: 4.0206
  - Boundary F1: 0.8562
- Compared with U-Net, this recovers DSC while slightly improving Boundary F1:
  - DSC: +0.0010
  - IoU: -0.0001
  - HD95: +0.6111
  - ASSD: +0.3663
  - Boundary F1: +0.0014

Problems:
- The ablation still misses the paper's HD95/ASSD claim on Synapse.
- Loss-weight tuning is showing diminishing returns: stronger surface terms
  reduce distance error somewhat but do not beat the U-Net baseline.

Next:
- Stop pure loss-weight tuning for Synapse and make the next ablation an
  architecture-level boundary feature/gating change.

## 2026-04-26 - boundary-gate-ablation-config

Goal:
- Prepare the next Synapse ablation as a small architecture-level
  boundary-feature gate instead of another loss-weight-only tweak.

Changed:
- `src/bds_lite/models/bds_lite.py`: added `BoundaryFeatureGate`, a neutral
  initialized 1x1 gate driven by the existing segmentation-side boundary proxy.
  The boundary decoder remains removable at inference; when enabled, the retained
  inference overhead is the tiny gate path only.
- `src/bds_lite/training/builders.py`: wired `model.boundary_gate` and
  `model.boundary_gate_scale` from config.
- `configs/experiments/synapse_bdslite_boundary_gate_surface.yaml`: added the
  next CUDA-ready ablation using low boundary supervision, feature distillation,
  strong signed surface loss, and the retained boundary-proxy gate.
- `tests/unit/test_builders.py`: added coverage for enabling the gate through
  config.

Commands:
- `PYTHONPATH=src python3 -m pytest tests/unit/test_builders.py -q`
- `PYTHONPATH=src python3 -c "import torch; from bds_lite.utils import load_config; from bds_lite.training import build_model; c=load_config('configs/experiments/synapse_bdslite_boundary_gate_surface.yaml'); m=build_model(c); y=m(torch.randn(2,1,224,224), return_boundary=False); z=m(torch.randn(2,1,224,224), return_boundary=True); print(c['experiment']['name'], m.boundary_gate is not None, tuple(y['seg_logits'].shape), sorted(z.keys()))"`
- `PYTHONPATH=src python3 -m pytest tests/unit/test_losses.py tests/unit/test_metrics.py -q`
- `.venv/bin/ruff check src/bds_lite/models/bds_lite.py src/bds_lite/training/builders.py tests/unit/test_builders.py`
- `.venv/bin/ruff format --check src/bds_lite/models/bds_lite.py src/bds_lite/training/builders.py tests/unit/test_builders.py`

Result:
- Builder tests passed: `3 passed`.
- Focused loss/metric tests passed: `5 passed`.
- Gate config loads, builds, and forwards:
  `synapse_bdslite_boundary_gate_surface True (2, 9, 224, 224)`.
- Ruff check passed and touched Python files were already formatted.

Problems:
- CUDA is not visible in this shell, so the new 150-epoch ablation was prepared
  but not launched here.

Next:
- Run the boundary-gate surface ablation from a CUDA-visible shell:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/synapse_bdslite_boundary_gate_surface.yaml --device cuda`

## 2026-04-26 - boundary-gate-ablation-eval

Goal:
- Evaluate the completed boundary-gate Synapse ablation and compare it with the
  prior BDS-Lite variants.

Changed:
- `results/checkpoints/synapse_bdslite_boundary_gate_surface/eval_val.csv`:
  evaluated the best checkpoint on Synapse validation.
- `results/tables/synapse_bdslite_v1.csv`: added the boundary-gate
  strong-surface row.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_gate_surface/best.pt --split val`

Result:
- Synapse BDS-Lite boundary-gate best checkpoint was selected at epoch 23 by
  validation DSC.
- Validation metrics over 473 slices:
  - DSC: 0.8427
  - IoU: 0.8234
  - HD95: 10.7395
  - ASSD: 3.8490
  - Boundary F1: 0.8488
- Compared with the previous tiny-boundary strong-surface run, the gate improves
  distance metrics:
  - HD95: 11.0865 -> 10.7395
  - ASSD: 4.0206 -> 3.8490
- Compared with U-Net, it remains slightly worse on the main distance metrics:
  - DSC: -0.0021
  - IoU: -0.0021
  - HD95: +0.2641
  - ASSD: +0.1947
  - Boundary F1: -0.0061

Problems:
- The architecture-level gate moves the right metric but still does not satisfy
  the Synapse HD95/ASSD claim.
- Because this checkpoint is selected by validation DSC, it may not be the best
  distance-metric checkpoint for this run.

Next:
- Sweep saved epochs for the boundary-gate run to check whether a non-DSC
  checkpoint beats U-Net on HD95/ASSD.

Follow-up:
- Started an epoch-25 validation sweep from this CPU-only shell, but the tool-side
  process did not produce `eval_val_epoch_0025.csv` in a reasonable time. Defer
  the saved-epoch sweep to a CUDA-visible terminal or run it with a timeout.
- Epoch 25 was evaluated from the CUDA-visible terminal:
  - DSC: 0.8260
  - IoU: 0.8065
  - HD95: 12.0601
  - ASSD: 4.4425
  - Boundary F1: 0.8379
- Epoch 25 is worse than the epoch-23 `best.pt` checkpoint on all mean metrics,
  so checkpoint selection is not hiding an immediate distance-metric win.

Next diagnostic:
- Boundary gate vs U-Net per-class HD95 deltas:
  - improved: label 1 (-1.3514), label 3 (-0.7667), label 7 (-0.9686)
  - roughly neutral: label 8 (+0.0670)
  - worsened: label 2 (+1.1016), label 4 (+0.2821), label 5 (+1.2266),
    label 6 (+2.5220)
- The gate moves the right metrics for several classes but is too strong or too
  poorly targeted for the harder foreground classes.

Next:
- Try a damped retained boundary gate with the same loss setting but
  `boundary_gate_scale: 0.25`.

## 2026-04-26 - boundary-gate-damped-config

Goal:
- Prepare a lower-scale retained boundary-gate ablation after the full-scale gate
  improved some per-class distance metrics but hurt labels 2, 5, and 6.

Changed:
- `configs/experiments/synapse_bdslite_boundary_gate_damped_surface.yaml`: added
  a CUDA-ready ablation matching the boundary-gate surface setup but reducing
  `boundary_gate_scale` from `0.5` to `0.25`.

Commands:
- `PYTHONPATH=src python3 -c "from bds_lite.utils import load_config; from bds_lite.training import build_model; c=load_config('configs/experiments/synapse_bdslite_boundary_gate_damped_surface.yaml'); m=build_model(c); print(c['experiment']['name'], m.boundary_gate is not None, m.boundary_gate.scale)"`
- `PYTHONPATH=src python3 -m pytest tests/unit/test_builders.py -q`

Result:
- Config loads and builds with the damped gate enabled:
  `synapse_bdslite_boundary_gate_damped_surface True 0.25`.
- Builder tests passed: `3 passed`.

Problems:
- CUDA is not visible in this shell, so training must be launched from the
  CUDA-visible terminal.

Next:
- Run the damped gate ablation:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/synapse_bdslite_boundary_gate_damped_surface.yaml --device cuda`

## 2026-04-26 - boundary-gate-damped-eval

Goal:
- Evaluate the completed damped boundary-gate Synapse ablation.

Changed:
- `results/checkpoints/synapse_bdslite_boundary_gate_damped_surface/eval_val.csv`:
  evaluated the best checkpoint on Synapse validation.
- `results/tables/synapse_bdslite_v1.csv`: added the damped boundary-gate row.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_gate_damped_surface/best.pt --split val`

Result:
- Synapse BDS-Lite damped boundary-gate best checkpoint was selected at epoch 42
  by validation DSC.
- Validation metrics over 473 slices:
  - DSC: 0.8545
  - IoU: 0.8350
  - HD95: 11.1790
  - ASSD: 4.4884
  - Boundary F1: 0.8699
- Compared with U-Net:
  - DSC: +0.0096
  - IoU: +0.0094
  - HD95: +0.7036
  - ASSD: +0.8341
  - Boundary F1: +0.0151
- This is the strongest overlap/local-boundary result so far, but it fails the
  paper's HD95/ASSD claim.

Problems:
- The distance failure is concentrated in label 6 (`HD95=23.6429`,
  `ASSD=12.6329`) and label 8 (`HD95=21.2516`, `ASSD=8.8077`).
- Gate scaling alone trades between DSC/Boundary F1 and distance metrics rather
  than improving both.

Next:
- Add validation-time checkpoint selection by a composite score that includes
  mean foreground Dice and boundary loss/proxy, or run saved-epoch evaluation
  for the damped run to check whether a non-DSC checkpoint keeps the high DSC
  while reducing HD95/ASSD.

Follow-up:
- Evaluated the nearest saved checkpoint after the epoch-42 best checkpoint:
  `epoch_0050.pt`.
- Epoch 50 validation metrics:
  - DSC: 0.8354
  - IoU: 0.8154
  - HD95: 11.2116
  - ASSD: 4.5417
  - Boundary F1: 0.8494
- Epoch 50 is worse than `best.pt` on all mean metrics, so the distance issue is
  not fixed by the nearest saved checkpoint.

Next:
- Add a direct SDF-distance error term to penalize far-away foreground
  probability errors more strongly than the signed surface loss does.

## 2026-04-26 - sdf-distance-error-loss

Goal:
- Prepare a targeted loss ablation for HD95/ASSD outliers after gate scaling
  improved DSC and Boundary F1 but left large distance errors on labels 6 and 8.

Changed:
- `src/bds_lite/training/losses.py`: added `SdfDistanceErrorLoss`, a
  nonnegative SDF-magnitude-weighted foreground probability error term, plus
  `LossWeights.distance` and `loss.distance_weight` config parsing.
- `scripts/train.py`: training now requires SDF targets when either
  `surface_weight` or `distance_weight` is enabled.
- `configs/base.yaml`: added `loss.distance_weight: 0.0`.
- `tests/unit/test_losses.py`: added coverage for the distance-error loss and
  all-zero absent-class SDF handling.
- `configs/experiments/synapse_bdslite_boundary_gate_damped_distance.yaml`:
  added a CUDA-ready ablation using the damped gate plus
  `distance_weight: 0.2`.

Commands:
- `PYTHONPATH=src python3 -m pytest tests/unit/test_losses.py -q`
- `PYTHONPATH=src python3 -c "import torch; from bds_lite.utils import load_config; from bds_lite.training import build_model, BDSLiteLoss; from bds_lite.training.losses import weights_from_config; c=load_config('configs/experiments/synapse_bdslite_boundary_gate_damped_distance.yaml'); m=build_model(c); loss_fn=BDSLiteLoss(weights_from_config(c)); outputs=m(torch.randn(2,1,224,224), return_boundary=True); batch={'mask': torch.randint(0,9,(2,224,224)), 'boundary': torch.rand(2,1,224,224), 'sdf': torch.randn(2,8,224,224)}; loss, scalars=loss_fn(outputs, batch); print(c['experiment']['name'], m.boundary_gate.scale, round(float(loss), 4), sorted(scalars))"`
- `PYTHONPATH=src python3 -m pytest tests/unit/test_builders.py -q`
- `.venv/bin/ruff check src/bds_lite/training/losses.py scripts/train.py tests/unit/test_losses.py`
- `.venv/bin/ruff format --check src/bds_lite/training/losses.py scripts/train.py tests/unit/test_losses.py`
- `PYTHONPATH=src python3 -m pytest tests/unit -q`

Result:
- Loss tests passed: `4 passed`.
- Builder tests passed: `3 passed`.
- Full unit tests passed: `16 passed`.
- Ruff check and format check passed.
- Config/loss smoke check produced all expected scalar keys, including
  `distance`.

Problems:
- CUDA is not visible in this shell, so training must be launched from the
  CUDA-visible terminal.

Next:
- Run the damped-gate distance-error ablation:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/synapse_bdslite_boundary_gate_damped_distance.yaml --device cuda`

## 2026-04-26 - boundary-gate-damped-distance-eval

Goal:
- Evaluate the completed damped boundary-gate distance-error Synapse ablation
  and decide whether the direct SDF-distance penalty fixes the HD95/ASSD gap.

Changed:
- `results/checkpoints/synapse_bdslite_boundary_gate_damped_distance/eval_val.csv`:
  evaluated the best checkpoint on Synapse validation.
- `results/tables/synapse_bdslite_v1.csv`: added the damped boundary-gate
  distance-error row.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_gate_damped_distance/best.pt --split val`

Result:
- Synapse BDS-Lite damped boundary-gate distance-error best checkpoint was
  selected at epoch 25 by validation DSC.
- Validation metrics over 473 slices:
  - DSC: 0.8463
  - IoU: 0.8274
  - HD95: 11.0551
  - ASSD: 4.1267
  - Boundary F1: 0.8568
- Compared with U-Net:
  - DSC: +0.0015
  - IoU: +0.0018
  - HD95: +0.5798
  - ASSD: +0.4724
  - Boundary F1: +0.0020
- Compared with the damped-gate strong-surface run, the distance-error term
  improves HD95 and ASSD but gives back most of the DSC and Boundary F1 gains.

Problems:
- The direct SDF-distance penalty still does not beat the U-Net baseline on
  HD95/ASSD.
- Distance errors remain largest for labels 6 and 8, with label 6 ASSD still
  high at 8.6781 and label 8 HD95 high at 21.5551.

Next:
- Stop Synapse-only loss/gate tuning unless a new diagnostic changes the
  picture. The immediate next research task is to run the best promising
  BDS-Lite setting on ACDC or ISIC2018 to test whether the boundary-quality
  trend appears on a second dataset.

## 2026-04-26 - acdc-damped-distance-config

Goal:
- Prepare the next cross-dataset experiment using the best-promising Synapse
  damped boundary-gate distance-error setup on ACDC.

Changed:
- `configs/experiments/acdc_bdslite_boundary_gate_damped_distance.yaml`: added
  an ACDC ablation with `boundary_gate_scale: 0.25`, low boundary supervision,
  feature distillation, strong signed surface loss, and `distance_weight: 0.2`.

Commands:
- `PYTHONPATH=src python3 -c "import torch; from bds_lite.utils import load_config; from bds_lite.training import build_model, BDSLiteLoss; from bds_lite.training.losses import weights_from_config; c=load_config('configs/experiments/acdc_bdslite_boundary_gate_damped_distance.yaml'); m=build_model(c); loss_fn=BDSLiteLoss(weights_from_config(c)); y=m(torch.randn(2,1,224,224), return_boundary=True); batch={'mask': torch.randint(0,4,(2,224,224)), 'boundary': torch.rand(2,1,224,224), 'sdf': torch.randn(2,3,224,224)}; loss, scalars=loss_fn(y,batch); print(c['experiment']['name'], c['dataset']['num_classes'], m.boundary_gate.scale, tuple(y['seg_logits'].shape), round(float(loss), 4), sorted(scalars))"`
- `PYTHONPATH=src python3 -m pytest tests/unit/test_builders.py tests/unit/test_losses.py -q`

Result:
- Config loads and builds `BDSLiteUNet` for ACDC with four segmentation classes
  and a damped boundary gate scale of 0.25.
- Forward/loss smoke test produced all expected scalar keys, including
  `surface` and `distance`.
- Focused tests passed: `7 passed`.

Problems:
- CUDA is not visible in this shell, so the 150-epoch ACDC run must be launched
  from the CUDA-visible terminal.

Next:
- Run:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/acdc_bdslite_boundary_gate_damped_distance.yaml --device cuda`

## 2026-04-26 - acdc-damped-distance-eval

Goal:
- Evaluate the completed ACDC damped boundary-gate distance-error ablation and
  check whether it provides the second-dataset boundary-quality trend required
  by `plan.md`.

Changed:
- `results/checkpoints/acdc_bdslite_boundary_gate_damped_distance/eval_val.csv`:
  evaluated the best checkpoint on ACDC validation.
- `results/tables/acdc_bdslite_v1.csv`: added the first ACDC U-Net vs BDS-Lite
  direct comparison table.
- `configs/experiments/isic_bdslite_boundary_gate_damped_distance.yaml`: added
  the same damped boundary-gate distance-error ablation for ISIC2018.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/acdc_bdslite_boundary_gate_damped_distance/best.pt --split val`
- `PYTHONPATH=src python3 -c "import torch; from bds_lite.utils import load_config; from bds_lite.training import build_model, BDSLiteLoss; from bds_lite.training.losses import weights_from_config; c=load_config('configs/experiments/isic_bdslite_boundary_gate_damped_distance.yaml'); m=build_model(c); loss_fn=BDSLiteLoss(weights_from_config(c)); y=m(torch.randn(2,3,224,224), return_boundary=True); batch={'mask': torch.randint(0,2,(2,224,224)), 'boundary': torch.rand(2,1,224,224), 'sdf': torch.randn(2,1,224,224)}; loss, scalars=loss_fn(y,batch); print(c['experiment']['name'], c['dataset']['num_classes'], c['dataset']['in_channels'], m.boundary_gate.scale, tuple(y['seg_logits'].shape), sorted(scalars))"`
- `PYTHONPATH=src python3 -m pytest tests/unit/test_builders.py tests/unit/test_losses.py -q`

Result:
- ACDC BDS-Lite damped boundary-gate distance-error best checkpoint was selected
  at epoch 47 by validation DSC.
- Validation metrics over 186 slices:
  - DSC: 0.7837
  - IoU: 0.7081
  - HD95: 8.5769
  - ASSD: 2.6966
  - Boundary F1: 0.8450
- Compared with ACDC U-Net:
  - DSC: -0.0082
  - IoU: -0.0113
  - HD95: +0.1845
  - ASSD: -0.0026
  - Boundary F1: -0.0104
- ISIC2018 config loads and builds with 3 input channels, 2 segmentation
  classes, damped boundary gate scale 0.25, and all expected loss scalar keys.
- Focused tests passed: `7 passed`.

Problems:
- This setting does not support the cross-dataset claim on ACDC. ASSD is
  essentially tied/slightly better, but DSC, HD95, and Boundary F1 are worse.

Next:
- Run the same ablation on ISIC2018, where boundary quality is the most relevant
  metric:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/isic_bdslite_boundary_gate_damped_distance.yaml --device cuda`

## 2026-04-26 - isic-damped-distance-eval

Goal:
- Evaluate the completed ISIC2018 damped boundary-gate distance-error ablation
  and check whether the boundary-quality trend appears on dermoscopy.

Changed:
- `results/checkpoints/isic2018_bdslite_boundary_gate_damped_distance/eval_val.csv`:
  evaluated the best checkpoint on ISIC2018 validation.
- `results/tables/isic_bdslite_v1.csv`: added the first ISIC2018 U-Net vs
  BDS-Lite direct comparison table.
- `configs/experiments/isic_bdslite_boundary_gate_damped_surface.yaml`: added a
  follow-up ISIC2018 ablation that keeps the damped boundary gate and strong
  signed surface term but removes the SDF distance-error term.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/isic2018_bdslite_boundary_gate_damped_distance/best.pt --split val`
- `PYTHONPATH=src python3 -c "import torch; from bds_lite.utils import load_config; from bds_lite.training import build_model, BDSLiteLoss; from bds_lite.training.losses import weights_from_config; c=load_config('configs/experiments/isic_bdslite_boundary_gate_damped_surface.yaml'); m=build_model(c); loss_fn=BDSLiteLoss(weights_from_config(c)); y=m(torch.randn(2,3,224,224), return_boundary=True); batch={'mask': torch.randint(0,2,(2,224,224)), 'boundary': torch.rand(2,1,224,224), 'sdf': torch.randn(2,1,224,224)}; loss, scalars=loss_fn(y,batch); print(c['experiment']['name'], c['loss']['distance_weight'], m.boundary_gate.scale, tuple(y['seg_logits'].shape), sorted(scalars))"`
- `PYTHONPATH=src python3 -m pytest tests/unit/test_builders.py tests/unit/test_losses.py -q`

Result:
- ISIC2018 BDS-Lite damped boundary-gate distance-error best checkpoint was
  selected at epoch 56 by validation DSC.
- Validation metrics over 519 slices:
  - DSC: 0.8649
  - IoU: 0.7872
  - HD95: 21.5486
  - ASSD: 6.9734
  - Boundary F1: 0.5317
- Compared with ISIC2018 U-Net:
  - DSC: +0.0016
  - IoU: +0.0006
  - HD95: +0.5372
  - ASSD: -0.1634
  - Boundary F1: -0.0025
- Follow-up config loads and builds with `distance_weight: 0.0`, damped boundary
  gate scale 0.25, and expected loss scalar keys.
- Focused tests passed: `7 passed`.

Problems:
- This setting gives a small DSC/ASSD improvement but still misses the direct
  boundary-quality claim because HD95 and Boundary F1 are worse.

Next:
- Run the ISIC2018 damped-gate strong-surface ablation without distance-error:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/isic_bdslite_boundary_gate_damped_surface.yaml --device cuda`

## 2026-04-26 - isic-damped-surface-eval

Goal:
- Evaluate the completed ISIC2018 damped boundary-gate strong-surface ablation
  without the SDF distance-error term.

Changed:
- `results/checkpoints/isic2018_bdslite_boundary_gate_damped_surface/eval_val.csv`:
  evaluated the best checkpoint on ISIC2018 validation.
- `results/tables/isic_bdslite_v1.csv`: added the damped boundary-gate
  strong-surface row.
- `configs/experiments/acdc_bdslite_boundary_gate_damped_surface.yaml`: added
  the same surface-only damped-gate ablation for ACDC.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/isic2018_bdslite_boundary_gate_damped_surface/best.pt --split val`
- `PYTHONPATH=src python3 -c "import torch; from bds_lite.utils import load_config; from bds_lite.training import build_model, BDSLiteLoss; from bds_lite.training.losses import weights_from_config; c=load_config('configs/experiments/acdc_bdslite_boundary_gate_damped_surface.yaml'); m=build_model(c); loss_fn=BDSLiteLoss(weights_from_config(c)); y=m(torch.randn(2,1,224,224), return_boundary=True); batch={'mask': torch.randint(0,4,(2,224,224)), 'boundary': torch.rand(2,1,224,224), 'sdf': torch.randn(2,3,224,224)}; loss, scalars=loss_fn(y,batch); print(c['experiment']['name'], c['loss']['distance_weight'], m.boundary_gate.scale, tuple(y['seg_logits'].shape), sorted(scalars))"`
- `PYTHONPATH=src python3 -m pytest tests/unit/test_builders.py tests/unit/test_losses.py -q`

Result:
- ISIC2018 BDS-Lite damped boundary-gate strong-surface best checkpoint was
  selected at epoch 38 by validation DSC.
- Validation metrics over 519 slices:
  - DSC: 0.8652
  - IoU: 0.7908
  - HD95: 18.2089
  - ASSD: 6.6049
  - Boundary F1: 0.5651
- Compared with ISIC2018 U-Net:
  - DSC: +0.0019
  - IoU: +0.0042
  - HD95: -2.8025
  - ASSD: -0.5319
  - Boundary F1: +0.0308
- Compared with the ISIC distance-error variant, removing distance-error improves
  every mean metric.
- ACDC follow-up config loads and builds with `distance_weight: 0.0`, damped
  boundary gate scale 0.25, and expected loss scalar keys.
- Focused tests passed: `7 passed`.

Problems:
- None for ISIC2018. This is the first clean cross-dataset positive result for
  the paper claim.

Next:
- Run the same surface-only damped-gate setting on ACDC to check whether the
  ACDC miss was caused by the distance-error term:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/acdc_bdslite_boundary_gate_damped_surface.yaml --device cuda`

## 2026-04-26 - acdc-damped-surface-eval

Goal:
- Evaluate the completed ACDC damped boundary-gate strong-surface ablation
  without the SDF distance-error term.

Changed:
- `results/checkpoints/acdc_bdslite_boundary_gate_damped_surface/eval_val.csv`:
  evaluated the best checkpoint on ACDC validation.
- `results/tables/acdc_bdslite_v1.csv`: added the damped boundary-gate
  strong-surface row.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/acdc_bdslite_boundary_gate_damped_surface/best.pt --split val`

Result:
- ACDC BDS-Lite damped boundary-gate strong-surface best checkpoint was selected
  at epoch 47 by validation DSC.
- Validation metrics over 186 slices:
  - DSC: 0.7878
  - IoU: 0.7141
  - HD95: 8.1615
  - ASSD: 2.8970
  - Boundary F1: 0.8475
- Compared with ACDC U-Net:
  - DSC: -0.0041
  - IoU: -0.0053
  - HD95: -0.2309
  - ASSD: +0.1977
  - Boundary F1: -0.0079
- Compared with the ACDC distance-error variant, removing distance-error
  improves DSC, IoU, HD95, and Boundary F1, but worsens ASSD.

Problems:
- This is still a mixed ACDC result: mean HD95 improves, but the result does not
  satisfy the full boundary-quality claim.

Next:
- Sweep saved ACDC surface-only checkpoints to check whether DSC-based
  checkpoint selection is hiding a better boundary-metric checkpoint.

Follow-up:
- Swept saved checkpoints for `acdc_bdslite_boundary_gate_damped_surface`.
- Mean validation metrics:
  - `best.pt`: DSC 0.7878, IoU 0.7141, HD95 8.1615, ASSD 2.8970,
    Boundary F1 0.8475
  - epoch 25: DSC 0.7713, IoU 0.6973, HD95 10.8460, ASSD 3.5544,
    Boundary F1 0.8286
  - epoch 50: DSC 0.7844, IoU 0.7107, HD95 9.9302, ASSD 3.2218,
    Boundary F1 0.8475
  - epoch 75: DSC 0.7708, IoU 0.6957, HD95 11.2071, ASSD 3.4987,
    Boundary F1 0.8322
  - epoch 100: DSC 0.7627, IoU 0.6878, HD95 11.3315, ASSD 3.5048,
    Boundary F1 0.8238
  - epoch 125: DSC 0.7631, IoU 0.6880, HD95 11.5153, ASSD 3.6447,
    Boundary F1 0.8213
  - epoch 150: DSC 0.7623, IoU 0.6871, HD95 11.9420, ASSD 3.7431,
    Boundary F1 0.8206
- The DSC-selected `best.pt` is also the best practical checkpoint for this
  ablation. Epoch 50 nearly ties Boundary F1 but is worse on DSC, HD95, and ASSD.

Next:
- Profile the positive ISIC2018 surface-only damped-gate model against the
  ISIC2018 U-Net baseline to quantify inference-cost overhead.

Follow-up:
- Profiled ISIC2018 U-Net and the positive ISIC2018 surface-only damped-gate
  BDS-Lite config on CPU with batch size 1, warmup 1, iterations 3.
- Wrote:
  - `results/tables/resource_profile_isic2018_unet_baseline.csv`
  - `results/tables/resource_profile_isic2018_bdslite_boundary_gate_damped_surface.csv`
  - `results/tables/resource_profile_isic2018_comparison.csv`
- CPU profile summary:
  - U-Net: 1,927,042 params, 15.970G FLOPs, 0.615 FPS, 1625.3 ms latency.
  - BDS-Lite serialized/training params: 2,198,003.
  - BDS-Lite deploy params excluding removable boundary decoder: 1,928,114.
  - BDS-Lite FLOPs with boundary branch disabled: 16.079G.
  - Active inference overhead: +1,072 deploy params (+0.06%) and +0.109G FLOPs
    (+0.68%) versus U-Net.

Follow-up:
- CUDA profiles were run outside the sandbox for the same two ISIC2018 configs
  with warmup 10 and iterations 50.
- Updated:
  - `results/tables/resource_profile_isic2018_unet_baseline.csv`
  - `results/tables/resource_profile_isic2018_bdslite_boundary_gate_damped_surface.csv`
  - `results/tables/resource_profile_isic2018_comparison.csv`
- CUDA profile summary:
  - U-Net: 560.92 FPS, 1.7828 ms latency, 65.45 MB peak memory.
  - BDS-Lite: 570.02 FPS, 1.7543 ms latency, 66.50 MB peak memory.
  - BDS-Lite active inference overhead remains +1,072 deploy params (+0.06%)
    and +0.109G FLOPs (+0.68%) versus U-Net.
  - Peak GPU memory increased by about 1.05 MB.

Problems:
- Timing difference is small enough to treat as effectively tied until repeated
  profiling is run.

Next:
- Consolidate the current Synapse, ACDC, and ISIC2018 ablation outcomes into a
  compact cross-dataset status table, then decide whether to continue tuning
  ACDC/Synapse or start formal Month 4 ablation reporting.

Follow-up:
- Added `results/tables/cross_dataset_bdslite_status.csv`, comparing the same
  damped boundary-gate strong-surface setting against U-Net on Synapse, ACDC,
  and ISIC2018.

Result:
- Synapse: mixed. DSC, IoU, and Boundary F1 improve, but HD95 and ASSD worsen.
- ACDC: mixed. HD95 improves, but DSC, ASSD, and Boundary F1 worsen.
- ISIC2018: positive. All tracked metrics improve, and CUDA profiling shows
  negligible active inference overhead.

Next:
- Treat the current method as promising but not yet publishable as a
  cross-dataset boundary-distillation claim. The next concrete method task is to
  add dataset-specific class/loss weighting for ACDC/Synapse hard classes rather
  than further global surface-weight tuning.

## 2026-04-26 - foreground-class-weight-loss

Goal:
- Add targeted foreground class weighting so ACDC/Synapse hard classes can be
  emphasized without changing global surface-loss strength for every class.

Changed:
- `src/bds_lite/training/losses.py`: added optional
  `LossWeights.foreground_class_weights` and config parsing from
  `loss.foreground_class_weights`.
- Weighted foreground Dice, signed surface loss, and SDF distance-error loss now
  support one weight per foreground class.
- `configs/base.yaml`: documented the default
  `loss.foreground_class_weights: null`.
- `tests/unit/test_losses.py`: added coverage for parsing, weighted composite
  loss execution, and invalid weight length.
- `configs/experiments/acdc_bdslite_boundary_gate_damped_surface_class1.yaml`:
  added an ACDC follow-up that weights foreground class 1 at 2.0 while keeping
  classes 2 and 3 at 1.0.

Commands:
- `PYTHONPATH=src python3 -m pytest tests/unit/test_losses.py -q`
- `.venv/bin/ruff check src/bds_lite/training/losses.py tests/unit/test_losses.py`
- `PYTHONPATH=src python3 -c "import torch; from bds_lite.utils import load_config; from bds_lite.training import build_model, BDSLiteLoss; from bds_lite.training.losses import weights_from_config; c=load_config('configs/experiments/acdc_bdslite_boundary_gate_damped_surface_class1.yaml'); m=build_model(c); w=weights_from_config(c); loss_fn=BDSLiteLoss(w); y=m(torch.randn(2,1,224,224), return_boundary=True); batch={'mask': torch.randint(0,4,(2,224,224)), 'boundary': torch.rand(2,1,224,224), 'sdf': torch.randn(2,3,224,224)}; loss, scalars=loss_fn(y,batch); print(c['experiment']['name'], w.foreground_class_weights, tuple(y['seg_logits'].shape), sorted(scalars))"`
- `PYTHONPATH=src python3 -m pytest tests/unit/test_losses.py tests/unit/test_builders.py -q`
- `.venv/bin/ruff format --check src/bds_lite/training/losses.py tests/unit/test_losses.py`
- `PYTHONPATH=src python3 -m pytest tests/unit -q`

Result:
- Weighted loss path works and validates expected class-weight length.
- ACDC class-weighted config builds and emits expected scalar keys.
- Full unit tests passed: `19 passed`.
- Ruff check passed and touched Python files are formatted.

Problems:
- None in implementation. The research risk is that class-1 emphasis may improve
  ACDC label 1 while hurting labels 2/3; this needs an actual training run.

Next:
- Run the ACDC class-1 weighted damped-gate surface ablation:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/acdc_bdslite_boundary_gate_damped_surface_class1.yaml --device cuda`

## 2026-04-26 - acdc-class1-weighted-eval

Goal:
- Evaluate the completed ACDC class-1 weighted damped boundary-gate
  strong-surface ablation.

Changed:
- `results/checkpoints/acdc_bdslite_boundary_gate_damped_surface_class1/eval_val.csv`:
  evaluated the best checkpoint on ACDC validation.
- `results/tables/acdc_bdslite_v1.csv`: added the class-1 weighted ACDC row.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/acdc_bdslite_boundary_gate_damped_surface_class1/best.pt --split val`

Result:
- ACDC BDS-Lite class-1 weighted damped boundary-gate strong-surface best
  checkpoint was selected at epoch 26 by validation DSC.
- Validation metrics over 186 slices:
  - DSC: 0.7908
  - IoU: 0.7152
  - HD95: 8.0854
  - ASSD: 2.3811
  - Boundary F1: 0.8494
- Compared with ACDC U-Net:
  - DSC: -0.0010
  - IoU: -0.0041
  - HD95: -0.3070
  - ASSD: -0.3182
  - Boundary F1: -0.0061
- Compared with unweighted ACDC surface-only BDS-Lite:
  - DSC: +0.0031
  - IoU: +0.0011
  - HD95: -0.0761
  - ASSD: -0.5159
  - Boundary F1: +0.0018

Problems:
- This is the best ACDC BDS-Lite result so far and now improves both HD95 and
  ASSD over U-Net, but it still does not beat U-Net on DSC or Boundary F1.

Next:
- Sweep saved checkpoints for the class-1 weighted ACDC run to check whether a
  later checkpoint improves Boundary F1 or overlap without losing distance
  metrics.

Follow-up:
- Swept saved checkpoints for `acdc_bdslite_boundary_gate_damped_surface_class1`.
- Mean validation metrics:
  - `best.pt`: DSC 0.7908, IoU 0.7152, HD95 8.0854, ASSD 2.3811,
    Boundary F1 0.8494
  - epoch 25: DSC 0.7766, IoU 0.7031, HD95 7.9475, ASSD 2.7405,
    Boundary F1 0.8369
  - epoch 50: DSC 0.7846, IoU 0.7074, HD95 11.4483, ASSD 3.0956,
    Boundary F1 0.8433
  - epoch 75: DSC 0.7640, IoU 0.6867, HD95 12.7291, ASSD 3.3878,
    Boundary F1 0.8163
  - epoch 100: DSC 0.7614, IoU 0.6847, HD95 12.4007, ASSD 3.3973,
    Boundary F1 0.8169
  - epoch 125: DSC 0.7560, IoU 0.6789, HD95 13.0294, ASSD 3.4427,
    Boundary F1 0.8089
  - epoch 150: DSC 0.7536, IoU 0.6761, HD95 13.7459, ASSD 3.5877,
    Boundary F1 0.8058
- `best.pt` remains the best overall checkpoint. Epoch 25 improves HD95 slightly
  but loses too much DSC, ASSD, and Boundary F1.
- Updated `results/tables/cross_dataset_bdslite_status.csv` to use the
  class-1 weighted ACDC result as the current best ACDC BDS-Lite setting.
- Added `configs/experiments/acdc_bdslite_boundary_gate_damped_surface_class1_mild.yaml`
  with foreground class weights `[1.5, 1.0, 1.0]`.

Next:
- Run the milder ACDC class-1 weighted ablation to test whether it preserves the
  HD95/ASSD gains while recovering DSC and Boundary F1:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/acdc_bdslite_boundary_gate_damped_surface_class1_mild.yaml --device cuda`

## 2026-04-27 - acdc-class1-mild-eval

Goal:
- Evaluate the completed milder ACDC class-1 weighted damped boundary-gate
  strong-surface ablation.

Changed:
- `results/checkpoints/acdc_bdslite_boundary_gate_damped_surface_class1_mild/eval_val.csv`:
  evaluated the best checkpoint on ACDC validation.
- `results/tables/acdc_bdslite_v1.csv`: added the class-1 mild row.
- `results/tables/cross_dataset_bdslite_status.csv`: updated the ACDC row to
  the mild class-1 result because it is now the strongest ACDC overlap result.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/acdc_bdslite_boundary_gate_damped_surface_class1_mild/best.pt --split val`

Result:
- ACDC BDS-Lite mild class-1 weighted damped boundary-gate strong-surface best
  checkpoint was selected at epoch 24 by validation DSC.
- Validation metrics over 186 slices:
  - DSC: 0.7966
  - IoU: 0.7220
  - HD95: 9.1083
  - ASSD: 2.4945
  - Boundary F1: 0.8506
- Compared with ACDC U-Net:
  - DSC: +0.0047
  - IoU: +0.0026
  - HD95: +0.7159
  - ASSD: -0.2047
  - Boundary F1: -0.0049
- Compared with stronger class-1 weighting:
  - DSC: +0.0057
  - IoU: +0.0067
  - HD95: +1.0229
  - ASSD: +0.1134
  - Boundary F1: +0.0012

Problems:
- Mild class-1 weighting recovers DSC/IoU and slightly improves Boundary F1, but
  loses the HD95 gain from stronger class-1 weighting.

Next:
- Sweep saved checkpoints for the mild class-1 run. If no checkpoint keeps the
  DSC gain while recovering HD95, try class weights `[1.5, 1.0, 1.25]` to support
  label 3 boundary distance without returning to the harsh class-1-only setting.

Follow-up:
- Swept saved checkpoints for
  `acdc_bdslite_boundary_gate_damped_surface_class1_mild`.
- Mean validation metrics:
  - `best.pt`: DSC 0.7966, IoU 0.7220, HD95 9.1083, ASSD 2.4945,
    Boundary F1 0.8506
  - epoch 25: DSC 0.7845, IoU 0.7105, HD95 8.1948, ASSD 2.5843,
    Boundary F1 0.8432
  - epoch 50: DSC 0.7786, IoU 0.7028, HD95 10.8884, ASSD 2.9800,
    Boundary F1 0.8380
  - epoch 75: DSC 0.7659, IoU 0.6892, HD95 11.2951, ASSD 3.1307,
    Boundary F1 0.8245
  - epoch 100: DSC 0.7654, IoU 0.6893, HD95 10.9449, ASSD 3.0376,
    Boundary F1 0.8171
  - epoch 125: DSC 0.7641, IoU 0.6878, HD95 10.7785, ASSD 3.0720,
    Boundary F1 0.8179
  - epoch 150: DSC 0.7606, IoU 0.6842, HD95 11.3332, ASSD 3.0477,
    Boundary F1 0.8131
- `best.pt` remains the right checkpoint. Epoch 25 improves HD95 but loses DSC,
  ASSD, and Boundary F1.
- Added `configs/experiments/acdc_bdslite_boundary_gate_damped_surface_class13_mild.yaml`
  with foreground weights `[1.5, 1.0, 1.25]`.

Next:
- Run the class-1/class-3 mild weighted ACDC ablation:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/acdc_bdslite_boundary_gate_damped_surface_class13_mild.yaml --device cuda`

## 2026-04-27 - acdc-class13-mild-eval

Goal:
- Evaluate the completed ACDC class-1/class-3 mild weighted damped boundary-gate
  strong-surface ablation.

Changed:
- `results/checkpoints/acdc_bdslite_boundary_gate_damped_surface_class13_mild/eval_val.csv`:
  evaluated the best checkpoint on ACDC validation.
- `results/tables/acdc_bdslite_v1.csv`: added the class-1/class-3 mild row.
- `results/tables/cross_dataset_bdslite_status.csv`: updated the ACDC row to
  the class-1/class-3 mild result, now the strongest ACDC setting.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/acdc_bdslite_boundary_gate_damped_surface_class13_mild/best.pt --split val`

Result:
- ACDC BDS-Lite class-1/class-3 mild weighted damped boundary-gate
  strong-surface best checkpoint was selected at epoch 25 by validation DSC.
- Validation metrics over 186 slices:
  - DSC: 0.7953
  - IoU: 0.7230
  - HD95: 6.7488
  - ASSD: 2.3791
  - Boundary F1: 0.8553
- Compared with ACDC U-Net:
  - DSC: +0.0034
  - IoU: +0.0037
  - HD95: -1.6436
  - ASSD: -0.3202
  - Boundary F1: -0.0002
- Compared with mild class-1 weighting:
  - DSC: -0.0012
  - IoU: +0.0011
  - HD95: -2.3595
  - ASSD: -0.1154
  - Boundary F1: +0.0047

Problems:
- Boundary F1 is still microscopically below U-Net, but the difference is only
  0.0002 and the result is otherwise clearly stronger.

Next:
- Sweep saved checkpoints for this ACDC run. If no checkpoint improves Boundary
  F1 while preserving HD95/ASSD gains, stop ACDC tuning and move back to
  Synapse hard-class weighting.

Follow-up:
- Swept saved checkpoints for
  `acdc_bdslite_boundary_gate_damped_surface_class13_mild`.
- Mean validation metrics:
  - `best.pt` / epoch 25: DSC 0.7953, IoU 0.7230, HD95 6.7488, ASSD 2.3791,
    Boundary F1 0.8553
  - epoch 50: DSC 0.7780, IoU 0.7023, HD95 11.1527, ASSD 3.2294,
    Boundary F1 0.8337
  - epoch 75: DSC 0.7697, IoU 0.6930, HD95 11.5388, ASSD 3.3394,
    Boundary F1 0.8248
  - epoch 100: DSC 0.7729, IoU 0.6962, HD95 11.6764, ASSD 3.4342,
    Boundary F1 0.8271
  - epoch 125: DSC 0.7724, IoU 0.6957, HD95 11.3777, ASSD 3.3281,
    Boundary F1 0.8289
  - epoch 150: DSC 0.7717, IoU 0.6949, HD95 11.3636, ASSD 3.4179,
    Boundary F1 0.8260
- No later checkpoint beats `best.pt`; ACDC tuning can pause.
- Added `configs/experiments/synapse_bdslite_boundary_gate_damped_surface_hardclass.yaml`
  with foreground weights emphasizing Synapse hard labels 5, 6, and 8:
  `[1.0, 1.0, 1.0, 1.0, 1.25, 1.75, 1.0, 1.5]`.

Next:
- Run the Synapse hard-class weighted damped-gate surface ablation:
  `PYTHONPATH=src python3 scripts/train.py --config configs/experiments/synapse_bdslite_boundary_gate_damped_surface_hardclass.yaml --device cuda`

## 2026-04-27 - synapse-hardclass-eval

Goal:
- Evaluate the completed Synapse hard-class weighted damped boundary-gate
  strong-surface ablation.

Changed:
- `results/checkpoints/synapse_bdslite_boundary_gate_damped_surface_hardclass/eval_val.csv`:
  evaluated the best checkpoint on Synapse validation.
- `results/tables/synapse_bdslite_v1.csv`: added the hard-class weighted row.

Commands:
- `PYTHONPATH=src python3 scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_gate_damped_surface_hardclass/best.pt --split val`

Result:
- Synapse BDS-Lite hard-class weighted damped boundary-gate strong-surface best
  checkpoint was selected at epoch 42 by validation DSC.
- Validation metrics over 473 slices:
  - DSC: 0.8456
  - IoU: 0.8259
  - HD95: 11.1070
  - ASSD: 4.3503
  - Boundary F1: 0.8583
- Compared with Synapse U-Net:
  - DSC: +0.0007
  - IoU: +0.0003
  - HD95: +0.6316
  - ASSD: +0.6960
  - Boundary F1: +0.0035
- Compared with unweighted damped-gate strong-surface:
  - DSC: -0.0089
  - IoU: -0.0091
  - HD95: -0.0720
  - ASSD: -0.1382
  - Boundary F1: -0.0116

Problems:
- Hard-class weights move the Synapse distance metrics in the right direction
  versus the unweighted damped-gate run, but not enough to beat U-Net HD95/ASSD.
- The weights improve label 6 distance substantially but hurt label 8 and label
  7 distance, so the current hard-class weighting is not well balanced.

Next:
- Sweep saved checkpoints for the Synapse hard-class run to check whether a
  non-DSC checkpoint gives a better HD95/ASSD tradeoff.

Follow-up:
- Swept saved checkpoints for
  `synapse_bdslite_boundary_gate_damped_surface_hardclass`.
- Mean validation metrics:
  - `best.pt`: DSC 0.8456, IoU 0.8259, HD95 11.1070, ASSD 4.3503,
    Boundary F1 0.8583
  - epoch 25: DSC 0.8103, IoU 0.7909, HD95 12.5390, ASSD 4.8092,
    Boundary F1 0.8228
  - epoch 50: DSC 0.8295, IoU 0.8096, HD95 11.4070, ASSD 4.3885,
    Boundary F1 0.8424
  - epoch 75: DSC 0.8286, IoU 0.8084, HD95 12.2032, ASSD 4.8762,
    Boundary F1 0.8451
  - epoch 100: DSC 0.8092, IoU 0.7879, HD95 13.1935, ASSD 5.1772,
    Boundary F1 0.8252
  - epoch 125: DSC 0.8110, IoU 0.7898, HD95 13.3780, ASSD 5.2330,
    Boundary F1 0.8265
  - epoch 150: DSC 0.8114, IoU 0.7902, HD95 13.5221, ASSD 5.2503,
    Boundary F1 0.8266
- No saved checkpoint beats `best.pt`; the hard-class run remains only a partial
  distance diagnostic and does not fix Synapse.

Next:
- Stop Synapse hard-class tuning for now. The immediate useful task is to create
  the Month 4 ablation report/table from the current best results and mark
  Synapse as the remaining failure case rather than continuing ad hoc tuning.

Follow-up:
- Added `results/tables/month4_ablation_summary.csv`.

Result:
- Current Month 4 status:
  - Synapse: mixed. Best BDS-Lite variant improves DSC, IoU, and Boundary F1,
    but worsens HD95/ASSD.
  - ACDC: mostly positive. Best BDS-Lite variant improves DSC, IoU, HD95, and
    ASSD; Boundary F1 is effectively tied but lower by 0.0002.
  - ISIC2018: positive. Best BDS-Lite variant improves all tracked metrics.
- The inference-cost claim is supported by the ISIC CUDA profile: active
  inference overhead is about +0.68% FLOPs and +0.06% deployable parameters.

Next:
- Move from exploratory tuning to formal reporting: add a short experiment note
  in `docs/experiment_plan.md` or a new `docs/month4_ablation_report.md`
  summarizing accepted variants, failure cases, and next ablations needed for
  publication.

## 2026-04-27 - synapse-label68-diagnostics

Goal:
- Create formal Synapse per-class diagnostic visualizations for the hard labels
  6 and 8 before any further tuning.

Changed:
- `scripts/visualize_synapse_label_diagnostics.py`: added a repeatable
  checkpoint-comparison diagnostic script. It compares a baseline checkpoint and
  candidate checkpoint on a dataset split, computes per-case metrics for selected
  labels, ranks the worst candidate-vs-baseline distance failures, and exports
  overlay PNGs.
- `results/tables/synapse_label68_diagnostics.csv`: added per-case Synapse
  validation diagnostics for labels 6 and 8 using U-Net as baseline and the
  hard-class weighted damped-gate BDS-Lite checkpoint as candidate.
- `results/figures/synapse_label68_diagnostics/`: added the top six HD95-worse
  overlays for label 6 and label 8.

Commands:
- `PYTHONPATH=src python3 scripts/visualize_synapse_label_diagnostics.py --device cpu --max-samples 2 --top-k 1 --output-dir /tmp/synapse_diag_smoke --table /tmp/synapse_diag_smoke.csv`
- `.venv/bin/ruff check scripts/visualize_synapse_label_diagnostics.py`
- `.venv/bin/ruff format --check scripts/visualize_synapse_label_diagnostics.py`
- `PYTHONPATH=src python3 scripts/visualize_synapse_label_diagnostics.py --device cpu --top-k 6`

Result:
- Smoke diagnostic passed.
- Ruff check and format check passed.
- Full validation diagnostic wrote 946 per-case rows and 12 overlays.
- Worst label-6 HD95 regressions include `case0006_slice081`
  (`dHD95=+37.214`, `dASSD=+34.425`) and `case0031_slice050`
  (`dHD95=+32.403`, `dDSC=-0.203`).
- Worst label-8 HD95 regressions include `case0023_slice071`
  (`dHD95=+49.162`) and `case0030_slice118` (`dHD95=+26.608`).

Problems:
- The diagnostics confirm that some HD95 regressions occur even when DSC or
  Boundary F1 improves, so overlap/local-boundary scores are masking distant
  false positives or missed disconnected structures.

Next:
- Add inference-cost ablation for BDS-Lite with boundary branch disabled versus
  retained, then start paper-facing Month 4 table/figure preparation.

## 2026-04-27 - boundary-branch-cost-ablation

Goal:
- Add the formal inference-cost ablation comparing BDS-Lite with the removable
  boundary branch disabled versus retained.

Changed:
- `scripts/profile_boundary_branch_ablation.py`: added a reproducible profiler
  for U-Net baseline, BDS-Lite deploy path, and BDS-Lite retained-boundary path.
- `results/tables/resource_profile_isic2018_boundary_branch_ablation.csv`: added
  CUDA resource profiles for the accepted ISIC2018 BDS-Lite setting.
- `docs/month4_ablation_report.md`: updated the efficiency section with the
  disabled-vs-retained boundary-branch result.

Commands:
- `PYTHONPATH=src python3 scripts/profile_boundary_branch_ablation.py --device cpu --warmup 0 --iterations 1 --output /tmp/boundary_branch_ablation_smoke.csv`
- `.venv/bin/ruff check scripts/profile_boundary_branch_ablation.py`
- `.venv/bin/ruff format --check scripts/profile_boundary_branch_ablation.py`
- `PYTHONPATH=src python3 -m py_compile scripts/profile_boundary_branch_ablation.py`
- `PYTHONPATH=src python3 scripts/profile_boundary_branch_ablation.py --device cpu`
- `PYTHONPATH=src python3 scripts/profile_boundary_branch_ablation.py --device cuda --warmup 10 --iterations 50`

Result:
- CUDA ablation, batch size 1, input `1x3x224x224`:
  - U-Net baseline: 1.927M active params, 15.970G FLOPs, 561.04 FPS,
    1.782 ms latency, 65.45 MB peak memory.
  - BDS-Lite boundary disabled: 1.928M active params, 16.079G FLOPs,
    564.56 FPS, 1.771 ms latency, 73.88 MB peak memory.
  - BDS-Lite boundary retained: 2.198M active params, 19.677G FLOPs,
    404.72 FPS, 2.471 ms latency, 82.31 MB peak memory.
- Disabled deploy path overhead versus U-Net: +0.06% active params and +0.68%
  FLOPs.
- Retained boundary branch overhead versus U-Net: +14.06% active params and
  +23.22% FLOPs.
- Retained boundary branch latency is +39.49% versus the disabled BDS-Lite
  deploy path.

Problems:
- CPU timing was noisy and should not be used for paper claims; CUDA timing is
  the relevant table.

Next:
- Start paper-facing Month 4 table and figure preparation.

## 2026-04-27 - month4-paper-artifacts

Goal:
- Start formal paper-facing Month 4 table and figure preparation from the
  consolidated result CSVs.

Changed:
- `scripts/prepare_month4_paper_artifacts.py`: added a small artifact builder
  that formats raw experiment CSVs into paper-facing tables and an initial SVG
  cost figure.
- `paper/tables/month4_cross_dataset_ablation_paper.csv`: added rounded
  cross-dataset U-Net vs BDS-Lite metrics with directional delta labels.
- `paper/tables/month4_boundary_branch_cost_paper.csv`: added rounded
  inference-cost ablation values for U-Net, BDS-Lite deploy, and retained
  boundary branch.
- `paper/figures/month4_boundary_branch_cost.svg`: added an initial bar figure
  for params, FLOPs, and latency.

Commands:
- `PYTHONPATH=src python3 scripts/prepare_month4_paper_artifacts.py`
- `.venv/bin/ruff check scripts/prepare_month4_paper_artifacts.py`
- `.venv/bin/ruff format --check scripts/prepare_month4_paper_artifacts.py`
- `PYTHONPATH=src python3 -m py_compile scripts/prepare_month4_paper_artifacts.py`

Result:
- Paper table artifacts now exist under `paper/tables/`.
- Initial cost figure exists under `paper/figures/`.

Problems:
- These are first-pass paper artifacts only. They still need final captioning,
  advisor review, and likely conversion into the journal's preferred table
  format.

Next:
- Draft paper-facing captions/notes for the Month 4 tables and the Synapse
  failure-case diagnostic figure set.

## 2026-04-27 - autonomous-workflow-and-month4-notes

Goal:
- Configure the repository for autonomous continuation and complete the pending
  Month 4 caption/notes task.

Changed:
- `AGENTS.md`: added repo-local autonomous workflow, environment, checks, plan
  tracking, and research-scope instructions.
- `plan.md`: added an execution progress section and moved the Month 4 caption
  task to completed.
- `paper/sections/month4_tables_and_figures_notes.md`: added draft captions,
  interpretation boundaries, and Synapse label 6/8 diagnostic notes.

Commands:
- `.venv/bin/ruff format --check .`
- `.venv/bin/ruff check .`
- `.venv/bin/python -m pytest`
- `.venv/bin/python -c "<CSV diagnostic summary>"`

Result:
- Autonomous workflow documentation exists for future sessions.
- Month 4 paper-facing table, cost figure, and diagnostic captions are drafted.
- Synapse is explicitly documented as a mixed/failure case for HD95 and ASSD,
  while ISIC2018 and ACDC remain the positive Month 4 evidence.

Problems:
- No new model training was performed in this session.

Next:
- Add a final cross-dataset aggregation/report generator beyond the Month 4
  snapshot, then rerun checks and update `plan.md`.

Follow-up:
- Added `scripts/prepare_cross_dataset_report.py` to build a reproducible
  dataset-level claim report from `results/tables/cross_dataset_bdslite_status.csv`
  and the boundary-branch cost ablation.
- Added `tests/unit/test_cross_dataset_report.py`.
- Generated `results/tables/cross_dataset_report_v1.csv` and
  `docs/cross_dataset_report.md`.
- The report marks ACDC and ISIC2018 as supporting the claim, Synapse as partial
  support, and the full cross-dataset boundary-distance claim as not ready.

Commands:
- `PYTHONPATH=src .venv/bin/python scripts/prepare_cross_dataset_report.py`
- `.venv/bin/python -m pytest tests/unit/test_cross_dataset_report.py`

Result:
- Cross-dataset aggregation/report generation is now scripted and tested.

Next:
- Implement or integrate an additional lightweight backbone such as MobileViT,
  EfficientFormer, or TinyViT.

Follow-up:
- Added a MobileViT-style lightweight U-Net baseline.
- `src/bds_lite/models/mobilevit_unet.py`: implemented a depthwise-separable
  encoder with a small bottleneck transformer block and reused the existing
  U-Net decoder/output contract.
- `src/bds_lite/training/builders.py`: added `model.name: mobilevit_unet`.
- `configs/experiments/isic_mobilevit_unet.yaml`: added an ISIC2018 baseline
  config.
- `tests/unit/test_builders.py`: added a build/forward shape test.
- `README.md` and `plan.md`: updated implementation status.

Commands:
- `.venv/bin/python -m pytest tests/unit/test_builders.py`
- `.venv/bin/python -m py_compile src/bds_lite/models/mobilevit_unet.py`
- `PYTHONPATH=src .venv/bin/python -c "<load config and build MobileViTUNet>"`
- `.venv/bin/ruff format --check .`
- `.venv/bin/ruff check .`
- `.venv/bin/python -m pytest`

Result:
- MobileViT-style baseline builds from config and returns the expected
  segmentation output shape.
- Full checks passed: Ruff format check, Ruff lint check, and 23 Pytest tests.

Next:
- Assess the final ablation item and add the most useful reproducible reporting
  artifact that can be completed without launching long training runs.

Follow-up:
- Ran a Synapse largest-component post-processing ablation for U-Net and
  BDS-Lite on the validation split.
- Added `scripts/prepare_final_ablation_report.py`.
- Added `tests/unit/test_final_ablation_report.py`.
- Generated `results/tables/final_ablation_status_v1.csv` and
  `docs/final_ablation_report.md`.
- Generated `results/checkpoints/synapse_unet_baseline/eval_val_lcc.csv` and
  `results/checkpoints/synapse_bdslite_boundary_gate_damped_surface/eval_val_lcc.csv`.

Commands:
- `PYTHONPATH=src .venv/bin/python scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_gate_damped_surface/best.pt --split val --keep-largest-component --output results/checkpoints/synapse_bdslite_boundary_gate_damped_surface/eval_val_lcc.csv`
- `PYTHONPATH=src .venv/bin/python scripts/evaluate.py --checkpoint results/checkpoints/synapse_unet_baseline/best.pt --split val --keep-largest-component --output results/checkpoints/synapse_unet_baseline/eval_val_lcc.csv`
- `PYTHONPATH=src .venv/bin/python scripts/prepare_final_ablation_report.py`
- `.venv/bin/python -m pytest tests/unit/test_final_ablation_report.py`

Result:
- Largest-component post-processing did not fix the Synapse distance failure.
  With post-processing, BDS-Lite remains worse than U-Net by +0.477 HD95 and
  +0.653 ASSD.
- Final ablation status is now scripted and records module, loss, inference-cost,
  targeted post-processing, and repeated-seed status.

Problems:
- No CUDA device is available in this session, so new training runs and repeated
  profiling were not attempted.

Next:
- Run full repository checks and leave the plan in a complete state for the
  current automation snapshot.

## 2026-04-27 - synapse-far-fp-loss-ablation

Goal:
- Test a narrower SDF loss for Synapse that penalizes far false-positive
  foreground probability without globally increasing missed-foreground penalty.

Changed:
- `src/bds_lite/training/losses.py`: added `SdfFarFalsePositiveLoss` and
  `far_fp_weight`.
- `scripts/train.py`: requires SDF targets when `far_fp_weight` is enabled.
- `configs/experiments/synapse_bdslite_boundary_gate_damped_surface_far_fp.yaml`:
  added the Synapse targeted ablation config.
- `tests/unit/test_losses.py`: added far false-positive loss coverage.
- `results/checkpoints/synapse_bdslite_boundary_gate_damped_surface_far_fp/eval_val.csv`:
  added validation metrics for the completed run.
- `results/tables/synapse_bdslite_v1.csv`,
  `results/tables/final_ablation_status_v1.csv`, and
  `docs/final_ablation_report.md`: recorded the result as a failed targeted
  ablation.

Commands:
- `.venv/bin/python -m pytest tests/unit/test_losses.py`
- `.venv/bin/ruff check src/bds_lite/training/losses.py scripts/train.py tests/unit/test_losses.py`
- `.venv/bin/ruff format src/bds_lite/training/losses.py`
- `.venv/bin/python -c "<CUDA one-batch far_fp smoke test>"`
- `.venv/bin/python scripts/train.py --config configs/experiments/synapse_bdslite_boundary_gate_damped_surface_far_fp.yaml`
- `.venv/bin/python scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_boundary_gate_damped_surface_far_fp/best.pt --split val --output results/checkpoints/synapse_bdslite_boundary_gate_damped_surface_far_fp/eval_val.csv`

Result:
- Training completed with best validation DSC 0.8460.
- Validation mean metrics:
  - DSC 0.84599
  - IoU 0.82648
  - HD95 11.73767
  - ASSD 4.26556
  - Boundary F1 0.85562
- Versus Synapse U-Net, the run is +0.001 DSC, +1.262 HD95, +0.611 ASSD,
  and +0.001 Boundary F1.

Problems:
- The loss did not fix Synapse distance metrics. It improves ASSD versus the
  unweighted damped-gate BDS-Lite run but worsens HD95 and loses most of the
  DSC/Boundary F1 gain.

Next:
- Do not promote the far false-positive variant. Keep Synapse marked as partial
  support and avoid repeated-seed final runs until a variant beats U-Net on
  HD95/ASSD.

## 2026-04-27 - synapse-distance-failure-diagnosis

Goal:
- Diagnose the Synapse HD95/ASSD failure before trying new fixes.

Diagnosis:
- The far false-positive SDF run improves mean DSC and Boundary F1 only
  marginally over U-Net, but worsens mean HD95 by +1.262 and ASSD by +0.611.
  The largest HD95 regression comes from label 8 (+4.276 HD95, +1.367 ASSD),
  followed by label 6 (+2.250 HD95, +1.151 ASSD); label 7 and label 3 also add
  large ASSD regressions. A saved-checkpoint sweep shows the distance metrics
  never satisfy the U-Net threshold: the best saved HD95/ASSD tradeoff is around
  epoch 50 with HD95 12.005 and ASSD 4.226, still above the accept criterion
  of HD95 < 10.476 and ASSD < 3.655. The evaluator computes symmetric
  surface-to-surface distances with per-sample spacing when available and uses
  `nanmean`, so the failure is unlikely to be a simple metric aggregation bug.
  The loss stack can plausibly interfere with small-organ spatial precision:
  boundary BCE/Dice gradients flow through the shared encoder, distillation pulls
  segmentation boundary features toward detached boundary features, and
  surface/far-FP SDF terms optimize soft foreground probability rather than the
  final argmax topology used by HD95/ASSD.

Commands:
- `PYTHONPATH=src .venv/bin/python scripts/evaluate.py --checkpoint <saved far-fp epoch> --split val --output /tmp/<epoch>_far_fp_eval_val.csv`
- `.venv/bin/python -c "<per-class far-fp vs U-Net delta summary>"`
- `.venv/bin/python -c "<far-fp saved-checkpoint sweep summary>"`

Result:
- Labels 8 and 6 are the primary HD95 failure drivers.
- Validation DSC selection did not hide an accepted HD95/ASSD checkpoint.

Next:
- Candidate A: remove all distance-based losses and train
  `configs/experiments/synapse_bdslite_no_surface.yaml`.

## 2026-04-27 - synapse-no-surface-candidate-a

Goal:
- Test Candidate A by removing all distance-based losses from the Synapse
  BDS-Lite setup.

Changed:
- `configs/experiments/synapse_bdslite_no_surface.yaml`: added a Synapse
  boundary-only BDS-Lite config with Dice, CE, and boundary BCE/Dice only.
- `results/checkpoints/synapse_bdslite_no_surface/eval_val.csv`: added the
  validation metrics for the completed run.
- `results/tables/synapse_bdslite_v1.csv`,
  `results/tables/final_ablation_status_v1.csv`, and
  `docs/final_ablation_report.md`: recorded Candidate A as rejected.

Commands:
- `CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --config configs/experiments/synapse_bdslite_no_surface.yaml > /tmp/bds-lite-train.log 2>&1; echo "DONE: exit $?"`
- `.venv/bin/python scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_no_surface/best.pt --split val --output results/checkpoints/synapse_bdslite_no_surface/eval_val.csv > /tmp/bds-lite-eval.log 2>&1; echo "DONE: exit $?"`
- `.venv/bin/ruff format --check .`
- `.venv/bin/ruff check .`
- `.venv/bin/python -m pytest`

Result:
- Training exit code: 0.
- Evaluation exit code: 0.
- Best checkpoint: epoch 34.
- Validation mean metrics:
  - DSC 0.845211
  - IoU 0.825670
  - HD95 12.211819
  - ASSD 4.782197
  - Boundary F1 0.857849
- Candidate A is rejected because it misses the accept criterion
  (HD95 < 10.476 and ASSD < 3.655).
- Full checks passed: Ruff format check, Ruff lint check, and 27 Pytest tests.

Problems:
- Removing SDF/surface/far-FP losses is not sufficient; the distance regression
  persists even with boundary-only auxiliary supervision.

Next:
- Candidate B: detach boundary decoder features from the shared encoder.

## 2026-04-27 - synapse-detached-boundary-candidate-b

Goal:
- Test Candidate B by detaching encoder features before the auxiliary boundary
  decoder so boundary loss gradients do not update the shared encoder.

Changed:
- `src/bds_lite/models/bds_lite.py`: added config-gated detached boundary
  decoder inputs.
- `src/bds_lite/training/builders.py`: wired
  `model.detach_boundary_features`.
- `configs/experiments/synapse_bdslite_detached_boundary.yaml`: added the
  detached-boundary Synapse config.
- `tests/unit/test_builders.py`: added builder coverage for the new flag.
- `results/checkpoints/synapse_bdslite_detached_boundary/eval_val.csv`: added
  validation metrics.
- `results/tables/synapse_bdslite_v1.csv`,
  `results/tables/final_ablation_status_v1.csv`, and
  `docs/final_ablation_report.md`: recorded Candidate B as rejected.

Commands:
- `.venv/bin/python -m pytest tests/unit/test_builders.py`
- `.venv/bin/python -m py_compile src/bds_lite/models/bds_lite.py src/bds_lite/training/builders.py`
- `CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --config configs/experiments/synapse_bdslite_detached_boundary.yaml > /tmp/bds-lite-train.log 2>&1; echo "DONE: exit $?"`
- `.venv/bin/python scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_detached_boundary/best.pt --split val --output results/checkpoints/synapse_bdslite_detached_boundary/eval_val.csv > /tmp/bds-lite-eval.log 2>&1; echo "DONE: exit $?"`
- `.venv/bin/ruff format --check .`
- `.venv/bin/ruff check .`
- `.venv/bin/python -m pytest`

Result:
- Training exit code: 0.
- Evaluation exit code: 0.
- Best checkpoint: epoch 36.
- Validation mean metrics:
  - DSC 0.848629
  - IoU 0.828980
  - HD95 12.500901
  - ASSD 4.642404
  - Boundary F1 0.859893
- Candidate B is rejected because it misses the accept criterion
  (HD95 < 10.476 and ASSD < 3.655).
- Full checks passed: Ruff format check, Ruff lint check, and 28 Pytest tests.

Problems:
- Removing boundary decoder gradients from the shared encoder improves DSC and
  Boundary F1 versus Candidate A, but distance metrics remain substantially
  worse than U-Net.

Next:
- Candidate C: use a weak boundary-only branch without the retained boundary
  gate.

## 2026-04-27 - synapse-weak-boundary-candidate-c

Goal:
- Test Candidate C with weak boundary supervision (`boundary_weight=0.05`) and
  no surface loss, no distance loss, and no retained boundary gate.

Changed:
- `configs/experiments/synapse_bdslite_weak_boundary.yaml`: added the final
  requested Synapse candidate config.
- `results/checkpoints/synapse_bdslite_weak_boundary/eval_val.csv`: added
  validation metrics.
- `results/tables/synapse_bdslite_v1.csv`,
  `results/tables/final_ablation_status_v1.csv`, and
  `docs/final_ablation_report.md`: recorded Candidate C as rejected.
- `plan.md`: recorded all three candidate failures and moved the Synapse blocker
  wording to a deeper deferred architectural/backbone fix.

Commands:
- `CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --config configs/experiments/synapse_bdslite_weak_boundary.yaml > /tmp/bds-lite-train.log 2>&1; echo "DONE: exit $?"`
- `.venv/bin/python scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_weak_boundary/best.pt --split val --output results/checkpoints/synapse_bdslite_weak_boundary/eval_val.csv > /tmp/bds-lite-eval.log 2>&1; echo "DONE: exit $?"`
- `.venv/bin/ruff format --check .`
- `.venv/bin/ruff check .`
- `.venv/bin/python -m pytest`

Result:
- Training exit code: 0.
- Evaluation exit code: 0.
- Best checkpoint: epoch 60.
- Validation mean metrics:
  - DSC 0.846448
  - IoU 0.825985
  - HD95 11.233156
  - ASSD 4.331589
  - Boundary F1 0.859333
- Candidate C is rejected because it misses the accept criterion
  (HD95 < 10.476 and ASSD < 3.655).
- Full checks passed: Ruff format check, Ruff lint check, and 28 Pytest tests.

Problems:
- All three requested shallow fixes failed:
  - Candidate A no-surface: HD95 12.211819, ASSD 4.782197.
  - Candidate B detached boundary: HD95 12.500901, ASSD 4.642404.
  - Candidate C weak boundary: HD95 11.233156, ASSD 4.331589.

Next:
- Synapse HD95/ASSD now requires architectural separation or a dataset-specific
  backbone; shallow loss/gradient fixes are exhausted.

## 2026-04-27 - autonomous-continuation-setup

Goal:
- Configure full autonomous productivity: update AGENTS.md with current research
  state and priority queue, implement Attention U-Net baseline, create repeated-seed
  configs, launch 7-experiment training chain, and add multi-seed aggregation script.

Changed:
- `AGENTS.md`: updated with current research state summary (ISIC positive, ACDC
  mostly positive, Synapse mixed), priority queue (Tier 1 baselines, Tier 2 seeds,
  Tier 3 reporting), and inference-cost reference.
- `src/bds_lite/models/blocks.py`: added `AttentionGate` and `AttentionUpBlock`.
- `src/bds_lite/models/unet.py`: added `AttentionUNetDecoder` and `AttentionUNet`.
- `src/bds_lite/models/__init__.py`: exported `AttentionUNet`.
- `src/bds_lite/training/builders.py`: wired `model.name: attention_unet`.
- `configs/experiments/isic_attention_unet_baseline.yaml`: ISIC2018 Attention U-Net.
- `configs/experiments/acdc_attention_unet_baseline.yaml`: ACDC Attention U-Net.
- `configs/experiments/synapse_attention_unet_baseline.yaml`: Synapse Attention U-Net.
- `configs/experiments/isic_bdslite_boundary_gate_damped_surface_seed2027.yaml`:
  seed 2027 variant of the accepted ISIC2018 BDS-Lite config.
- `configs/experiments/isic_bdslite_boundary_gate_damped_surface_seed2028.yaml`:
  seed 2028 variant.
- `configs/experiments/acdc_bdslite_boundary_gate_damped_surface_class13_mild_seed2027.yaml`:
  seed 2027 variant of the accepted ACDC BDS-Lite config.
- `configs/experiments/acdc_bdslite_boundary_gate_damped_surface_class13_mild_seed2028.yaml`:
  seed 2028 variant.
- `scripts/prepare_seed_averaged_results.py`: aggregates multi-seed eval CSVs
  into mean ± std per metric.
- `tests/unit/test_seed_averaged_results.py`: unit coverage for aggregation script.
- `tests/unit/test_builders.py`: added Attention U-Net builder test.
- `plan.md`: updated completed items and unfinished queue.
- `memory/project_data.md`: updated with full dataset and experiment state.

Commands:
- `.venv/bin/ruff format --check .`
- `.venv/bin/ruff check .`
- `.venv/bin/python -m pytest`
- `bash /tmp/bds-lite-train-chain.sh > /tmp/bds-lite-train-chain.log 2>&1` (background)

Result:
- Full checks passed: Ruff format, Ruff lint, 32 Pytest tests.
- Attention U-Net builds from config and forwards expected output shapes for all
  three datasets.
- 7-experiment training chain launched in background:
  1. `isic2018_mobilevit_unet` (new paper lightweight baseline)
  2. `isic2018_bdslite_unet` (ablation: no gate/surface)
  3. `acdc_bdslite_unet` (ablation: no gate/surface)
  4. `isic2018_bdslite_boundary_gate_damped_surface_seed2027`
  5. `isic2018_bdslite_boundary_gate_damped_surface_seed2028`
  6. `acdc_bdslite_boundary_gate_damped_surface_class13_mild_seed2027`
  7. `acdc_bdslite_boundary_gate_damped_surface_class13_mild_seed2028`

Problems:
- None. Seed aggregation script reports partial results (1/3 seeds) until chain
  completes.

Next:
- When training chain completes: run `scripts/prepare_seed_averaged_results.py`
  and update paper tables.
- Train Attention U-Net baselines on ISIC2018, ACDC, and Synapse.
- Update `results/tables/baseline_results_month1.csv` with MobileViT-UNet and
  Attention U-Net rows.

## 2026-04-27 - init-from-checkpoint-feature

Goal:
- Implement U-Net pre-training initialization for BDS-Lite as the next
  architectural approach for the Synapse HD95/ASSD failure.

Changed:
- `scripts/train.py`: added `_maybe_init_from_checkpoint` helper that copies
  U-Net encoder and decoder weights into BDS-Lite (encoder → encoder,
  decoder.* → seg_decoder.*) using name/shape matching with `strict=False`.
- `configs/base.yaml`: added `trainer.init_from_checkpoint: null` default.
- `configs/experiments/synapse_bdslite_init_from_unet.yaml`: new Synapse
  BDS-Lite config that initializes encoder+decoder from the trained U-Net
  checkpoint and trains with half the learning rate.
- `scripts/update_baseline_table.py`: new script to add/update baseline rows
  in `results/tables/baseline_results_month1.csv` for all tracked baselines.
- `/tmp/bds-lite-train-chain2.sh`: second training chain prepared for
  Attention U-Net baselines + Synapse init-from-UNet experiment.

Commands:
- `.venv/bin/ruff check scripts/train.py`
- `.venv/bin/ruff format --check scripts/train.py`
- `.venv/bin/python -m pytest`
- Smoke test: `_maybe_init_from_checkpoint` loaded=92 skipped=48

Result:
- init_from_checkpoint feature verified: 92 U-Net weights copied into BDS-Lite,
  48 skipped (boundary-branch-specific parameters).
- Full checks: Ruff format, Ruff lint, 32 Pytest tests passed.

Problems:
- None.

Next:
- Wait for chain 1 to complete; then launch chain 2 (Attention U-Net + Synapse
  init-from-UNet).
- After chain 2: run `scripts/update_baseline_table.py` and
  `scripts/prepare_seed_averaged_results.py` to update all result tables.


## 2026-04-27 - mobilevit-resource-profile

Goal: Profile MobileViT-UNet resource usage; update paper tables and plan.

Changed files:
- `results/tables/resource_profile_isic2018_mobilevit_unet.csv` (new)
- `results/tables/baseline_results_month1.csv` (added MobileViT row)
- `paper/tables/paper_main_table.md` (updated with MobileViT results)
- `paper/tables/paper_main_table.csv` (updated)
- `paper/sections/04_experiments.md` (updated Section 4.6.2 with MobileViT in cost table)
- `plan.md` (updated current status and unfinished sections)

Commands:
```bash
PYTHONPATH=src .venv/bin/python scripts/profile_model.py \
  --config configs/experiments/isic_mobilevit_unet.yaml \
  --device cuda --warmup 10 --iterations 30
PYTHONPATH=src .venv/bin/python scripts/update_baseline_table.py
PYTHONPATH=src .venv/bin/python scripts/prepare_paper_main_table.py
```

Result:
- MobileViT-UNet ISIC2018: 0.993M params, 6.98G FLOPs, 455 FPS, 57.1 MB.
  Significantly more efficient than U-Net (1.927M, 15.97G, 561 FPS, 65.5 MB).
  Best ISIC2018 metrics so far: DSC=0.8797, HD95=14.90 (MobileViT > BDS-Lite > U-Net).
  MobileViT is slower despite fewer FLOPs due to memory-bandwidth-bound depthwise convs.

Problems: None.

Next:
- Wait for chain1 to complete (~100 min remaining at start of this session).
- Launch chain2 after chain1 (Attention U-Net ×3 + synapse_bdslite_init_from_unet).
- Run process_training_results.py after chain2 to regenerate all tables.


## 2026-04-27 - paper-corrections-and-acdc-bdslite-unet

Goal: Fix paper inaccuracies and record acdc_bdslite_unet results.

Changed files:
- `paper/sections/06_conclusion.md`: Fixed Synapse attribution (was "compact organs
  (aorta, gallbladder)" → "Liver (+6.02mm) and Left Kidney (+1.81mm)").
- `paper/sections/07_figures_todo.md`: Corrected Figure 5 caption and per-class
  delta table (Liver and Left Kidney as primary regressions, Aorta improved).
- `paper/sections/04_experiments.md`: Added ACDC ablation table (Section 4.6.1b);
  fixed Section 4.8 Synapse label attribution.
- `paper/tables/paper_main_table.md` and `.csv`: Regenerated with acdc_bdslite_unet.
- `plan.md`: Updated current status with ACDC bdslite_unet results.

Commands:
```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_paper_main_table.py
.venv/bin/ruff format --check . && .venv/bin/ruff check .
.venv/bin/python -m pytest
```

Result:
- acdc_bdslite_unet (boundary+distill only, λ_b=0.4):
  DSC=0.7939, IoU=0.7201, HD95=9.07, ASSD=2.34, BF1=0.8450
- Pattern: boundary+distill alone WORSENS HD95 on ACDC (+0.68mm vs U-Net 8.39).
  Gate+surface recovers and achieves −1.64mm total. Gate/surface are essential.
- Chain1 seed experiments now running (ISIC/ACDC seed2027/2028). ETA ~15:40.

Problems: None.

Next:
- Wait for chain1 seed experiments to complete (~15:40 ETA).
- Launch chain2 (Attention U-Net ×3 + synapse_bdslite_init_from_unet).
- After chain2: run scripts/process_training_results.py.


## 2026-04-27 - autonomous-refresh-seeds-and-far-fp-rerun

Goal:
- Refresh autonomous workflow instructions, aggregate completed seed runs, and
  rerun the requested Synapse far false-positive experiment with redirected
  training output.

Changed files:
- `AGENTS.md`: updated autonomous queue, exact training command discipline, and
  current Chain1/Chain2 status.
- `results/tables/seed_averaged_results.csv`: regenerated with all 3 seeds for
  ISIC2018 and ACDC full BDS-Lite variants.
- `results/tables/baseline_results_month1.csv`,
  `results/tables/cross_dataset_report_v1.csv`, `paper/tables/paper_main_table.csv`,
  `paper/tables/paper_main_table.md`, and `docs/cross_dataset_report.md`:
  regenerated through `scripts/process_training_results.py`.
- `results/tables/final_ablation_status_v1.csv` and
  `docs/final_ablation_report.md`: regenerated.
- `paper/sections/04_experiments.md`: added repeated-seed summary and
  conservative interpretation.
- `plan.md`: updated current status and unfinished items.

Commands:
```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_seed_averaged_results.py
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m pytest
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py \
  --config configs/experiments/synapse_bdslite_boundary_gate_damped_surface_far_fp.yaml \
  > /tmp/bds-lite-train.log 2>&1; echo "DONE: exit $?"
PYTHONPATH=src .venv/bin/python scripts/evaluate.py \
  --checkpoint results/checkpoints/synapse_bdslite_boundary_gate_damped_surface_far_fp/best.pt \
  --split val > /tmp/bds-lite-eval.log 2>&1; echo "DONE: exit $?"
PYTHONPATH=src .venv/bin/python scripts/process_training_results.py \
  > /tmp/bds-lite-results.log 2>&1; echo "DONE: exit $?"
PYTHONPATH=src .venv/bin/python scripts/prepare_final_ablation_report.py \
  > /tmp/bds-lite-final-ablation.log 2>&1; echo "DONE: exit $?"
```

Result:
- Seed aggregation now includes all 3 seeds:
  - ISIC2018 full BDS-Lite: DSC=0.863461, IoU=0.788694, HD95=19.428451,
    ASSD=6.793360, BF1=0.559016.
  - ACDC full BDS-Lite: DSC=0.790382, IoU=0.718544, HD95=8.036378,
    ASSD=2.532352, BF1=0.847304.
- Synapse far-FP rerun completed and evaluated:
  DSC=0.845994, IoU=0.826476, HD95=11.737667, ASSD=4.265555, BF1=0.855618.
  It still fails the Synapse acceptance threshold (HD95 < 10.476, ASSD < 3.655).
- Pre-training checks before the long run passed: Ruff format check, Ruff lint,
  and 32 Pytest tests.

Problems:
- Far-FP loss remains rejected; it does not fix Synapse HD95/ASSD.
- ACDC three-seed mean is weaker than the selected seed: HD95/ASSD remain better
  than U-Net, but DSC/BF1 are seed-sensitive.

Next:
- Run Chain2: Attention U-Net baselines for ISIC2018, ACDC, Synapse, plus
  `synapse_bdslite_init_from_unet`.
- After Chain2, regenerate tables and check whether init-from-UNet passes the
  Synapse HD95/ASSD threshold; if not, run `synapse_bdslite_two_stage`.


## 2026-04-27 - chain2-chain3-baselines-and-synapse-fixes

Goal:
- Complete the pending Attention U-Net baselines and test the queued Synapse
  architectural fixes.

Changed files:
- `results/tables/baseline_results_month1.csv`, `paper/tables/paper_main_table.csv`,
  `paper/tables/paper_main_table.md`, `results/tables/cross_dataset_report_v1.csv`,
  and `docs/cross_dataset_report.md`: regenerated after Chain2.
- `scripts/prepare_final_ablation_report.py`: extended final ablation reporting
  to include init-from-UNet and two-stage Synapse fix attempts when their eval
  CSVs exist.
- `results/tables/final_ablation_status_v1.csv` and
  `docs/final_ablation_report.md`: regenerated with the new Synapse fix rows.
- `paper/sections/04_experiments.md`: filled Attention U-Net and
  init-from-UNet results; noted the rejected two-stage result.
- `plan.md`: updated current status and unfinished items.

Commands:
```bash
# Chain2, all training/eval output redirected to /tmp/bds-lite-train-chain2.log
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --config configs/experiments/isic_attention_unet_baseline.yaml
PYTHONPATH=src .venv/bin/python scripts/evaluate.py --checkpoint results/checkpoints/isic2018_attention_unet_baseline/best.pt --split val
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --config configs/experiments/acdc_attention_unet_baseline.yaml
PYTHONPATH=src .venv/bin/python scripts/evaluate.py --checkpoint results/checkpoints/acdc_attention_unet_baseline/best.pt --split val
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --config configs/experiments/synapse_attention_unet_baseline.yaml
PYTHONPATH=src .venv/bin/python scripts/evaluate.py --checkpoint results/checkpoints/synapse_attention_unet_baseline/best.pt --split val
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --config configs/experiments/synapse_bdslite_init_from_unet.yaml
PYTHONPATH=src .venv/bin/python scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_init_from_unet/best.pt --split val

PYTHONPATH=src .venv/bin/python scripts/process_training_results.py \
  > /tmp/bds-lite-results.log 2>&1; echo "DONE: exit $?"

# Chain3 fallback, output redirected to /tmp/bds-lite-train-chain3.log
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --config configs/experiments/synapse_bdslite_two_stage.yaml
PYTHONPATH=src .venv/bin/python scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_two_stage/best.pt --split val
PYTHONPATH=src .venv/bin/python scripts/prepare_final_ablation_report.py \
  > /tmp/bds-lite-final-ablation.log 2>&1; echo "DONE: exit $?"
```

Result:
- Attention U-Net baselines:
  - ISIC2018: DSC=0.862846, IoU=0.784853, HD95=20.317001, ASSD=7.111236,
    BF1=0.530973.
  - ACDC: DSC=0.789178, IoU=0.715475, HD95=6.265351, ASSD=1.939628,
    BF1=0.840750.
  - Synapse: DSC=0.843557, IoU=0.825126, HD95=12.054598, ASSD=4.711136,
    BF1=0.858520.
- Synapse init-from-UNet: DSC=0.844041, IoU=0.824700, HD95=10.832038,
  ASSD=4.005983, BF1=0.856737. It improves over full BDS-Lite HD95 but still
  fails the U-Net threshold (HD95 < 10.476, ASSD < 3.655).
- Synapse two-stage: DSC=0.794956, IoU=0.772708, HD95=15.472012,
  ASSD=6.015294, BF1=0.808844. Rejected.
- Final verification after script/report edits passed: Ruff format check, Ruff
  lint, and 32 Pytest tests.

Problems:
- Both Synapse architectural training-schedule fixes failed. The remaining
  Synapse issue is not solved by shallow loss, post-processing, initialization,
  or freezing changes.

Next:
- If continuing Synapse work, move to a more separated architecture or
  dataset-specific backbone instead of another shallow training tweak.
- Prepare qualitative comparison/boundary-map figures if the paper moves from
  evidence snapshot to full draft submission package.


## 2026-04-27 - publication-first-hr-rescue

Goal:
- Reframe the repo around a publication-first BSPC path: ISIC2018 and ACDC as
  the positive core, Synapse as a controlled limitation unless one serious
  high-resolution architectural rescue passes.

Changed files:
- `plan.md`: added current publication-first strategy, Synapse rescue
  acceptance gate, HR rescue result, and next active task.
- `src/bds_lite/models/bds_lite.py`: added `HighResolutionSpatialPath`,
  `HighResolutionFusion`, and `BDSLiteUNetHR`.
- `src/bds_lite/models/__init__.py` and `src/bds_lite/training/builders.py`:
  exported and wired `bdslite_unet_hr` / `bdslite_dual_stream_unet`.
- `tests/unit/test_builders.py`: added HR builder/forward/boundary-toggle tests
  and alias coverage.
- `configs/experiments/synapse_bdslite_hr_rescue.yaml`: added the single
  CUDA-ready Synapse HR rescue config.
- `scripts/prepare_publication_strategy_report.py`: added publication-core
  report/table generation.
- `scripts/prepare_final_ablation_report.py`: added optional HR rescue status.
- `scripts/process_training_results.py`: includes the publication strategy
  report in post-training refresh.
- `docs/publication_strategy_report.md`,
  `results/tables/publication_core_results_v1.csv`,
  `docs/final_ablation_report.md`, and
  `results/tables/final_ablation_status_v1.csv`: regenerated.
- `results/checkpoints/synapse_bdslite_hr_rescue/eval_val.csv`: generated by
  evaluation after training.

Commands:
```bash
PYTHONPATH=src .venv/bin/pytest tests/unit/test_builders.py tests/unit/test_losses.py tests/unit/test_metrics.py -q
PYTHONPATH=src .venv/bin/python scripts/prepare_publication_strategy_report.py > /tmp/bds-lite-publication-report.log 2>&1; echo "DONE: exit $?"
PYTHONPATH=src .venv/bin/python - <<'PY'
from bds_lite.utils import load_config
from bds_lite.training import build_model
cfg = load_config("configs/experiments/synapse_bdslite_hr_rescue.yaml")
model = build_model(cfg)
print(type(model).__name__)
print(cfg["experiment"]["name"], cfg["trainer"]["device"])
PY
.venv/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
PYTHONPATH=src .venv/bin/python scripts/train.py --config configs/experiments/synapse_bdslite_hr_rescue.yaml --device cuda > /tmp/bds-lite-hr-rescue-train.log 2>&1; echo "DONE: exit $?"
PYTHONPATH=src .venv/bin/python scripts/evaluate.py --checkpoint results/checkpoints/synapse_bdslite_hr_rescue/best.pt --split val > /tmp/bds-lite-hr-rescue-eval.log 2>&1; echo "DONE: exit $?"
PYTHONPATH=src .venv/bin/python scripts/prepare_publication_strategy_report.py > /tmp/bds-lite-publication-report.log 2>&1; echo "DONE: exit $?"
PYTHONPATH=src .venv/bin/python scripts/prepare_final_ablation_report.py > /tmp/bds-lite-final-ablation.log 2>&1; echo "DONE: exit $?"
.venv/bin/ruff format .
.venv/bin/python -m py_compile scripts/prepare_publication_strategy_report.py scripts/prepare_final_ablation_report.py scripts/process_training_results.py
.venv/bin/ruff format --check .
.venv/bin/ruff check .
PYTHONPATH=src .venv/bin/python -m pytest tests/unit -q
```

Result:
- Focused pre-training tests: `20 passed`; exit 0.
- CUDA check: `True`, `1`; training proceeded on CUDA.
- HR rescue training: exit 0.
- HR rescue evaluation: exit 0.
- Synapse U-Net baseline: DSC=0.844832, HD95=10.475386, ASSD=3.654300,
  BF1=0.854835.
- Synapse HR rescue: DSC=0.846577, IoU=0.826954, HD95=10.619353,
  ASSD=3.734170, BF1=0.855972.
- Acceptance decision: rejected. DSC is +0.001745 versus U-Net, but HD95 is
  +0.143967 and ASSD is +0.079870, so it does not beat U-Net on the required
  distance metrics.
- Publication table/report now separate positive core datasets from the
  diagnostic Synapse limitation.
- Final verification:
  - `.venv/bin/ruff format .`: exit 0.
  - `.venv/bin/python -m py_compile ...`: exit 0.
  - `.venv/bin/ruff format --check .`: exit 0.
  - `.venv/bin/ruff check .`: exit 0.
  - `PYTHONPATH=src .venv/bin/python -m pytest tests/unit -q`: exit 0,
    `34 passed`.

Problems:
- The serious HR architectural rescue did not clear the Synapse HD95/ASSD gate.
- Synapse remains a limitation, not a bug or a headline success.

Next:
- Stop Synapse chasing for this paper.
- Proceed to manuscript preparation and qualitative figures around the
  ISIC2018 + ACDC positive core, with Synapse written as a diagnostic limitation.

## 2026-05-13 - gsl-recovery

Goal:
- Restore the GSL (Generalised Surface Loss) implementation that was absent from
  `main` — the loss module existed in a stash commit (254d4a1) but was never
  merged. Verify the restored code produces numerically equivalent outputs to the
  historical P2 ISIC2018 seed1 run before re-enabling it.

Changed:
- `src/bds_lite/losses/gsl.py`: restored from stash; implements `gsl_alpha` and
  `gsl_region` variants.
- `src/bds_lite/losses/boundary_dou.py`: restored from stash; Boundary DoU loss.
- `src/bds_lite/losses/__init__.py`: exports `gsl_alpha`, `gsl_region`,
  `boundary_dou`.
- `src/bds_lite/training/losses.py`: wired GSL and Boundary DoU into
  `build_loss_fn`; added `LossWeights.gsl_weight` and `LossWeights.dou_weight`.
- `scripts/train.py`: added `--loss-override` flag to support per-seed GSL
  schedule configs.
- `tests/unit/test_recent_boundary_losses.py`: unit tests covering both GSL
  variants and Boundary DoU.
- `configs/experiments/isic2018_p2_unet_gsl_seed1.yaml`: config used for
  verification run.
- `scripts/train_step3_10ep.py`: 10-epoch verification script for bitwise check
  against historical P2 output.
- `manuscript_v2/ANCHOR_ABSTRACT.md`: added hold flag pending GSL re-run
  verification (commit 066cb58).

Commands:
- `git stash show -p 254d4a1 -- src/bds_lite/losses/ > /tmp/gsl_stash.patch`
- `python3 -m pytest tests/unit/test_recent_boundary_losses.py -q`
- `python3 scripts/train_step3_10ep.py --config configs/experiments/isic2018_p2_unet_gsl_seed1.yaml`
- `ruff format . && ruff check .`

Result:
- GSL integration committed at `dbcb814`.
- Verification: `gsl_alpha` term bitwise identical to historical P2 seed1 output
  over 10 epochs; `gsl_region` term matches to ~1e-9.
- Pre-commit hooks passed after fixing two ruff violations in
  `train_step3_10ep.py` (F841 unused variable, E402 import order).

Problems:
- Pre-commit hooks initially failed on `train_step3_10ep.py` due to F841 and
  E402; resolved inline without `--no-verify`.
- The `p = 0.026` claim in `ANCHOR_ABSTRACT.md` has no supporting artifact in
  the repository; hold flag added, not removed.

Next:
- Commit splits manifests to lock in reproducible train/val/test partitions.

## 2026-05-13 - splits-manifests

Goal:
- Commit reproducible JSON manifests of the train/val/test splits for all three
  datasets so that exact partition identity can be verified from committed files
  without re-running converters.

Changed:
- `data/splits/isic2018/{train,val}.json`: 2075 / 519 filenames, seed 2026.
- `data/splits/acdc/{train,val,test}.json`: 1350 / 186 / 366 slice IDs,
  seed 2026.
- `data/splits/synapse/{train,val,test}.json`: 1738 / 473 / 1568 slice IDs,
  seed 2026.

Commands:
- `python3 -c "from bds_lite.data import NpzSegmentationDataset; ..."`
  (verified manifest contents against on-disk NPZ directories)
- `python3 scripts/verify_splits_reproducibility.py --seed 2026`

Result:
- All eight manifests committed at `3ed1e68`.
- Reproducibility verified: re-running converters with seed=2026 on ISIC2018,
  ACDC, and Synapse produces byte-identical split assignments.
- Synapse test partition confirmed present at 1568 slices — the initial recovery
  summary's claim of "Synapse test split absent" was a misdiagnosis; the split
  existed but raw source data was temporarily missing from this machine.

Problems:
- Synapse verification initially blocked because `/home/jingchen/vit-refined/
  HiFormer/data/Synapse/` was not mounted. Resolved by locating the source
  directory and copying raw H5 files to `data/raw/Synapse/` (gitignored).
- ISIC2018 has no held-out test partition; the converter never created one. The
  val split (519 images) is the closest available held-out set and is used as
  the de facto evaluation partition throughout.

Next:
- Rebase recovery branch onto `origin/main` and push.

## 2026-05-13 - branch-hygiene

Goal:
- Merge the `agent-run-recovery` branch (39 commits) into `main` and publish,
  resolving the divergence that accumulated while recovery work was offline.

Changed:
- No source or config files changed; rebase-only operation.

Commands:
- `git fetch origin`
- `git rebase origin/main`
- `git push origin main`

Result:
- 39 recovery commits rebased onto `origin/main`, which had two PR merge commits
  (`ec8f51d`, `5914d35`) from April that were absent from the recovery branch.
- Pushed to origin; tip is `3ed1e68`.
- Branches `agent-run-recovery` and `rebase-scratch` kept locally as safety
  nets.

Problems:
- `origin/main` had moved while the recovery branch was offline. Rebase required
  rather than fast-forward merge.

Next:
- Run post-recovery evaluation to determine which manuscript numerical claims are
  reproducible from committed code and committed checkpoints.

## 2026-05-13 - post-recovery-eval

Goal:
- Determine which manuscript numerical claims are reproducible from committed
  code, committed checkpoints, and committed data splits. Re-evaluate all
  historical checkpoints on current splits with `strict=True` loading and
  document discrepancies.

Changed:
- `scripts/adapt_legacy_bdslite_for_eval.py` (untracked): adapter for ISIC2018
  ablation-era BDS-Lite checkpoints; handles `model.name=bds_lite` → rename to
  `bdslite_unet` and `projection_head` → `seg_to_boundary_feature` state-dict
  rename (pure rename, identical shape and role).
- `scripts/adapt_legacy_acdc_for_eval.py` (untracked): same adapter for ACDC.
- `scripts/adapt_legacy_synapse_for_eval.py` (untracked): same adapter for
  Synapse.
- `scripts/eval_gsl_baselines.py` (untracked): evaluates p2 (UNet+GSL) and p2b
  (BDS-Lite gate-removed+GSL) checkpoints on all three datasets; honours
  `inference_gate=removed` by setting `model.boundary_gate = None` post-load.
- `scripts/eval_p1b_gate_comparison.py` (untracked): evaluates p1b BDS-Lite full
  vs gate-removed checkpoints on all three datasets.
- `results/post_recovery_eval/isic2018_val/summary.csv` (untracked): 6
  checkpoints (BDS-Lite full + UNet, seeds 1–3) on ISIC2018 val.
- `results/post_recovery_eval/acdc_test/summary.csv` (untracked): 6 checkpoints
  on ACDC test.
- `results/post_recovery_eval/synapse_test/summary.csv` (untracked): 6
  checkpoints on Synapse test.
- `results/post_recovery_eval/effect_sizes.csv` (untracked): per-dataset,
  per-metric effect sizes (mean diff, std diff, sign pattern) without p-values.
- `results/post_recovery_eval/gsl_baselines/summary.csv` (untracked): 18
  checkpoints (UNet+GSL and BDS-Lite-gate-removed+GSL, seeds 1–3, all three
  datasets).
- `results/post_recovery_eval/p1b_gate_comparison/summary.csv` (untracked): 18
  p1b checkpoints (BDS-Lite full vs gate-removed, seeds 1–3, all three
  datasets).
- `docs/gate_analysis_writeup.md` (untracked): ~3-paragraph plain-English
  summary of gate weight, activation, and removal analysis.

Commands:
- `python3 scripts/adapt_legacy_bdslite_for_eval.py`
- `python3 scripts/adapt_legacy_acdc_for_eval.py`
- `python3 scripts/adapt_legacy_synapse_for_eval.py`
- `python3 scripts/eval_gsl_baselines.py`
- `python3 scripts/eval_p1b_gate_comparison.py`
- `python3 -c "import torch; torch.load(..., weights_only=False)"`
  (config inspection for each checkpoint class before writing adapter scripts)

Result:
- All 48 checkpoints loaded with `strict=True`; no load failures.
- DSC reproduces to ~1e-9 across all datasets, confirming current test splits
  are the same partitions used in historical evaluation.
- BF1 drifted: current code reports ~+0.021–0.026 higher than historical values
  on ACDC. Historical evaluation script wrote `.json` output and is unrecoverable
  from git; no parameter combination in current code reproduces historical values.
  Direction of BDS vs UNet comparison is consistent across partitions.
- Synapse HD95/ASSD are inflated ~4× versus historical values: the converter
  never wrote spacing metadata, so current evaluation uses unit pixel spacing
  while historical scripts hardcoded physical spacing (~0.25 mm/pixel). Ordering
  between BDS and UNet is preserved.
- ISIC2018 historical "test" partition is lost: the converter created only
  train/val; no 3-way split mechanism exists in the committed codebase.
  Abstract BF1 values 0.4900/0.4826 are not reproducible from any committed
  artifact.
- `p = 0.026` in the abstract has no supporting artifact; minimum achievable
  two-sided paired Wilcoxon p-value at n=3 is 0.25.
- BDS-Lite shows consistent directional BF1 improvement over vanilla U-Net on
  ISIC2018 val (+0.007) and Synapse test (+0.003); ACDC is effectively tied
  (mean diff +0.0002).
- Gate-removable claim is robust: 25/27 full-vs-removed pairs differ <0.001 on
  both Dice and BF1; maximum absolute Dice difference across all pairs is 0.0048.
- GSL comparison: UNet+GSL is competitive or marginally better than
  BDS-Lite+GSL on 2 of 3 datasets (ISIC2018 and Synapse); BDS-Lite's
  architectural contribution (gate-removable, no inference overhead) is distinct
  from numerically dominating boundary-aware losses.

Problems:
- Historical BF1 implementation is unrecoverable; current values are not
  directly comparable to the submitted manuscript v2 numbers.
- Synapse distance metrics are measured in pixels throughout this evaluation;
  physical-spacing values cannot be recovered without raw H5 spacing fields
  (which are absent from the Synapse H5 files used for conversion).
- The `p = 0.026` claim must be removed or replaced with a qualitative
  consistency statement before resubmission.

Next:
- Reconcile manuscript_v2 claims with post-recovery numbers.
- Replace significance-based framing with qualitative consistency + effect size.
- Produce boundary-map figures from p1b checkpoints.
- Revise ANCHOR_ABSTRACT.md to lift the hold flag with corrected evidence.
