#!/usr/bin/env bash
# End-to-end reproduction map for the BDS-Lite study.
#
# This repository ships the lightweight machine-readable artifacts of record
# (per-case CSVs, aggregate CSV/JSON, cluster statistics, figures). The raw
# datasets, trained checkpoints, and per-case prediction arrays are NOT in git:
#   - raw datasets: download per docs/data_access.md (third-party licenses)
#   - checkpoints + prediction arrays: Zenodo (see ZENODO_MANIFEST.csv)
#
# Stages that need raw data / a GPU / the Zenodo bundle are guarded so this script
# is safe to read top-to-bottom. Set RUN_HEAVY=1 to attempt the compute stages.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="src:${PYTHONPATH:-}"
PY="${PYTHON:-python}"
RUN_HEAVY="${RUN_HEAVY:-0}"

echo "==> 0. Environment"
echo "    pip install -r requirements.txt   (exact pins; see docs/compute.md)"

echo "==> 1. Data (needs raw downloads; see docs/data_access.md)"
if [[ -d data/processed/isic2018/train ]]; then
  echo "    processed data present."
else
  echo "    [skipped] run scripts/convert_{isic2018,acdc,synapse}.py then"
  echo "    scripts/prepare_boundary_targets.py to build data/processed/."
fi

if [[ "$RUN_HEAVY" != "1" ]]; then
  echo "==> Compute stages (train/evaluate/analysis) are gated. Re-run with RUN_HEAVY=1."
  echo "    The committed artifacts of record already reflect the stages below:"
  echo "      results/main_test_means.csv, results/matched_gsl_summary.csv,"
  echo "      results/profiling/*, analysis/outputs/cluster_level_statistics.csv,"
  echo "      analysis/outputs/failure_case_manifest.csv, figures/*."
  exit 0
fi

echo "==> 2. Train (GPU; 150 epochs each). Config of record: configs/run_resolved/"
for ds in isic2018 acdc synapse; do
  for model in unet bds_lite_full; do
    for seed in 1 2 3; do
      "$PY" scripts/train.py --config "configs/run_resolved/${ds}_${model}_seed${seed}.yaml"
    done
  done
done
echo "    U-Net+GSL comparator (matched protocol):"
bash scripts/gsl/run_phase16_gsl.sh

echo "==> 3. Evaluate -> per-case predictions + aggregate evals"
"$PY" scripts/reeval_main_seeds.py
"$PY" scripts/profile_model.py

echo "==> 4. Cluster-aware analysis (SINGLE full-script run; see docs/statistics.md)"
echo "    Requires the Zenodo prediction arrays under outputs/evaluations/predictions/."
"$PY" analysis/generate_rescue_analysis.py

echo "==> Done. Compare regenerated analysis/outputs/ against the committed copies."
