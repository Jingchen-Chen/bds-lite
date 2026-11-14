#!/bin/bash
# Train all 9 EGE-UNet runs sequentially.
# Output: outputs/runs/<dataset>_ege_unet_seed<N>/best.pt
# Logs:   /tmp/ege_train_<dataset>_seed<N>.log

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

for cfg in \
    configs/experiments/isic2018_ege_unet_seed1.yaml \
    configs/experiments/isic2018_ege_unet_seed2.yaml \
    configs/experiments/isic2018_ege_unet_seed3.yaml \
    configs/experiments/acdc_ege_unet_seed1.yaml \
    configs/experiments/acdc_ege_unet_seed2.yaml \
    configs/experiments/acdc_ege_unet_seed3.yaml \
    configs/experiments/synapse_ege_unet_seed1.yaml \
    configs/experiments/synapse_ege_unet_seed2.yaml \
    configs/experiments/synapse_ege_unet_seed3.yaml
do
    name=$(basename "$cfg" .yaml)
    logfile="/tmp/ege_train_${name}.log"
    echo "=== $(date -Iseconds)  starting $name  =="
    python scripts/train.py --config "$cfg" >"$logfile" 2>&1
    tail -2 "$logfile"
    echo
done
echo "=== $(date -Iseconds)  all EGE-UNet runs complete ==="
