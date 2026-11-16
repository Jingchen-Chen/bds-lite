#!/usr/bin/env bash
# Launch the follow-up supervisor (PLAN.md Phase 20, step 2) only when the GPU is free.
# Another project shared this GPU on 2026-11-09; the follow-up never runs beside it. The script
# polls every 60 s and launches once no compute process has been on the GPU for 10 consecutive
# minutes and memory use is below 1 GiB (the supervisor then applies its own cool-launch guard).
# It has no external-timestamping step and does not wait for any blockchain confirmation
# (Amendment 1, protocols/followup/AMENDMENT_1.md).
#   bash analysis/followup/wait_and_launch.sh e1e2
set -euo pipefail
cd "$(dirname "$0")/../.."
queue="${1:?queue name (e1e2 or e3)}"
mkdir -p results/followup/supervision
log=results/followup/supervision/wait_${queue}.log
free_since=""
while true; do
  if [[ -e revision/STOP_KD_TRAINING ]]; then
    echo "$(date -u +%FT%TZ) stop marker present; not launching" | tee -a "$log"; exit 2
  fi
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  now=$(date +%s)
  if [[ -z "$apps" && "$mem" -lt 1024 ]]; then
    free_since=${free_since:-$now}
    if (( now - free_since >= 600 )); then break; fi
  else
    free_since=""
  fi
  echo "$(date -u +%FT%TZ) apps=[${apps//$'\n'/,}] mem=${mem}MiB free_for=$(( free_since ? now - free_since : 0 ))s" >> "$log"
  sleep 60
done
echo "$(date -u +%FT%TZ) GPU free for 10 min; launching supervisor for $queue" | tee -a "$log"
setsid nohup env PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python scripts/kd_followup/supervise.py \
  --queue "protocols/followup/queue_followup_${queue}.yaml" \
  > "results/followup/supervision/supervisor_stdout_${queue}.log" 2>&1 &
echo "$(date -u +%FT%TZ) supervisor pid $!" | tee -a "$log"
