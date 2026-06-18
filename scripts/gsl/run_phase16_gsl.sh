#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

OUT="results/gsl"
LOG_DIR="$OUT/logs"
STATUS_FILE="$OUT/execution_status.csv"
START_EPOCH="$(date +%s)"
STARTED_AT="$(date --iso-8601=seconds)"
MAX_SECONDS=28800
mkdir -p "$LOG_DIR" "$OUT/evaluations"

if [[ -e "$STATUS_FILE" ]]; then
  echo "Refusing to overwrite existing $STATUS_FILE" >&2
  exit 2
fi

printf 'dataset,seed,stage,split,status,started_at,ended_at,command\n' > "$STATUS_FILE"

run_command() {
  local dataset="$1"
  local seed="$2"
  local stage="$3"
  local split="$4"
  local log_path="$5"
  shift 5
  local command_text
  command_text="$(printf '%q ' "$@")"
  local started ended
  started="$(date --iso-8601=seconds)"
  if "$@" >"$log_path" 2>&1; then
    ended="$(date --iso-8601=seconds)"
    printf '%s,%s,%s,%s,completed,%s,%s,"%s"\n' \
      "$dataset" "$seed" "$stage" "$split" "$started" "$ended" "$command_text" \
      >> "$STATUS_FILE"
  else
    ended="$(date --iso-8601=seconds)"
    printf '%s,%s,%s,%s,failed,%s,%s,"%s"\n' \
      "$dataset" "$seed" "$stage" "$split" "$started" "$ended" "$command_text" \
      >> "$STATUS_FILE"
    return 1
  fi
}

for dataset in isic2018 acdc synapse; do
  for seed in 1 2 3; do
    experiment="phase16_matched_gsl_${dataset}_seed${seed}"
    config="configs/gsl/${experiment}.yaml"
    run_dir="$OUT/runs/$experiment"
    checkpoint_dir="$OUT/checkpoints/$experiment"

    elapsed_seconds=$(( $(date +%s) - START_EPOCH ))
    if (( elapsed_seconds >= MAX_SECONDS )); then
      echo "Stopping before $experiment: eight-hour wall-clock limit reached" >&2
      exit 6
    fi
    if [[ -e "$run_dir" || -e "$checkpoint_dir" ]]; then
      echo "Refusing to overwrite existing output for $experiment" >&2
      exit 3
    fi
    free_kb="$(df --output=avail . | tail -n 1)"
    if (( free_kb < 15728640 )); then
      echo "Stopping before $experiment: less than 15 GiB free disk" >&2
      exit 4
    fi

    run_command \
      "$dataset" "$seed" train "" "$LOG_DIR/${experiment}_train.log" \
      env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python \
      scripts/gsl/train_memory_bounded_gsl.py \
      --config "$config" --device cuda

    if rg -n '\b(nan|inf)\b' "$LOG_DIR/${experiment}_train.log" \
      | rg -v 'best val_dsc=-inf'; then
      echo "Stopping after $experiment: non-finite value found in log" >&2
      exit 5
    fi

    if [[ "$dataset" == "isic2018" ]]; then
      splits=(val)
    else
      splits=(val test)
    fi
    for split in "${splits[@]}"; do
      run_command \
        "$dataset" "$seed" evaluate "$split" \
        "$LOG_DIR/${experiment}_evaluate_${split}.log" \
        env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
        .venv/bin/python scripts/evaluate.py \
        --checkpoint "$checkpoint_dir/best.pt" \
        --split "$split" \
        --output "$OUT/evaluations/${experiment}_${split}.csv"
    done
  done
done

run_command \
  "all" "all" aggregate "" "$LOG_DIR/aggregate.log" \
  .venv/bin/python \
  scripts/gsl/aggregate_phase16_gsl.py

ENDED_AT="$(date --iso-8601=seconds)"
printf '{"started_at":"%s","ended_at":"%s","status":"completed"}\n' \
  "$STARTED_AT" "$ENDED_AT" > "$OUT/execution_summary.json"
