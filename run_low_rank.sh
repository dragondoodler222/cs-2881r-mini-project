#!/bin/bash

# Minimal variant of your runner to use *rank pruning* (low-rank) instead of sparsity pruning.
# Only changes: switch executed file to main_low_rank.py and loop over --rank values.
# Everything else (environment, model list, logging layout) mirrors your original style.

# --- Environment (kept) ---
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HUGGINGFACE_HUB_TIMEOUT=300
export REQUESTS_TIMEOUT=300
# export HF_DATASETS_OFFLINE=1   # uncomment if you want to force offline mode

# --- Config you likely already had ---
MODELS=( "qwen2_5-7b" )           # adapt if you want a different model id
PRUNE_DATA="align"                  # dataset used to build low-rank projections (e.g., align, align_short, alpaca_cleaned_no_safety)
RANKS=(250)            # <-- requested sweep
TOP_REMOVE=0                       # set to 1 to pass --top_remove (remove most-important ranks). default here: do NOT pass
SAVE_ROOT="out_low_rank"           # output root
LOGDIR="logs_low_rank"             # logs

mkdir -p "$LOGDIR"

timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }

run_eval() {
  local model="$1"
  local rank="$2"

  local tag="model=${model}_rank=${rank}"
  local ts="$(timestamp)"
  local logfile="${LOGDIR}/${ts}_${tag}.log"

  echo ">>> Running low-rank prune: ${tag}  (prune_data=${PRUNE_DATA})"
  echo "    Log: ${logfile}"

  # Build args
  PY=python
  SCRIPT=main_low_rank.py
  ARGS=(
    --model "$model"
    --prune_method low_rank
    --prune_data "$PRUNE_DATA"
    --rank "$rank"
    --save "${SAVE_ROOT}/${model}/low_rank/rank_${rank}"
    --eval_zero_shot
    --eval_attack
    --save_attack_res
  )

  if [[ "$TOP_REMOVE" == "1" ]]; then
    ARGS+=( --top_remove )
  fi

  # Run
  set -o pipefail
  $PY "$SCRIPT" "${ARGS[@]}" 2>&1 | tee "$logfile"
  local status=${PIPESTATUS[0]}

  if [[ $status -ne 0 ]]; then
    echo "FAILED: ${tag} (status $status)"
    exit $status
  else
    echo "DONE: ${tag}"
  fi
}

# --- Sweep ---
for model in "${MODELS[@]}"; do
  for rank in "${RANKS[@]}"; do
    run_eval "$model" "$rank"
    sleep 5   # brief pause between runs
  done
done

echo "All low-rank runs completed."
