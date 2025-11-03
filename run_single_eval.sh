#!/bin/bash

# Quick script to run a single evaluation with increased timeouts
# Usage: ./run_single_eval.sh <model> <method> <sparsity>

# Increase HuggingFace timeouts to prevent network errors
export HF_HUB_DOWNLOAD_TIMEOUT=300  # 5 minutes
export HUGGINGFACE_HUB_TIMEOUT=300
export REQUESTS_TIMEOUT=300

MODEL=${1:-qwen2_5-7b}
METHOD=${2:-wanda}
SPARSITY=${3:-0.00}
TYPE="unstructured"

save_dir="out/${MODEL}/${TYPE}/${METHOD}/eval_${SPARSITY}"
log_file="eval_logs/${MODEL}_${METHOD}_${SPARSITY}.log"

echo "Running evaluation: model=${MODEL} method=${METHOD} sparsity=${SPARSITY}"
echo "Log file: ${log_file}"

python main.py \
    --model $MODEL \
    --prune_method $METHOD \
    --prune_data align \
    --sparsity_ratio $SPARSITY \
    --sparsity_type $TYPE \
    --neg_prune \
    --save $save_dir \
    --eval_zero_shot \
    --eval_attack \
    --save_attack_res 2>&1 | tee "${log_file}"

echo "Evaluation completed!"
