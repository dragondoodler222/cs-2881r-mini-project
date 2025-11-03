#!/bin/bash

# Configuration
METHODS=("wanda")  # Can add "wandg" "random" if needed
MODELS=("llama2-7b-chat-hf" "qwen-7b-chat")
SPARSITY_RATIOS=(0.00 0.01 0.02 0.03 0.04 0.05)
TYPE="unstructured"

# Create log directory
LOG_DIR="eval_logs"
mkdir -p $LOG_DIR

# Function to run evaluation
run_eval() {
    local model=$1
    local method=$2
    local sparsity=$3
    local save_dir="out/${model}/${TYPE}/${method}/eval_${sparsity}"
    
    echo "Starting evaluation: model=${model} method=${method} sparsity=${sparsity}"
    
    # Run evaluation with background process but wait for it to complete
    nohup python main.py \
        --model $model \
        --prune_method $method \
        --prune_data align \
        --sparsity_ratio $sparsity \
        --sparsity_type $TYPE \
        --neg_prune \
        --save $save_dir \
        --eval_zero_shot \
        --eval_attack \
        --save_attack_res > "${LOG_DIR}/${model}_${method}_${sparsity}.log" 2>&1
    
    # Check if the run was successful
    if [ $? -eq 0 ]; then
        echo "Completed: model=${model} method=${method} sparsity=${sparsity}"
    else
        echo "Failed: model=${model} method=${method} sparsity=${sparsity}"
        exit 1
    fi
}

# Main execution loop
for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        for sparsity in "${SPARSITY_RATIOS[@]}"; do
            run_eval $model $method $sparsity
            
            # Optional: add a small delay between runs to let GPU cool down
            sleep 30
        done
    done
done

echo "All evaluations completed!"