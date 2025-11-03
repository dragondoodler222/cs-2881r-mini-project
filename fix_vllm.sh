#!/bin/bash
#
# Fix corrupted VLLM installation
# This script cleans Python bytecode cache and reinstalls VLLM
#

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              Fixing Corrupted VLLM Installation                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Clean Python bytecode cache
echo "Step 1/4: Cleaning Python bytecode cache..."
find ~/miniconda3/envs/prune_llm_py311 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find ~/miniconda3/envs/prune_llm_py311 -type f -name "*.pyc" -delete 2>/dev/null || true
echo "✓ Cache cleaned"
echo ""

# Step 2: Activate environment
echo "Step 2/4: Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prune_llm_py311
echo "✓ Environment activated: prune_llm_py311"
echo ""

# Step 3: Reinstall VLLM
echo "Step 3/4: Reinstalling VLLM..."
pip uninstall vllm -y > /dev/null 2>&1 || true
pip install vllm==0.11.0
echo "✓ VLLM reinstalled"
echo ""

# Step 4: Verify installation
echo "Step 4/4: Verifying VLLM installation..."
python -c "from vllm import LLM; print('✅ VLLM imported successfully')"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ Fix Complete!                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "You can now re-run your evaluation:"
echo "  ./run.sh"
echo ""

