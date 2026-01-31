#!/bin/bash
#
# COMPLETE DEPLOYMENT GUIDE FOR OOM ERROR FIX
#
# This guide walks through the complete process to fix the 17 OOM errors
# in text_html and mix_html checkpoints using automatic text truncation.
#

set -e

WORKSPACE="/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL"
cd "$WORKSPACE/inference"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "================================================================================"
echo "COMPLETE DEPLOYMENT GUIDE - OOM ERROR FIX WITH TEXT TRUNCATION"
echo "================================================================================"
echo -e "${NC}"

# Step 0: Verify prerequisites
echo -e "\n${YELLOW}[Step 0] Verifying prerequisites...${NC}"
echo "  Checking Python..."
python --version
echo "  Checking model directory..."
if [ -d "/data/pan/4xin/models/Qwen3-VL-8B-Instruct" ]; then
    echo -e "    ${GREEN}✓${NC} Model directory exists"
else
    echo -e "    ${YELLOW}✗${NC} Model directory not found"
fi

echo "  Checking data directory..."
if [ -d "/data/pan/4xin/datasets/RealHiTBench" ]; then
    echo -e "    ${GREEN}✓${NC} Data directory exists"
else
    echo -e "    ${YELLOW}✗${NC} Data directory not found"
fi

# Step 1: Display analysis
echo -e "\n${YELLOW}[Step 1] Current Status Analysis${NC}"
echo ""
echo "  Text HTML mode:"
echo "    - Completed: 3043 samples ✓"
echo "    - Failed with OOM: 17 samples ⚠️"
echo "    - OOM samples: [2747, 2748, 2749, 2750, 2751, 2758-2763, 2966-2968, 3019-3021]"
echo ""
echo "  Mix HTML mode:"
echo "    - Completed: 3043 samples ✓"
echo "    - Failed with ERROR: 17 samples ⚠️"
echo "    - Same sample IDs as text_html"
echo ""
echo "  Processing approach:"
echo "    - Use existing 3043 successful results"
echo "    - Reprocess only 17 failed samples with text truncation (MAX_INPUT_TOKENS=100,000)"
echo "    - Avoid re-running entire dataset (~15 hours saved)"
echo ""

# Step 2: Copy checkpoints
echo -e "${YELLOW}[Step 2] Setting up directories and copying checkpoints...${NC}"
mkdir -p ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/
mkdir -p ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_mix_html_truncate/

echo "  Copying text_html checkpoint..."
cp ../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_html_a100/checkpoint.json \
   ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/checkpoint.json
TEXT_OOM=$(grep -c '\[ERROR\] OOM:' ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/checkpoint.json || echo 0)
echo -e "    ${GREEN}✓${NC} Copied with $TEXT_OOM OOM errors"

echo "  Copying mix_html checkpoint..."
cp ../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_mix_html_default/checkpoint.json \
   ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_mix_html_truncate/checkpoint.json
MIX_OOM=$(grep -c '\[ERROR\]' ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_mix_html_truncate/checkpoint.json || echo 0)
echo -e "    ${GREEN}✓${NC} Copied with $MIX_OOM ERROR samples"
echo ""

# Step 3: Test truncation function
echo -e "${YELLOW}[Step 3] Testing truncation functionality...${NC}"
python test_truncation.py 2>&1 | grep -E "(✓|✗|Loading|Importing|Testing)" || true
echo ""

# Step 4: Display run commands
echo -e "${YELLOW}[Step 4] Next: Run the reprocessing commands${NC}"
echo ""
echo -e "  ${BLUE}Option A: Interactive mode (recommended for monitoring)${NC}"
echo "    python run_text_html_truncate.py"
echo "    python run_mix_html_truncate.py"
echo ""
echo -e "  ${BLUE}Option B: Background mode (detach from terminal)${NC}"
echo "    nohup python run_text_html_truncate.py > text_html.log 2>&1 &"
echo "    nohup python run_mix_html_truncate.py > mix_html.log 2>&1 &"
echo ""
echo -e "  ${BLUE}Option C: Use tmux for multiple windows${NC}"
echo "    tmux new-session -d -s qwen3vl"
echo "    tmux send-keys -t qwen3vl 'cd $WORKSPACE/inference && python run_text_html_truncate.py' Enter"
echo "    tmux new-window -t qwen3vl -n mix"
echo "    tmux send-keys -t qwen3vl:mix 'cd $WORKSPACE/inference && python run_mix_html_truncate.py' Enter"
echo ""

# Step 5: Expected results
echo -e "${YELLOW}[Step 5] Expected Results${NC}"
echo ""
echo "  Time estimate: ~30 minutes per modality"
echo "  Output directory: result/qwen3vl_local_a100_truncate/"
echo ""
echo "  Typical log output:"
echo "    Resuming from checkpoint:"
echo "      - Successful: 3043 queries"
echo "      - Failed (will retry): 17 queries"
echo ""
echo "  For each failed sample:"
echo "    [TRUNCATE] Input too large (334,162 tokens), truncating to 100,000"
echo "    [TRUNCATE] Result: 99,847 tokens (original: 334,162)"
echo "    Prediction: [successful answer]"
echo "    Processing Time: ~45s per sample"
echo ""

# Step 6: Verification
echo -e "${YELLOW}[Step 6] Verification (run after completion)${NC}"
echo ""
echo "  Check text_html results:"
echo "    python << 'EOF'"
echo "import json"
echo "with open('../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/checkpoint.json') as f:"
echo "    data = json.load(f)"
echo "    errors = len([r for r in data['results'] if r['Prediction'].startswith('[ERROR')])"
echo "    print(f'Total: {len(data[\"results\"])}')"
echo "    print(f'Errors: {errors}')"
echo "    print('✓ All fixed!' if errors == 0 else f'✗ {errors} still failing')"
echo "EOF"
echo ""

echo -e "${GREEN}"
echo "================================================================================"
echo "SETUP COMPLETE - Ready to start reprocessing"
echo "================================================================================"
echo -e "${NC}"
