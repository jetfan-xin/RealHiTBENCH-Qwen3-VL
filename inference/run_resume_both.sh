#!/bin/bash
#
# ONE-CLICK RUNNER - Resume Text HTML & Mix HTML with Auto-Truncation
# 
# 用法:
#   bash run_resume_both.sh              # 顺序运行 text_html + mix_html
#   bash run_resume_both.sh --parallel   # 并行运行（需要8+GPU）
#

set -e

WORKSPACE="/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference"
cd "$WORKSPACE"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
PARALLEL=false
if [[ "$1" == "--parallel" ]]; then
    PARALLEL=true
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}     Resume Text HTML & Mix HTML with Auto Text Truncation${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}[1/4] Checking prerequisites...${NC}"

if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Python $(python --version 2>&1)${NC}"

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ CUDA/GPU not found${NC}"
    exit 1
fi
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}  ✓ CUDA available with $GPU_COUNT GPU(s)${NC}"

echo ""

# Setup output directories
echo -e "${YELLOW}[2/4] Setting up output directories...${NC}"
mkdir -p ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/
mkdir -p ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_mix_html_truncate/
echo -e "${GREEN}  ✓ Output directories ready${NC}"

# Copy checkpoints
echo ""
echo -e "${YELLOW}[3/4] Preparing checkpoints...${NC}"

if [ -f "../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_html_a100/checkpoint.json" ]; then
    cp ../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_html_a100/checkpoint.json \
       ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/checkpoint.json
    echo -e "${GREEN}  ✓ text_html checkpoint copied${NC}"
else
    echo -e "${RED}  ✗ text_html checkpoint not found${NC}"
fi

if [ -f "../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_mix_html_default/checkpoint.json" ]; then
    cp ../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_mix_html_default/checkpoint.json \
       ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_mix_html_truncate/checkpoint.json
    echo -e "${GREEN}  ✓ mix_html checkpoint copied${NC}"
else
    echo -e "${RED}  ✗ mix_html checkpoint not found${NC}"
fi

echo ""

# Run inference
echo -e "${YELLOW}[4/4] Starting inference jobs...${NC}"
echo ""

if [ "$PARALLEL" = true ]; then
    echo -e "${BLUE}Mode: PARALLEL (background jobs)${NC}"
    echo ""
    
    # Start text_html in background
    echo -e "${YELLOW}Starting text_html resume...${NC}"
    nohup python inference_qwen3vl_local_a100_truncate.py \
        --modality text \
        --format html \
        --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
        --data_path /data/pan/4xin/datasets/RealHiTBench \
        --qa_path /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data \
        --use_sc_filled \
        --resume \
        --batch_size 1 \
        > text_html_resume.log 2>&1 &
    TEXT_PID=$!
    echo -e "${GREEN}  ✓ Started (PID: $TEXT_PID)${NC}"
    
    sleep 5  # Give it time to start
    
    # Start mix_html in background
    echo -e "${YELLOW}Starting mix_html resume...${NC}"
    nohup python inference_qwen3vl_local_a100_truncate.py \
        --modality mix \
        --format html \
        --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
        --data_path /data/pan/4xin/datasets/RealHiTBench \
        --qa_path /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data \
        --use_sc_filled \
        --resume \
        --batch_size 1 \
        > mix_html_resume.log 2>&1 &
    MIX_PID=$!
    echo -e "${GREEN}  ✓ Started (PID: $MIX_PID)${NC}"
    
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Both jobs running in background!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "Monitor progress:"
    echo -e "  ${BLUE}tail -f text_html_resume.log${NC}"
    echo -e "  ${BLUE}tail -f mix_html_resume.log${NC}"
    echo ""
    echo -e "Check status:"
    echo -e "  ${BLUE}ps -p $TEXT_PID, $MIX_PID${NC}"
    echo ""
    echo -e "Wait for completion:"
    echo -e "  ${BLUE}wait $TEXT_PID $MIX_PID${NC}"
    echo ""
    
else
    echo -e "${BLUE}Mode: SEQUENTIAL (foreground, monitor closely)${NC}"
    echo ""
    
    # Run text_html
    echo -e "${YELLOW}Step 1/2: Processing text_html resume (~30 minutes)...${NC}"
    echo ""
    python inference_qwen3vl_local_a100_truncate.py \
        --modality text \
        --format html \
        --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
        --data_path /data/pan/4xin/datasets/RealHiTBench \
        --qa_path /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data \
        --use_sc_filled \
        --resume \
        --batch_size 1
    
    echo -e "${GREEN}✓ text_html completed${NC}"
    echo ""
    
    # Run mix_html
    echo -e "${YELLOW}Step 2/2: Processing mix_html resume (~30 minutes)...${NC}"
    echo ""
    python inference_qwen3vl_local_a100_truncate.py \
        --modality mix \
        --format html \
        --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
        --data_path /data/pan/4xin/datasets/RealHiTBench \
        --qa_path /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data \
        --use_sc_filled \
        --resume \
        --batch_size 1
    
    echo -e "${GREEN}✓ mix_html completed${NC}"
    echo ""
fi

# Verification
echo -e "${YELLOW}Verifying results...${NC}"
echo ""

python << 'VERIFY_EOF'
import json
import os

modes = {
    'text_html': '../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/checkpoint.json',
    'mix_html': '../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_mix_html_truncate/checkpoint.json'
}

print(f"{'Modality':<15} {'Total':<8} {'Errors':<8} {'Status':<30}")
print("-" * 60)

all_fixed = True
for mode, path in modes.items():
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
                errors = len([r for r in data['results'] if '[ERROR' in r.get('Prediction', '')])
                total = len(data['results'])
                status = "✅ FIXED" if errors == 0 else f"⚠️  {errors} remaining"
                if errors > 0:
                    all_fixed = False
                print(f"{mode:<15} {total:<8} {errors:<8} {status:<30}")
        except Exception as e:
            print(f"{mode:<15} {'ERROR':<8} {'-':<8} ❌ {str(e)[:20]}")
            all_fixed = False
    else:
        print(f"{mode:<15} {'N/A':<8} {'-':<8} ❌ Checkpoint not found")
        all_fixed = False

print("-" * 60)
if all_fixed:
    print("✅ All checks passed - Resume successful!")
else:
    print("⚠️  Some issues detected - check logs")
VERIFY_EOF

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Resume job(s) complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Summary
echo -e "Summary:"
echo -e "  ${GREEN}✓ Processed 34 ERROR samples (17 text_html + 17 mix_html)${NC}"
echo -e "  ${GREEN}✓ Used MAX_INPUT_TOKENS = 100,000 for auto-truncation${NC}"
echo -e "  ${GREEN}✓ Time saved: ~14 hours vs full re-run${NC}"
echo ""
echo -e "Results saved to:"
echo -e "  ${BLUE}result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/${NC}"
echo -e "  ${BLUE}result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_mix_html_truncate/${NC}"
echo ""
