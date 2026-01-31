#!/bin/bash
#
# Quick script to copy existing checkpoints and reprocess OOM errors
# This is the RECOMMENDED way to fix OOM samples without re-running everything
#

set -e  # Exit on error

echo "================================================================================"
echo "COPY CHECKPOINTS AND REPROCESS OOM ERRORS"
echo "================================================================================"
echo ""

WORKSPACE="/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL"
cd "$WORKSPACE/inference"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Setup directories
echo -e "${YELLOW}[1/4] Setting up output directories...${NC}"
mkdir -p ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/
mkdir -p ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_mix_html_truncate/
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# 2. Copy text_html checkpoint
echo -e "${YELLOW}[2/4] Copying text_html checkpoint...${NC}"
SRC_TEXT="../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_html_a100/checkpoint.json"
DST_TEXT="../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/checkpoint.json"

if [ -f "$SRC_TEXT" ]; then
    cp "$SRC_TEXT" "$DST_TEXT"
    echo -e "${GREEN}✓ Copied: $SRC_TEXT${NC}"
    
    # Count OOM errors
    OOM_COUNT=$(grep -o '\[ERROR\] OOM:' "$DST_TEXT" | wc -l)
    echo -e "  Found ${YELLOW}$OOM_COUNT OOM errors${NC} to reprocess"
else
    echo -e "${YELLOW}⚠ Source checkpoint not found: $SRC_TEXT${NC}"
fi
echo ""

# 3. Copy mix_html checkpoint
echo -e "${YELLOW}[3/4] Copying mix_html checkpoint...${NC}"
SRC_MIX="../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_mix_html_default/checkpoint.json"
DST_MIX="../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_mix_html_truncate/checkpoint.json"

if [ -f "$SRC_MIX" ]; then
    cp "$SRC_MIX" "$DST_MIX"
    echo -e "${GREEN}✓ Copied: $SRC_MIX${NC}"
    
    # Count OOM errors
    OOM_COUNT=$(grep -o '\[ERROR\]' "$DST_MIX" | wc -l)
    echo -e "  Found ${YELLOW}$OOM_COUNT ERROR samples${NC} to reprocess"
else
    echo -e "${YELLOW}⚠ Source checkpoint not found: $SRC_MIX${NC}"
fi
echo ""

# 4. Instructions
echo "================================================================================"
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo "================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Reprocess text_html OOM errors (recommended first):"
echo "     cd $WORKSPACE/inference"
echo "     python run_text_html_truncate.py"
echo ""
echo "  2. Reprocess mix_html OOM errors:"
echo "     python run_mix_html_truncate.py"
echo ""
echo "Expected behavior:"
echo "  - Script will load existing checkpoint"
echo "  - Skip 3043+ already successful samples"
echo "  - Reprocess only the 17 OOM/ERROR samples with text truncation"
echo "  - Save results to: result/qwen3vl_local_a100_truncate/"
echo ""
echo "Estimated time: ~30 minutes (vs ~15 hours for full re-run)"
echo "================================================================================"
