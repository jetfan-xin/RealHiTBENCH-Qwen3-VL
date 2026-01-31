#!/bin/bash
# Parallel launcher for Structure Comprehending re-inference across 9 modalities on 3 GPUs
# GPU allocation (only GPUs 1,2,3):
#   GPU 1: image -> text_latex -> text_html (sequential on same GPU)
#   GPU 2: mix_latex -> mix_csv -> text_csv (sequential on same GPU)
#   GPU 3: mix_html -> mix_markdown -> text_markdown (sequential on same GPU)

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="${SCRIPT_DIR}/../result/qwen3vl_local_a100/sc_rerun_logs"

# Create log directory
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Structure Comprehending Re-inference (3 GPUs)"
echo "Started at: $(date)"
echo "Using GPUs: 1,2,3"
echo "========================================"
echo ""

# Function to run inference on specific GPU
run_on_gpu() {
    local gpu_id=$1
    local script_name=$2
    local modality=$3
    
    echo "[GPU ${gpu_id}] Starting ${modality}..."
    CUDA_VISIBLE_DEVICES=${gpu_id} python "${SCRIPT_DIR}/${script_name}" \
        > "${LOG_DIR}/${modality}_gpu${gpu_id}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[GPU ${gpu_id}] ✓ ${modality} completed successfully"
    else
        echo "[GPU ${gpu_id}] ✗ ${modality} FAILED (see log: ${LOG_DIR}/${modality}_gpu${gpu_id}.log)"
        return 1
    fi
}

# Launch 3 parallel jobs (one per GPU)
echo "Launching parallel inference jobs..."
echo ""

# GPU 1: image -> text_latex -> text_html
(
    run_on_gpu 1 "run_sc_image.py" "image" && \
    run_on_gpu 1 "run_text_latex.py" "text_latex" && \
    run_on_gpu 1 "run_text_html.py" "text_html"
) &
PID_GPU1=$!

# GPU 2: mix_latex -> mix_csv -> text_csv
(
    run_on_gpu 2 "run_sc_mix_latex.py" "mix_latex" && \
    run_on_gpu 2 "run_sc_mix_csv.py" "mix_csv" && \
    run_on_gpu 2 "run_text_csv.py" "text_csv"
) &
PID_GPU2=$!

# GPU 3: mix_html -> mix_markdown -> text_markdown
(
    run_on_gpu 3 "run_sc_mix_html.py" "mix_html" && \
    run_on_gpu 3 "run_sc_mix_markdown.py" "mix_markdown" && \
    run_on_gpu 3 "run_text_markdown.py" "text_markdown"
) &
PID_GPU3=$!

# Wait for jobs to complete
echo "Waiting for parallel jobs to complete..."
echo ""

wait $PID_GPU1
STATUS_GPU1=$?

wait $PID_GPU2
STATUS_GPU2=$?

wait $PID_GPU3
STATUS_GPU3=$?

# Check if any failed
FAILED=0
if [ $STATUS_GPU1 -ne 0 ]; then
    echo "ERROR: GPU1 sequence (image/text_latex/text_html) failed"
    FAILED=1
fi
if [ $STATUS_GPU2 -ne 0 ]; then
    echo "ERROR: GPU2 sequence (mix_latex/mix_csv/text_csv) failed"
    FAILED=1
fi
if [ $STATUS_GPU3 -ne 0 ]; then
    echo "ERROR: GPU3 sequence (mix_html/mix_markdown/text_markdown) failed"
    FAILED=1
fi

if [ $FAILED -eq 1 ]; then
    echo ""
    echo "Some jobs failed. See logs in ${LOG_DIR}/"
    exit 1
fi

echo ""
echo "========================================"
echo "All Structure Comprehending inference completed!"
echo "Finished at: $(date)"
echo "========================================"
echo ""
echo "Logs saved to: ${LOG_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Verify checkpoint files were created in result/qwen3vl_local_a100/"
echo "  2. Run merge script: python inference/merge_sc_results.py"
echo ""

exit 0
