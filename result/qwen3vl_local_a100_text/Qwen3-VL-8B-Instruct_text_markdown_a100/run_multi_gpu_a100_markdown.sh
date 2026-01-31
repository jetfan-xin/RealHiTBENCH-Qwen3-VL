#!/bin/bash
# Multi-GPU parallel inference script for RealHiTBench
# Each GPU runs an independent process with its own shard of data
set -euo pipefail

# Activate conda environment
source /export/home/pan/miniconda3/etc/profile.d/conda.sh
conda activate 4xin-hit

# Verify Python environment
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Configuration
MODEL_DIR="/data/pan/4xin/models/Qwen3-VL-8B-Instruct"
DATA_PATH="/data/pan/4xin/datasets/RealHiTBench"
QA_PATH="/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data"

# Number of GPUs/shards
NUM_SHARDS=4
GPUS=(0 1 2 3)

# GPU waiting parameters
MEM_THRESHOLD=10          # MiB - GPU memory threshold
STABLE_SECONDS=10         # Wait for GPUs to be stable for this many seconds
CHECK_INTERVAL=1          # Check every second
HEARTBEAT_EVERY=60        # Log status every 60 seconds

# Common arguments (从0开始，不使用checkpoint)
COMMON_ARGS="--model_dir $MODEL_DIR \
    --data_path $DATA_PATH \
    --qa_path $QA_PATH \
    --modality text \
    --format markdown \
    --batch_size 1 \
    --use_flash_attn \
    --use_model_parallel \
    --num_shards $NUM_SHARDS"

echo "=================================="
echo "Multi-GPU Inference Configuration"
echo "=================================="
echo "Model: $MODEL_DIR"
echo "Data: $DATA_PATH"
echo "GPUs: ${GPUS[*]}"
echo "Shards: $NUM_SHARDS"
echo "Starting from scratch (no checkpoint)"
echo ""

# Wait for GPUs to be free
echo "[GPU Watcher] Waiting for GPUs ${GPUS[*]} to be free..."
echo "[GPU Watcher] Memory threshold: ${MEM_THRESHOLD} MiB"
echo "[GPU Watcher] Required stable time: ${STABLE_SECONDS}s"
echo ""

stable=0
last_state="unknown"
tick=0

while true; do
    ts=$(date "+%F %T")
    
    # Check if all GPUs are free
    all_free=1
    status_parts=()
    for g in "${GPUS[@]}"; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | head -n1 | xargs)
        status_parts+=("GPU${g}=${used}MiB")
        if [[ "$used" -ge "$MEM_THRESHOLD" ]]; then
            all_free=0
        fi
    done
    
    if [[ "$all_free" -eq 1 ]]; then
        state="all_free"
        stable=$((stable+CHECK_INTERVAL))
    else
        state="not_free"
        stable=0
    fi
    
    # Log state changes
    if [[ "$state" != "$last_state" ]]; then
        echo "[GPU Watcher] ${ts} State: ${state} (${status_parts[*]})"
        last_state="$state"
    fi
    
    # Heartbeat
    tick=$((tick+CHECK_INTERVAL))
    if (( tick % HEARTBEAT_EVERY == 0 )); then
        echo "[GPU Watcher] ${ts} Still waiting... stable=${stable}/${STABLE_SECONDS}s (${status_parts[*]})"
    fi
    
    # Start when stable
    if [[ "$stable" -ge "$STABLE_SECONDS" ]]; then
        echo "[GPU Watcher] ${ts} All GPUs free for ${STABLE_SECONDS}s - Starting inference!"
        echo ""
        break
    fi
    
    sleep "$CHECK_INTERVAL"
done

echo "Starting multi-GPU inference with $NUM_SHARDS shards..."
echo ""

# Create log directory
LOG_DIR="/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/result/qwen3vl_local_a100/logs"
mkdir -p $LOG_DIR

# Launch each shard on a separate GPU
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    LOG_FILE="$LOG_DIR/shard_${SHARD_ID}_$(date +%Y%m%d_%H%M%S).log"
    echo "Launching shard $SHARD_ID on GPU $SHARD_ID..."
    echo "  Log: $LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=$SHARD_ID nohup python /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference/inference_qwen3vl_local_a100.py \
        $COMMON_ARGS \
        --shard_id $SHARD_ID \
        > "$LOG_FILE" 2>&1 &
    
    echo "  PID: $!"
    sleep 2  # Stagger process starts to avoid race conditions
done

echo ""
echo "All shards launched! Monitor with:"
echo "  tail -f $LOG_DIR/shard_*.log"
echo ""
echo "Check GPU usage with:"
echo "  watch -n 1 nvidia-smi"
