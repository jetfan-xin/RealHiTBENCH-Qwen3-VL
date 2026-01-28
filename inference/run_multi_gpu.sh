#!/bin/bash
# Multi-GPU parallel inference script for RealHiTBench
# Each GPU runs an independent process with its own shard of data

# Activate conda environment
source /ltstorage/home/4xin/miniconda3/etc/profile.d/conda.sh
conda activate HIT

# Verify Python environment
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Configuration
MODEL_DIR="/mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct"
DATA_PATH="/mnt/data1/users/4xin/RealHiTBench"
QA_PATH="/ltstorage/home/4xin/image_table/RealHiTBench/data"
SKIP_CHECKPOINT="/ltstorage/home/4xin/image_table/RealHiTBench/result/qwen3vl_local/Qwen3-VL-8B-Instruct_image/checkpoint_batch.json"

# Number of GPUs/shards
NUM_SHARDS=3

# Common arguments
COMMON_ARGS="--model_dir $MODEL_DIR \
    --data_path $DATA_PATH \
    --qa_path $QA_PATH \
    --modality image \
    --batch_size 1 \
    --use_flash_attn \
    --use_model_parallel \
    --resume \
    --num_shards $NUM_SHARDS"

# Add skip checkpoint if it exists
if [ -f "$SKIP_CHECKPOINT" ]; then
    COMMON_ARGS="$COMMON_ARGS --skip_checkpoint $SKIP_CHECKPOINT"
    echo "Will skip IDs from checkpoint: $SKIP_CHECKPOINT"
fi

echo "Starting multi-GPU inference with $NUM_SHARDS shards..."
echo "Model: $MODEL_DIR"
echo "Data: $DATA_PATH"
echo ""

# Create log directory
LOG_DIR="../result/qwen3vl_local/logs"
mkdir -p $LOG_DIR

# Launch each shard on a separate GPU
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    LOG_FILE="$LOG_DIR/shard_${SHARD_ID}_$(date +%Y%m%d_%H%M%S).log"
    echo "Launching shard $SHARD_ID on GPU $SHARD_ID..."
    echo "  Log: $LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=$SHARD_ID nohup python inference_qwen3vl_local.py \
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
