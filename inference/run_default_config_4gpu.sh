#!/bin/bash
#
# 使用官方默认 processor 配置，4 GPU 并行运行 5 个模态的实验
# 模态: image, mix_html, mix_latex, mix_csv, mix_markdown
# 数据: QA_final_sc_filled.json (SC-filled)
#
# GPU 分配:
#   GPU 0: image
#   GPU 1: mix_html
#   GPU 2: mix_latex
#   GPU 3: mix_csv + mix_markdown (顺序)
#
# ⚠️ 警告: 官方默认配置允许高达 ~16.8M 像素的图像
#          可能导致 OOM，特别是大表格图像
#

set -e

# ==================== 配置 ====================
MODEL_DIR="/data/pan/4xin/models/Qwen3-VL-8B-Instruct"
DATA_PATH="/data/pan/4xin/datasets/RealHiTBench"
QA_PATH="/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../result/qwen3vl_local_a100_default/logs"
INFERENCE_SCRIPT="${SCRIPT_DIR}/inference_qwen3vl_local_a100_default.py"

# 创建日志目录
mkdir -p "$LOG_DIR"

# ==================== 通用参数 ====================
COMMON_ARGS="--model_dir $MODEL_DIR \
    --data_path $DATA_PATH \
    --qa_path $QA_PATH \
    --use_sc_filled \
    --batch_size 1 \
    --use_model_parallel \
    --resume"

# ==================== 打印配置 ====================
echo "============================================================"
echo "Official Default Config Experiments - 4 GPU Parallel"
echo "============================================================"
echo "Model: $MODEL_DIR"
echo "Data: $DATA_PATH"
echo "QA: $QA_PATH/QA_final_sc_filled.json"
echo "Log dir: $LOG_DIR"
echo ""
echo "⚠️  WARNING: Using official default processor settings"
echo "    This allows up to ~16.8M pixels per image"
echo "    May cause OOM on large table images!"
echo ""
echo "GPU Allocation:"
echo "  GPU 0: image"
echo "  GPU 1: mix_html"
echo "  GPU 2: mix_latex"
echo "  GPU 3: mix_csv → mix_markdown (sequential)"
echo "============================================================"
echo ""

# ==================== 启动任务 ====================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# GPU 0: image
echo "[GPU 0] Starting: image"
CUDA_VISIBLE_DEVICES=0 nohup python "$INFERENCE_SCRIPT" \
    $COMMON_ARGS \
    --modality image \
    > "${LOG_DIR}/image_gpu0_${TIMESTAMP}.log" 2>&1 &
PID_IMAGE=$!
echo "[GPU 0] image started (PID: $PID_IMAGE)"

# GPU 1: mix_html
echo "[GPU 1] Starting: mix_html"
CUDA_VISIBLE_DEVICES=1 nohup python "$INFERENCE_SCRIPT" \
    $COMMON_ARGS \
    --modality mix \
    --format html \
    > "${LOG_DIR}/mix_html_gpu1_${TIMESTAMP}.log" 2>&1 &
PID_MIX_HTML=$!
echo "[GPU 1] mix_html started (PID: $PID_MIX_HTML)"

# GPU 2: mix_latex
echo "[GPU 2] Starting: mix_latex"
CUDA_VISIBLE_DEVICES=2 nohup python "$INFERENCE_SCRIPT" \
    $COMMON_ARGS \
    --modality mix \
    --format latex \
    > "${LOG_DIR}/mix_latex_gpu2_${TIMESTAMP}.log" 2>&1 &
PID_MIX_LATEX=$!
echo "[GPU 2] mix_latex started (PID: $PID_MIX_LATEX)"

# GPU 3: mix_csv → mix_markdown (sequential)
echo "[GPU 3] Starting: mix_csv (then mix_markdown)"
(
    CUDA_VISIBLE_DEVICES=3 python "$INFERENCE_SCRIPT" \
        $COMMON_ARGS \
        --modality mix \
        --format csv \
        > "${LOG_DIR}/mix_csv_gpu3_${TIMESTAMP}.log" 2>&1
    
    echo "[GPU 3] mix_csv finished, starting mix_markdown"
    
    CUDA_VISIBLE_DEVICES=3 python "$INFERENCE_SCRIPT" \
        $COMMON_ARGS \
        --modality mix \
        --format markdown \
        > "${LOG_DIR}/mix_markdown_gpu3_${TIMESTAMP}.log" 2>&1
) &
PID_GPU3=$!
echo "[GPU 3] mix_csv+mix_markdown started (PID: $PID_GPU3)"

echo ""
echo "============================================================"
echo "All tasks launched!"
echo ""
echo "Process IDs:"
echo "  image (GPU 0):    $PID_IMAGE"
echo "  mix_html (GPU 1): $PID_MIX_HTML"
echo "  mix_latex (GPU 2): $PID_MIX_LATEX"
echo "  GPU 3 tasks:      $PID_GPU3"
echo ""
echo "Monitor with:"
echo "  gpustat -i 1"
echo "  tail -f ${LOG_DIR}/*.log"
echo ""
echo "Check status:"
echo "  ps aux | grep inference_qwen3vl_local_a100_default"
echo "============================================================"
