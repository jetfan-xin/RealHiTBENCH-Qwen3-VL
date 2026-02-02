#!/bin/bash
#
# 使用官方默认 processor 配置运行实验
# 模态: image, mix_html, mix_latex, mix_csv, mix_markdown, mix_json,
#       text_html, text_latex, text_csv, text_markdown, text_json
# 数据: QA_final_sc_filled.json (SC-filled)
#
# ⚠️ 警告: 官方默认配置允许高达 ~16.8M 像素的图像
#          可能导致 OOM，特别是大表格图像
#
# 使用方法:
#   ./run_default_config_experiments.sh           # 顺序运行所有模态
#   ./run_default_config_experiments.sh image     # 只运行 image 模态
#   ./run_default_config_experiments.sh mix_html  # 只运行 mix_html 模态
#   ./run_default_config_experiments.sh text_json 2  # 在 GPU 2 上运行 text_json
#

set -e

# ==================== 配置 ====================
MODEL_DIR="/mnt/data2/projects/pan/4xin/models/Qwen3-VL-8B-Instruct"
DATA_PATH="/mnt/data2/projects/pan/4xin/datasets/RealHiTBench"
QA_PATH="/ltstorage/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data"
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

# ==================== 函数定义 ====================

run_experiment() {
    local modality=$1
    local format=$2
    local task_name=$3
    local gpu_id=${4:-0}
    
    echo "=========================================="
    echo "Starting: $task_name"
    echo "  Modality: $modality"
    echo "  Format: $format"
    echo "  GPU: $gpu_id"
    echo "  Time: $(date)"
    echo "=========================================="
    
    # 使用纳秒级时间戳确保每次运行都有不同的日志文件名
    local timestamp=$(date +%Y%m%d_%H%M%S_%N | cut -c1-20)  # YYYYMMDD_HHMMSS_nanoseconds (truncate to 20 chars)
    local log_file="${LOG_DIR}/${task_name}_${timestamp}.log"
    
    if [ "$modality" == "image" ]; then
        CUDA_VISIBLE_DEVICES=$gpu_id python "$INFERENCE_SCRIPT" \
            $COMMON_ARGS \
            --modality image \
            2>&1 | tee "$log_file"
    else
        CUDA_VISIBLE_DEVICES=$gpu_id python "$INFERENCE_SCRIPT" \
            $COMMON_ARGS \
            --modality $modality \
            --format $format \
            2>&1 | tee "$log_file"
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[✓] $task_name completed successfully"
    else
        echo "[✗] $task_name FAILED with exit code $exit_code"
    fi
    
    return $exit_code
}

# ==================== 主程序 ====================

echo "============================================================"
echo "Official Default Config Experiments (SC-filled)"
echo "============================================================"
echo "Model: $MODEL_DIR"
echo "Data: $DATA_PATH"
echo "QA: $QA_PATH/QA_final_sc_filled.json"
echo "Log dir: $LOG_DIR"
echo ""
echo "⚠️  WARNING: Using official default processor settings"
echo "    This allows up to ~16.8M pixels per image"
echo "    May cause OOM on large table images!"
echo "============================================================"
echo ""

# 如果指定了特定模态，只运行那个模态
if [ $# -ge 1 ]; then
    case "$1" in
        image)
            run_experiment "image" "" "image" "${2:-0}"
            ;;
        mix_html)
            run_experiment "mix" "html" "mix_html" "${2:-0}"
            ;;
        mix_latex)
            run_experiment "mix" "latex" "mix_latex" "${2:-0}"
            ;;
        mix_csv)
            run_experiment "mix" "csv" "mix_csv" "${2:-0}"
            ;;
        mix_markdown)
            run_experiment "mix" "markdown" "mix_markdown" "${2:-0}"
            ;;
        mix_json)
            run_experiment "mix" "json" "mix_json" "${2:-0}"
            ;;
        text_html)
            run_experiment "text" "html" "text_html" "${2:-0}"
            ;;
        text_latex)
            run_experiment "text" "latex" "text_latex" "${2:-0}"
            ;;
        text_csv)
            run_experiment "text" "csv" "text_csv" "${2:-0}"
            ;;
        text_markdown)
            run_experiment "text" "markdown" "text_markdown" "${2:-0}"
            ;;
        text_json)
            run_experiment "text" "json" "text_json" "${2:-0}"
            ;;
        all)
            # 运行所有模态
            run_experiment "image" "" "image" "0"
            run_experiment "mix" "html" "mix_html" "0"
            run_experiment "mix" "latex" "mix_latex" "0"
            run_experiment "mix" "csv" "mix_csv" "0"
            run_experiment "mix" "markdown" "mix_markdown" "0"
            run_experiment "mix" "json" "mix_json" "0"
            run_experiment "text" "html" "text_html" "0"
            run_experiment "text" "latex" "text_latex" "0"
            run_experiment "text" "csv" "text_csv" "0"
            run_experiment "text" "markdown" "text_markdown" "0"
            run_experiment "text" "json" "text_json" "0"
            ;;
        *)
            echo "Usage: $0 [modality] [gpu_id]"
            echo ""
            echo "Modalities:"
            echo "  image         - Image only"
            echo "  mix_html      - Image + HTML table"
            echo "  mix_latex     - Image + LaTeX table"
            echo "  mix_csv       - Image + CSV table"
            echo "  mix_markdown  - Image + Markdown table"
            echo "  mix_json      - Image + JSON table"
            echo "  text_html     - HTML table only (no image)"
            echo "  text_latex    - LaTeX table only (no image)"
            echo "  text_csv      - CSV table only (no image)"
            echo "  text_markdown - Markdown table only (no image)"
            echo "  text_json     - JSON table only (no image)"
            echo "  all           - Run all modalities"
            echo ""
            echo "Examples:"
            echo "  $0                     # Run all modalities sequentially"
            echo "  $0 image               # Run image modality only"
            echo "  $0 mix_html 1          # Run mix_html on GPU 1"
            echo "  $0 text_json 2         # Run text_json on GPU 2"
            exit 1
            ;;
    esac
else
    # 默认运行所有模态
    echo "Running all modalities sequentially..."
    echo ""
    
    run_experiment "image" "" "image" "0"
    run_experiment "mix" "html" "mix_html" "0"
    run_experiment "mix" "latex" "mix_latex" "0"
    run_experiment "mix" "csv" "mix_csv" "0"
    run_experiment "mix" "markdown" "mix_markdown" "0"
    run_experiment "mix" "json" "mix_json" "0"
    run_experiment "text" "html" "text_html" "0"
    run_experiment "text" "latex" "text_latex" "0"
    run_experiment "text" "csv" "text_csv" "0"
    run_experiment "text" "markdown" "text_markdown" "0"
    run_experiment "text" "json" "text_json" "0"
fi

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "Results saved to: ../result/qwen3vl_local_a100_default/"
echo "============================================================"
