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
#   ./run_default_config_experiments_rz.sh                        # 顺序运行所有模态
#   ./run_default_config_experiments_rz.sh image                  # 只运行 image 模态
#   ./run_default_config_experiments_rz.sh mix_html               # 只运行 mix_html 模态
#   ./run_default_config_experiments_rz.sh text_json 2            # 在 GPU 2 上运行 text_json
#   ./run_default_config_experiments_rz.sh text_json 0,1,2        # 多GPU并行: 在GPU 0,1,2 上各跑一个shard
#   ./run_default_config_experiments_rz.sh text_json 0,1,2,3      # 多GPU并行: 在GPU 0,1,2,3 上各跑一个shard
#   cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference && MAX_QUERIES=6 bash -x ./run_default_config_experiments_rz.sh text_json 0,1,2 2>&1 | grep -A 5 "COMMON_ARGS\|shard_id" # 用 6 个数据点测试多GPU并行（每个GPU处理2个） 

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
# 测试模式: 设置 MAX_QUERIES 限制处理的数据数量 (设为 -1 表示处理全部)
MAX_QUERIES=${MAX_QUERIES:--1}

echo "MAX_QUERIES = $MAX_QUERIES"

COMMON_ARGS="--model_dir $MODEL_DIR \
    --data_path $DATA_PATH \
    --qa_path $QA_PATH \
    --use_sc_filled \
    --batch_size 1 \
    --use_model_parallel \
    --resume \
    --max_queries $MAX_QUERIES"

echo "COMMON_ARGS = $COMMON_ARGS"

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

# ==================== 多GPU并行函数 ====================
# 在多个GPU上同时运行同一模态的不同shards

run_experiment_parallel() {
    local modality=$1
    local format=$2
    local task_name=$3
    local gpu_list=$4  # 逗号分隔的GPU列表，如 "0,1,2"
    
    # 将GPU列表转换为数组
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_list"
    local num_shards=${#GPU_ARRAY[@]}
    
    echo "=========================================="
    echo "Starting PARALLEL: $task_name"
    echo "  Modality: $modality"
    echo "  Format: $format"
    echo "  GPUs: $gpu_list ($num_shards shards)"
    echo "  Time: $(date)"
    echo "=========================================="
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local pids=()
    
    # 为每个GPU启动一个shard
    for shard_id in $(seq 0 $((num_shards - 1))); do
        local gpu_id=${GPU_ARRAY[$shard_id]}
        local log_file="${LOG_DIR}/${task_name}_shard${shard_id}_gpu${gpu_id}_${timestamp}.log"
        
        echo "  Launching shard $shard_id on GPU $gpu_id..."
        echo "    Log: $log_file"
        
        if [ "$modality" == "image" ]; then
            CUDA_VISIBLE_DEVICES=$gpu_id nohup python "$INFERENCE_SCRIPT" \
                $COMMON_ARGS \
                --modality image \
                --num_shards $num_shards \
                --shard_id $shard_id \
                > "$log_file" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES=$gpu_id nohup python "$INFERENCE_SCRIPT" \
                $COMMON_ARGS \
                --modality $modality \
                --format $format \
                --num_shards $num_shards \
                --shard_id $shard_id \
                > "$log_file" 2>&1 &
        fi
        
        pids+=($!)
        echo "    PID: ${pids[-1]}"
        sleep 2  # 错开启动避免竞争条件
    done
    
    echo ""
    echo "All $num_shards shards launched!"
    echo ""
    echo "Monitor logs with:"
    echo "  tail -f ${LOG_DIR}/${task_name}_shard*_${timestamp}.log"
    echo ""
    echo "Check GPU usage with:"
    echo "  watch -n 1 nvidia-smi"
    echo ""
    
    # 自动等待所有进程完成
    echo "Waiting for all shards to complete..."
    local all_success=true
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "[✓] Shard $i (PID ${pids[$i]}) completed successfully"
        else
            echo "[✗] Shard $i (PID ${pids[$i]}) FAILED with exit code $exit_code"
            all_success=false
        fi
    done
    
    if $all_success; then
        echo ""
        echo "[✓] All shards completed successfully!"
        echo ""
        echo "To merge results, run:"
        echo "  python merge_shards_a100.py --modality $modality --format $format --num_shards $num_shards"
    else
        echo ""
        echo "[!] Some shards failed. Check logs for details."
    fi
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
    # 检测是否为多GPU模式（第二个参数包含逗号）
    is_parallel=false
    if [ $# -ge 2 ] && [[ "$2" == *","* ]]; then
        is_parallel=true
    fi
    
    case "$1" in
        image)
            if $is_parallel; then
                run_experiment_parallel "image" "" "image" "$2"
            else
                run_experiment "image" "" "image" "${2:-0}"
            fi
            ;;
        mix_html)
            if $is_parallel; then
                run_experiment_parallel "mix" "html" "mix_html" "$2"
            else
                run_experiment "mix" "html" "mix_html" "${2:-0}"
            fi
            ;;
        mix_latex)
            if $is_parallel; then
                run_experiment_parallel "mix" "latex" "mix_latex" "$2"
            else
                run_experiment "mix" "latex" "mix_latex" "${2:-0}"
            fi
            ;;
        mix_csv)
            if $is_parallel; then
                run_experiment_parallel "mix" "csv" "mix_csv" "$2"
            else
                run_experiment "mix" "csv" "mix_csv" "${2:-0}"
            fi
            ;;
        mix_markdown)
            if $is_parallel; then
                run_experiment_parallel "mix" "markdown" "mix_markdown" "$2"
            else
                run_experiment "mix" "markdown" "mix_markdown" "${2:-0}"
            fi
            ;;
        mix_json)
            if $is_parallel; then
                run_experiment_parallel "mix" "json" "mix_json" "$2"
            else
                run_experiment "mix" "json" "mix_json" "${2:-0}"
            fi
            ;;
        text_html)
            if $is_parallel; then
                run_experiment_parallel "text" "html" "text_html" "$2"
            else
                run_experiment "text" "html" "text_html" "${2:-0}"
            fi
            ;;
        text_latex)
            if $is_parallel; then
                run_experiment_parallel "text" "latex" "text_latex" "$2"
            else
                run_experiment "text" "latex" "text_latex" "${2:-0}"
            fi
            ;;
        text_csv)
            if $is_parallel; then
                run_experiment_parallel "text" "csv" "text_csv" "$2"
            else
                run_experiment "text" "csv" "text_csv" "${2:-0}"
            fi
            ;;
        text_markdown)
            if $is_parallel; then
                run_experiment_parallel "text" "markdown" "text_markdown" "$2"
            else
                run_experiment "text" "markdown" "text_markdown" "${2:-0}"
            fi
            ;;
        text_json)
            if $is_parallel; then
                run_experiment_parallel "text" "json" "text_json" "$2"
            else
                run_experiment "text" "json" "text_json" "${2:-0}"
            fi
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
            echo "Usage: $0 [modality] [gpu_id | gpu_list]"
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
            echo "Single GPU Examples:"
            echo "  $0                     # Run all modalities sequentially"
            echo "  $0 image               # Run image modality only"
            echo "  $0 mix_html 1          # Run mix_html on GPU 1"
            echo "  $0 text_json 2         # Run text_json on GPU 2"
            echo ""
            echo "Multi-GPU Parallel Examples (use comma-separated GPU list):"
            echo "  $0 text_json 0,1,2     # Run text_json on 3 GPUs in parallel (3 shards)"
            echo "  $0 text_json 0,1,2,3   # Run text_json on 4 GPUs in parallel (4 shards)"
            echo "  $0 image 0,1           # Run image on 2 GPUs in parallel (2 shards)"
            echo "  $0 mix_html 0,1,2,3    # Run mix_html on 4 GPUs in parallel"
            echo ""
            echo "For 3071 items with 3 GPUs: each shard processes ~1024 items"
            echo "After parallel run completes, merge results with:"
            echo "  python merge_shards_a100.py --modality <modality> --format <format> --num_shards <N>"
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
