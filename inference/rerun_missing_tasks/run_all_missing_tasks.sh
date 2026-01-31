#!/bin/bash
# 主脚本：运行所有缺失任务的推理
# 生成时间: 2026-01-31T14:22:58.777896
# 总共需要运行: 9 个配置

set -e  # 遇到错误时退出

# GPU配置
# 使用方法：
#   默认（使用GPU 0）: bash run_all_missing_tasks.sh
#   指定单个GPU:      bash run_all_missing_tasks.sh 2
#   指定多个GPU顺序:  bash run_all_missing_tasks.sh 0,1,2,3
GPU_DEVICES=${1:-0}  # 默认使用GPU 0
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

echo "========================================================================"
echo "运行所有缺失任务的推理"
echo "========================================================================"
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "总配置数: 9"
echo "总任务数: 89"
echo ""


echo "------------------------------------------------------------------------"
echo "[1/9] 运行: qwen3vl_default_pic/image"
echo "  任务数: 1 (incomplete: 1, error: 0)"
echo "  使用truncate: False"
echo "------------------------------------------------------------------------"

python inference/rerun_missing_tasks/rerun_qwen3vl_default_pic_image_None.py

if [ $? -ne 0 ]; then
    echo "✗ 失败: qwen3vl_default_pic/image"
    exit 1
fi

echo "✓ 完成: qwen3vl_default_pic/image"
echo ""


echo "------------------------------------------------------------------------"
echo "[2/9] 运行: qwen3vl_default_pic/mix_html"
echo "  任务数: 17 (incomplete: 0, error: 17)"
echo "  使用truncate: True"
echo "------------------------------------------------------------------------"

python inference/rerun_missing_tasks/rerun_qwen3vl_default_pic_mix_html.py

if [ $? -ne 0 ]; then
    echo "✗ 失败: qwen3vl_default_pic/mix_html"
    exit 1
fi

echo "✓ 完成: qwen3vl_default_pic/mix_html"
echo ""


echo "------------------------------------------------------------------------"
echo "[3/9] 运行: qwen3vl_resize_pic/image"
echo "  任务数: 1 (incomplete: 1, error: 0)"
echo "  使用truncate: False"
echo "------------------------------------------------------------------------"

python inference/rerun_missing_tasks/rerun_qwen3vl_resize_pic_image_None.py

if [ $? -ne 0 ]; then
    echo "✗ 失败: qwen3vl_resize_pic/image"
    exit 1
fi

echo "✓ 完成: qwen3vl_resize_pic/image"
echo ""


echo "------------------------------------------------------------------------"
echo "[4/9] 运行: qwen3vl_resize_pic/mix_csv"
echo "  任务数: 11 (incomplete: 1, error: 10)"
echo "  使用truncate: True"
echo "------------------------------------------------------------------------"

python inference/rerun_missing_tasks/rerun_qwen3vl_resize_pic_mix_csv.py

if [ $? -ne 0 ]; then
    echo "✗ 失败: qwen3vl_resize_pic/mix_csv"
    exit 1
fi

echo "✓ 完成: qwen3vl_resize_pic/mix_csv"
echo ""


echo "------------------------------------------------------------------------"
echo "[5/9] 运行: qwen3vl_resize_pic/mix_html"
echo "  任务数: 25 (incomplete: 25, error: 0)"
echo "  使用truncate: True"
echo "------------------------------------------------------------------------"

python inference/rerun_missing_tasks/rerun_qwen3vl_resize_pic_mix_html.py

if [ $? -ne 0 ]; then
    echo "✗ 失败: qwen3vl_resize_pic/mix_html"
    exit 1
fi

echo "✓ 完成: qwen3vl_resize_pic/mix_html"
echo ""


echo "------------------------------------------------------------------------"
echo "[6/9] 运行: qwen3vl_resize_pic/mix_latex"
echo "  任务数: 1 (incomplete: 1, error: 0)"
echo "  使用truncate: True"
echo "------------------------------------------------------------------------"

python inference/rerun_missing_tasks/rerun_qwen3vl_resize_pic_mix_latex.py

if [ $? -ne 0 ]; then
    echo "✗ 失败: qwen3vl_resize_pic/mix_latex"
    exit 1
fi

echo "✓ 完成: qwen3vl_resize_pic/mix_latex"
echo ""


echo "------------------------------------------------------------------------"
echo "[7/9] 运行: qwen3vl_text/text_csv"
echo "  任务数: 15 (incomplete: 15, error: 0)"
echo "  使用truncate: True"
echo "------------------------------------------------------------------------"

python inference/rerun_missing_tasks/rerun_qwen3vl_text_text_csv.py

if [ $? -ne 0 ]; then
    echo "✗ 失败: qwen3vl_text/text_csv"
    exit 1
fi

echo "✓ 完成: qwen3vl_text/text_csv"
echo ""


echo "------------------------------------------------------------------------"
echo "[8/9] 运行: qwen3vl_text/text_html"
echo "  任务数: 17 (incomplete: 0, error: 17)"
echo "  使用truncate: True"
echo "------------------------------------------------------------------------"

python inference/rerun_missing_tasks/rerun_qwen3vl_text_text_html.py

if [ $? -ne 0 ]; then
    echo "✗ 失败: qwen3vl_text/text_html"
    exit 1
fi

echo "✓ 完成: qwen3vl_text/text_html"
echo ""


echo "------------------------------------------------------------------------"
echo "[9/9] 运行: qwen3vl_text/text_latex"
echo "  任务数: 1 (incomplete: 1, error: 0)"
echo "  使用truncate: True"
echo "------------------------------------------------------------------------"

python inference/rerun_missing_tasks/rerun_qwen3vl_text_text_latex.py

if [ $? -ne 0 ]; then
    echo "✗ 失败: qwen3vl_text/text_latex"
    exit 1
fi

echo "✓ 完成: qwen3vl_text/text_latex"
echo ""


echo "========================================================================"
echo "✅ 所有缺失任务推理完成！"
echo "========================================================================"
