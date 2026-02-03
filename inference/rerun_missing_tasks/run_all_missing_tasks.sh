#!/bin/bash
# 主脚本：运行所有缺失任务的推理
# 生成时间: 2026-02-03T02:04:02.378546
# 总共需要运行: 12 个配置

set -e  # 遇到错误时退出

echo "========================================================================"
echo "运行所有缺失任务的推理"
echo "========================================================================"
echo "总配置数: 12"
echo "总任务数: 147"
echo ""


echo "------------------------------------------------------------------------"
echo "[1/12] 运行: qwen3vl_default_pic/image"
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
echo "[2/12] 运行: qwen3vl_default_pic/mix_csv"
echo "  任务数: 2 (incomplete: 0, error: 2)"
echo "  使用truncate: True"
echo "------------------------------------------------------------------------"

python inference/rerun_missing_tasks/rerun_qwen3vl_default_pic_mix_csv.py

if [ $? -ne 0 ]; then
    echo "✗ 失败: qwen3vl_default_pic/mix_csv"
    exit 1
fi

echo "✓ 完成: qwen3vl_default_pic/mix_csv"
echo ""


echo "------------------------------------------------------------------------"
echo "[3/12] 运行: qwen3vl_default_pic/mix_html"
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
echo "[4/12] 运行: qwen3vl_default_pic/mix_markdown"
echo "  任务数: 28 (incomplete: 0, error: 28)"
echo "  使用truncate: True"
echo "------------------------------------------------------------------------"

python inference/rerun_missing_tasks/rerun_qwen3vl_default_pic_mix_markdown.py

if [ $? -ne 0 ]; then
    echo "✗ 失败: qwen3vl_default_pic/mix_markdown"
    exit 1
fi

echo "✓ 完成: qwen3vl_default_pic/mix_markdown"
echo ""


echo "------------------------------------------------------------------------"
echo "[5/12] 运行: qwen3vl_resize_pic/image"
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
echo "[6/12] 运行: qwen3vl_resize_pic/mix_csv"
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
echo "[7/12] 运行: qwen3vl_resize_pic/mix_html"
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
echo "[8/12] 运行: qwen3vl_resize_pic/mix_latex"
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
echo "[9/12] 运行: qwen3vl_text/text_csv"
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
echo "[10/12] 运行: qwen3vl_text/text_html"
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
echo "[11/12] 运行: qwen3vl_text/text_latex"
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


echo "------------------------------------------------------------------------"
echo "[12/12] 运行: qwen3vl_text/text_markdown"
echo "  任务数: 28 (incomplete: 0, error: 28)"
echo "  使用truncate: True"
echo "------------------------------------------------------------------------"

python inference/rerun_missing_tasks/rerun_qwen3vl_text_text_markdown.py

if [ $? -ne 0 ]; then
    echo "✗ 失败: qwen3vl_text/text_markdown"
    exit 1
fi

echo "✓ 完成: qwen3vl_text/text_markdown"
echo ""


echo "========================================================================"
echo "✅ 所有缺失任务推理完成！"
echo "========================================================================"
