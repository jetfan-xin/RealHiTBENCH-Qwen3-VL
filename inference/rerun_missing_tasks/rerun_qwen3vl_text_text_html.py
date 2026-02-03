#!/usr/bin/env python3
"""
重新运行缺失的任务: qwen3vl_text/text_html

生成时间: 2026-02-03T02:04:02.377783
任务来源: qwen3vl_text/text_html/results.json
需要重新运行的任务数: 17
  - Incomplete runs: 0
  - Error tasks: 17

# 使用truncate版本防止OOM (error_ids: 17)
"""

import subprocess
import sys
import os
import json

def main():
    # 配置
    modality = "text"
    format_type = "html"
    batch_size = 1  # 使用batch_size=1避免OOM
    model_dir = "/data/pan/4xin/models/Qwen3-VL-8B-Instruct"
    data_path = "/data/pan/4xin/datasets/RealHiTBench"
    qa_path = "/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data"
    
    # 需要重新运行的任务ID
    task_ids = [2747, 2748, 2749, 2750, 2751, 2758, 2759, 2760, 2761, 2762, 2763, 2966, 2967, 2968, 3019, 3020, 3021]
    
    print("=" * 80)
    print(f"重新运行缺失任务: {modality}" + (f"_{format_type}" if format_type else ""))
    print("=" * 80)
    print(f"配置: qwen3vl_text")
    print(f"任务数量: {len(task_ids)}")
    print(f"Inference脚本: inference_qwen3vl_local_a100_truncate_with_task_ids.py")
    print(f"使用文本截断: True")
    print()
    
    # 构建命令
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inference_path = os.path.join(script_dir, "inference_qwen3vl_local_a100_truncate_with_task_ids.py")
    
    cmd = [
        sys.executable,
        inference_path,
        "--modality", modality,
        "--model_dir", model_dir,
        "--data_path", data_path,
        "--qa_path", qa_path,
        "--use_sc_filled",
        "--batch_size", str(batch_size),
        "--task_ids", ",".join(map(str, task_ids)),  # 指定任务ID
        "--resume"  # 使用resume模式，会加载已有checkpoint并合并
    ]
    
    if format_type:
        cmd.extend(["--format", format_type])
    
    print(f"命令: {' '.join(cmd)}")
    print("-" * 80)
    print()
    
    # 运行推理
    result = subprocess.run(cmd, cwd=script_dir)
    
    if result.returncode == 0:
        print()
        print("=" * 80)
        print(f"✓ 成功完成 {modality}" + (f"_{format_type}" if format_type else "") + f" 的 {len(task_ids)} 个缺失任务")
        print("=" * 80)
    else:
        print()
        print("=" * 80)
        print(f"✗ 运行失败，退出码: {result.returncode}")
        print("=" * 80)
        sys.exit(result.returncode)

if __name__ == '__main__':
    main()
