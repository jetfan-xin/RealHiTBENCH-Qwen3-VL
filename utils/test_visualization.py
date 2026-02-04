#!/usr/bin/env python3
"""
快速测试脚本：验证可视化notebook的基本功能

可以单独运行来测试图表生成功能，不需要Jupyter环境
"""

import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置路径
PROJECT_ROOT = Path('/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL')
RESULT_DIR = PROJECT_ROOT / 'result' / 'complied'
MODEL_DIR = RESULT_DIR / 'qwen3vl_default_pic'

# 检查项目结构
print("="*80)
print("RealHiTBench 可视化快速测试")
print("="*80)

print(f"\n[1] 检查项目结构...")
print(f"  项目根目录: {PROJECT_ROOT}")
if PROJECT_ROOT.exists():
    print(f"  ✓ 项目目录存在")
else:
    print(f"  ✗ 项目目录不存在")
    sys.exit(1)

# 检查数据目录
print(f"\n[2] 检查结果数据...")
SUPPORTED_MODALITIES = ['mix_html', 'mix_json', 'mix_latex', 'mix_markdown', 'image']

modality_status = {}
for modality in SUPPORTED_MODALITIES:
    modality_path = MODEL_DIR / modality
    results_file = modality_path / 'results.json'
    
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results_count = len(data.get('results', []))
            aggregate = data.get('aggregate_metrics', {})
            
            modality_status[modality] = 'success'
            print(f"  ✓ {modality}: {results_count} 个样本")
            
            # 验证aggregate_metrics
            if 'by_question_type' in aggregate:
                question_types = list(aggregate['by_question_type'].keys())
                print(f"    - 问题类型: {', '.join(question_types)}")
            
        except json.JSONDecodeError as e:
            modality_status[modality] = 'error'
            print(f"  ✗ {modality}: JSON 解析错误 - {e}")
        except Exception as e:
            modality_status[modality] = 'error'
            print(f"  ✗ {modality}: 读取失败 - {e}")
    else:
        modality_status[modality] = 'missing'
        print(f"  ⚠️  {modality}: 文件不存在 ({results_file})")

# 统计
success_count = sum(1 for s in modality_status.values() if s == 'success')
print(f"\n[3] 数据检查总结")
print(f"  成功: {success_count}/{len(SUPPORTED_MODALITIES)}")
print(f"  失败: {sum(1 for s in modality_status.values() if s == 'error')}")
print(f"  缺失: {sum(1 for s in modality_status.values() if s == 'missing')}")

# 如果有数据，进行详细分析
if success_count > 0:
    print(f"\n[4] 数据内容验证...")
    
    # 选择第一个成功的模态进行详细验证
    for modality, status in modality_status.items():
        if status == 'success':
            results_file = MODEL_DIR / modality / 'results.json'
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为DataFrame
            results_list = data.get('results', [])
            df = pd.json_normalize(results_list)
            
            print(f"  详细分析 {modality}:")
            print(f"    - 总样本数: {len(df)}")
            print(f"    - 列数: {len(df.columns)}")
            print(f"    - 主要列: {', '.join(df.columns[:5])}...")
            
            # 检查QuestionType
            if 'QuestionType' in df.columns:
                question_types = df['QuestionType'].unique()
                print(f"    - 问题类型: {list(question_types)}")
                
                # 验证指标列
                task_metrics = {
                    'Fact Checking': ['F1', 'EM'],
                    'Numerical Reasoning': ['F1', 'EM'],
                    'Structure Comprehending': ['F1', 'EM'],
                    'Data Analysis': ['ROUGE-L', 'F1', 'EM'],
                    'Visualization': ['ECR', 'Pass']
                }
                
                print(f"    - 指标验证:")
                for task, metrics in task_metrics.items():
                    task_data = df[df['QuestionType'] == task]
                    if len(task_data) > 0:
                        available_metrics = [m for m in metrics if m in df.columns]
                        print(f"      • {task}: {len(task_data)} 样本, 指标: {available_metrics}")
            
            break

# 检查utils中的函数
print(f"\n[5] 检查utils中的函数...")
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.visualize_aggregate_metrics import load_aggregate_metrics, plot_heatmap, plot_metric_bars
    print(f"  ✓ visualize_aggregate_metrics 模块加载成功")
except ImportError as e:
    print(f"  ⚠️  visualize_aggregate_metrics 模块加载失败: {e}")

# 最后的建议
print(f"\n[6] 建议...")
print(f"  1. 使用Jupyter运行 notebook: utils/result_visualization.ipynb")
print(f"  2. 或使用以下命令转换为Python脚本:")
print(f"     jupyter nbconvert --to script utils/result_visualization.ipynb")
print(f"  3. 图表将保存到:")
for modality in SUPPORTED_MODALITIES:
    if modality_status.get(modality) == 'success':
        output_dir = MODEL_DIR / modality
        print(f"     - {output_dir}")

print(f"\n{'='*80}")
print("✓ 快速测试完成！")
print(f"{'='*80}\n")
