# Shard 结果合并工具使用指南

## 概述

`merge_shards_by_modality.py` 脚本用于合并指定模态的多个 shard 的推理结果。

## 功能特性

- ✅ 自动查找每个 shard 中的最新 `results_*.json` 文件
- ✅ 合并所有 shard 的结果为单一文件
- ✅ 计算按任务类型（QuestionType）分组的聚合指标（平均值）
- ✅ 支持多种模态（image、text_json、text_csv、mix_html、mix_latex 等）
- ✅ 生成时间戳的合并结果，避免覆盖
- ✅ 详细的合并过程日志输出

## 使用方法

### 基本用法

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/utils

# 合并 text_json 模态的 3 个 shard
python merge_shards_by_modality.py --result_dir ../result/qwen3vl_local_a100_default --modality text_json --num_shards 3

# 合并 image 模态的 3 个 shard
python merge_shards_by_modality.py --result_dir ../result/qwen3vl_local_a100_default --modality image --num_shards 3

# 合并 mix_html 模态的 4 个 shard
python merge_shards_by_modality.py --result_dir ../result/qwen3vl_local_a100_default --modality mix_html --num_shards 4
```

### 参数说明

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--result_dir` | 结果基目录（包含各 shard 目录的目录） | `../result/qwen3vl_local_a100_default` | ✗ |
| `--model_name` | 模型名称（用于目录命名） | `Qwen3-VL-8B-Instruct` | ✗ |
| `--modality` | 模态后缀（e.g., text_json、mix_html、image） | - | ✓ |
| `--num_shards` | shard 数量 | 3 | ✗ |

## 输出说明

### 输出目录
```
<result_dir>/<model_name>_<modality>_default_merged/
```

例如：
```
/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_text_json_default_merged/
```

### 输出文件
合并后的结果保存为 `results_merged_YYYYMMDD_HHMMSS.json`

文件结构：
```json
{
  "config": {
    "model_name": "Qwen3-VL-8B-Instruct",
    "modality": "text_json",
    "num_shards": 3,
    "merged_timestamp": "2026-02-03T01:44:26.123456",
    "shard_info": {
      "0": {
        "file": "results_20260202_204841.json",
        "count": 1019,
        "path": "..."
      },
      ...
    }
  },
  "aggregate_metrics": {
    "Data Analysis": {
      "EM": 11.0727,
      "F1": 38.7652,
      "ProcessingTime": 5.9447,
      "ROUGE-L": 38.8575,
      "SacreBLEU": 15.8675
    },
    ...
  },
  "results": [
    {
      "id": "...",
      "FileName": "...",
      "QuestionType": "...",
      "Question": "...",
      "Reference": "...",
      "Prediction": "...",
      "Metrics": {...},
      "ProcessingTime": ...
    },
    ...
  ]
}
```

## 支持的模态

| 模态 | 说明 |
|------|------|
| `image` | 纯图像模态 |
| `text_json` | 仅 JSON 表格（无图像） |
| `text_csv` | 仅 CSV 表格（无图像） |
| `text_html` | 仅 HTML 表格（无图像） |
| `text_latex` | 仅 LaTeX 表格（无图像） |
| `text_markdown` | 仅 Markdown 表格（无图像） |
| `mix_json` | 图像 + JSON 表格 |
| `mix_csv` | 图像 + CSV 表格 |
| `mix_html` | 图像 + HTML 表格 |
| `mix_latex` | 图像 + LaTeX 表格 |
| `mix_markdown` | 图像 + Markdown 表格 |

## 实际工作流例子

### 场景：使用多GPU并行运行后合并结果

1. **运行多GPU并行推理**
   ```bash
   cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference
   ./run_default_config_experiments_rz.sh text_json 0,1,2
   ```

2. **等待所有 shard 完成**
   监控日志：
   ```bash
   tail -f ../result/qwen3vl_local_a100_default/logs/text_json_shard*
   ```

3. **合并 shard 结果**
   ```bash
   cd ../utils
   python merge_shards_by_modality.py --result_dir ../result/qwen3vl_local_a100_default --modality text_json --num_shards 3
   ```

4. **查看合并结果**
   ```bash
   # 查看合并统计
   python3 -c "
   import json
   with open('../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_text_json_default_merged/results_merged_*.json') as f:
       d = json.load(f)
       print(f'Total results: {len(d[\"results\"])}')
       print(f'Metrics by task type:')
       for task, metrics in d['aggregate_metrics'].items():
           print(f'  {task}: {metrics}')
   "
   ```

## 常见问题

### Q: 脚本找不到 shard 目录怎么办？
A: 检查以下几点：
- Shard 目录名称格式是否正确（应为 `{model_name}_{modality}_default_shard{id}`）
- `--result_dir` 路径是否正确
- Shard 目录是否存在且有读权限

### Q: 合并后没有找到 results_*.json 文件？
A: 检查：
- 推理任务是否真的完成
- Shard 目录中是否有 `results_*.json` 文件
- 文件是否完整（没有被截断）

### Q: 如何只合并某些 shard？
A: 目前脚本会自动找到所有 shard。如果要跳过某个 shard，可以：
1. 临时重命名 shard 目录
2. 修改 `--num_shards` 参数（减少数量）
3. 手动编辑脚本

### Q: 合并的结果包含重复数据吗？
A: 不会。脚本只是简单地合并结果数据，不进行去重。如果需要去重，请检查原始的推理结果是否有重复。

## 性能提示

- 对于大规模结果（>100K 项目），合并过程可能需要几分钟
- 输出文件大小约为所有 shard 结果文件大小之和
- 推荐在后台运行：`nohup python merge_shards_by_modality.py ... > merge.log 2>&1 &`

## 扩展功能

可以修改脚本以支持以下功能：
- 自动检测 shard 数量（无需指定 `--num_shards`）
- 支持增量合并（只合并新的 shard）
- 支持结果去重
- 生成HTML格式的合并报告

如需要这些功能，请联系开发者。
