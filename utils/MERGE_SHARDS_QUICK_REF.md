# Shard 合并 - 快速参考

## 快速开始

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/utils

# 合并指定模态的 shard
python merge_shards_by_modality.py --modality <模态> --num_shards <数量>
```

## 常用命令

```bash
# text_json (3 shards)
python merge_shards_by_modality.py --modality text_json --num_shards 3

# image (3 shards)
python merge_shards_by_modality.py --modality image --num_shards 3

# mix_html (4 shards)
python merge_shards_by_modality.py --modality mix_html --num_shards 4

# 所有 text 模态 (3 shards)
for fmt in json csv html latex markdown; do
  python merge_shards_by_modality.py --modality text_$fmt --num_shards 3
done

# 所有 mix 模态 (3 shards)
for fmt in json csv html latex markdown; do
  python merge_shards_by_modality.py --modality mix_$fmt --num_shards 3
done
```

## 输出位置

合并后的结果位于：
```
../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_<模态>_default_merged/
```

## 支持的模态列表

**纯文本（无图像）：**
- text_json
- text_csv
- text_html
- text_latex
- text_markdown

**纯图像：**
- image

**混合（图像+文本）：**
- mix_json
- mix_csv
- mix_html
- mix_latex
- mix_markdown

## 一键合并所有结果

```bash
#!/bin/bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/utils

# 合并所有模态
python merge_shards_by_modality.py --modality image --num_shards 3
for fmt in json csv html latex markdown; do
  python merge_shards_by_modality.py --modality text_$fmt --num_shards 3
  python merge_shards_by_modality.py --modality mix_$fmt --num_shards 3
done

echo "✓ All shards merged!"
```

## 查看合并结果统计

```bash
python3 << 'EOF'
import json
import glob

# 查找最新的合并文件
result_dir = '../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_text_json_default_merged'
merged_file = sorted(glob.glob(f'{result_dir}/results_merged_*.json'))[-1]

with open(merged_file) as f:
    d = json.load(f)
    
print(f"✓ 合并结果统计")
print(f"  总结果数: {len(d['results'])}")
print(f"  包含 shard 数: {len(d['config']['shard_info'])}")
print(f"\n  各任务类型性能指标:")
for task, metrics in d['aggregate_metrics'].items():
    print(f"\n  {task}:")
    for metric, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"    {metric}: {value:.4f}")
EOF
```

## 故障排查

| 问题 | 解决方案 |
|------|--------|
| 找不到 shard 目录 | 检查 `--result_dir` 路径，默认为 `../result/qwen3vl_local_a100_default` |
| 没有 results_*.json | 检查推理任务是否完成，是否在 shard 目录中生成结果文件 |
| 合并速度慢 | 正常现象，大文件合并可能需要几分钟，可使用 `nohup` 后台运行 |
| 输出文件损坏 | 检查磁盘空间和权限，确保有足够空间写入合并结果 |

## 更多信息

详见 [MERGE_SHARDS_GUIDE.md](MERGE_SHARDS_GUIDE.md)
