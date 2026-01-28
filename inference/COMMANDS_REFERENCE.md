# Qwen3-VL Local Inference 命令参考手册

本文档包含 RealHiTBench 评估的所有运行场景的命令。

---

## 📋 目录

1. [Image-Only 模态](#1-image-only-模态)
2. [Text-Only 模态](#2-text-only-模态)
3. [Mix 模态（Image + Text）](#3-mix-模态image--text)
4. [测试与调试](#4-测试与调试)
5. [批量推理 vs 单任务推理](#5-批量推理-vs-单任务推理)
6. [恢复中断的评估](#6-恢复中断的评估)
7. [多进程分片并行（推荐）](#7-多进程分片并行推荐)
8. [参数说明](#8-参数说明)

---

## 1. Image-Only 模态

### 1.1 完整评估（3071个任务，推荐batch_size=3）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --batch_size 3' \
    > ../result/qwen3vl_local/image_full_batch3.log 2>&1 &

# 查看日志
tail -f ../result/qwen3vl_local/image_full_batch3.log
----
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
    --data_path /data/pan/4xin/datasets/RealHiTBench \
    --qa_path /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data \
    --modality image \
    --batch_size 3' \
    > ../result/qwen3vl_local/image_full_batch3.log 2>&1 &

# 查看日志
tail -f ../result/qwen3vl_local/image_full_batch3.log
```

**预计时间**：~1小时（使用3个GPU，batch_size=3）

### 1.2 完整评估（单GPU，batch_size=1）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python inference_qwen3vl_local.py \
    --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
    --data_path /data/pan/4xin/datasets/RealHiTBench \
    --qa_path /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data \
    --modality image \
    --batch_size 1' \
    > ../result/qwen3vl_local/image_full_single.log 2>&1 &

tail -f ../result/qwen3vl_local/image_full_single.log
```

### 1.3 测试（仅5个任务）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

nohup python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --batch_size 1 \
    --max_queries 5 \
    > ../result/qwen3vl_local/image_test_5.log 2>&1 &

tail -f ../result/qwen3vl_local/image_test_5.log

---
# 在rzgpu1上
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference

nohup python inference_qwen3vl_local.py \
    --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
    --data_path /data/pan/4xin/datasets/RealHiTBench \
    --qa_path /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data \
    --modality image \
    --batch_size 1 \
    --max_queries 5 \
    > ../result/qwen3vl_local/image_test_5.log 2>&1 &

tail -f ../result/qwen3vl_local/image_test_5.log
```

---

## 2. Text-Only 模态

### 2.1 HTML格式（完整评估）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality text \
    --format html \
    --batch_size 3' \
    > ../result/qwen3vl_local/text_html_batch3.log 2>&1 &

tail -f ../result/qwen3vl_local/text_html_batch3.log
```

### 2.2 LaTeX格式（完整评估）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality text \
    --format latex \
    --batch_size 3' \
    > ../result/qwen3vl_local/text_latex_batch3.log 2>&1 &

tail -f ../result/qwen3vl_local/text_latex_batch3.log
```

### 2.3 Markdown格式（完整评估）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality text \
    --format markdown \
    --batch_size 3' \
    > ../result/qwen3vl_local/text_markdown_batch3.log 2>&1 &

tail -f ../result/qwen3vl_local/text_markdown_batch3.log
```

### 2.4 CSV格式（完整评估）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality text \
    --format csv \
    --batch_size 3' \
    > ../result/qwen3vl_local/text_csv_batch3.log 2>&1 &

tail -f ../result/qwen3vl_local/text_csv_batch3.log
```

---

## 3. Mix 模态（Image + Text）

### 3.1 Mix + LaTeX（完整评估）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality mix \
    --format latex \
    --batch_size 3' \
    > ../result/qwen3vl_local/mix_latex_batch3.log 2>&1 &

tail -f ../result/qwen3vl_local/mix_latex_batch3.log
```

### 3.2 Mix + HTML（完整评估）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality mix \
    --format html \
    --batch_size 3' \
    > ../result/qwen3vl_local/mix_html_batch3.log 2>&1 &

tail -f ../result/qwen3vl_local/mix_html_batch3.log
```

---

## 4. 测试与调试

### 4.1 快速测试（5个任务，验证配置）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

# Image测试
python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --max_queries 5

# Text测试
python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality text \
    --format html \
    --max_queries 5

# Mix测试
python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality mix \
    --format latex \
    --max_queries 5
```

### 4.2 特定问题类型测试

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

# 仅测试Fact Checking任务
python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --question_type "Fact Checking" \
    --max_queries 10
```

可选的问题类型：
- `Fact Checking`
- `Numerical Reasoning`
- `Data Analysis`
- `Visualization`
- `Structure Comprehending`

---

## 5. 批量推理 vs 单任务推理

### 5.1 批量推理（推荐，更快）

```bash
# batch_size=3（推荐，适合3个GPU）
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --batch_size 3' \
    > ../result/qwen3vl_local/image_batch3.log 2>&1 &

# batch_size=5（更激进，需要更多VRAM）
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --batch_size 5' \
    > ../result/qwen3vl_local/image_batch5.log 2>&1 &
```

**优点**：更高的GPU利用率，处理速度更快  
**注意**：需要更多VRAM，batch_size根据GPU内存调整

### 5.2 单任务推理（更稳定）

```bash
# batch_size=1（默认）
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --batch_size 1' \
    > ../result/qwen3vl_local/image_single.log 2>&1 &
```

**优点**：VRAM占用少，更稳定  
**缺点**：速度较慢，每10个任务才保存一次checkpoint

---

## 6. 恢复中断的评估

### 6.1 从checkpoint恢复（自动）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

# 添加 --resume 参数
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --batch_size 3 \
    --resume' \
    > ../result/qwen3vl_local/image_resumed.log 2>&1 &
```

**说明**：
- 自动读取对应的checkpoint文件
- batch_size > 1: `checkpoint_batch.json`
- batch_size = 1: `checkpoint.json`
- 跳过已处理的任务，从中断处继续

### 6.2 清除checkpoint重新开始

```bash
# 删除旧的checkpoint
rm -f /ltstorage/home/4xin/image_table/RealHiTBench/result/qwen3vl_local/Qwen3-VL-8B-Instruct_image/checkpoint*.json

# 重新运行
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --batch_size 3' \
    > ../result/qwen3vl_local/image_fresh.log 2>&1 &
```

---

## 7. 多进程分片并行（推荐）

### 7.1 为什么需要多进程分片？

**问题**：使用 `batch_size>1` 或 `DataParallel` 时，发现只有一个GPU在计算，其他GPU只占显存不工作。

**原因**：Qwen3-VL使用 `model.generate()` 进行推理，这是自回归生成过程，无法使用 PyTorch 的 `DataParallel` 实现数据并行。

**解决方案**：使用多进程分片 - 每个GPU运行独立的Python进程，处理数据的不同分片。

### 7.2 快速启动（推荐）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

# 使用自动化脚本启动3个GPU并行推理
./run_multi_gpu.sh

# 查看运行状态
gpustat -i 1

# 监控日志
tail -f ../result/qwen3vl_local/logs/shard_*.log
```

### 7.3 手动启动（自定义配置）

#### 步骤1: 启动各分片

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

# GPU 0 - 处理 Shard 0/3
CUDA_VISIBLE_DEVICES=0 nohup python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --batch_size 1 \
    --use_flash_attn \
    --use_model_parallel \
    --resume \
    --shard_id 0 \
    --num_shards 3 \
    --skip_checkpoint ../result/qwen3vl_local/Qwen3-VL-8B-Instruct_image/checkpoint_batch.json \
    > ../result/qwen3vl_local/logs/shard_0.log 2>&1 &

# GPU 1 - 处理 Shard 1/3
CUDA_VISIBLE_DEVICES=1 nohup python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --batch_size 1 \
    --use_flash_attn \
    --use_model_parallel \
    --resume \
    --shard_id 1 \
    --num_shards 3 \
    --skip_checkpoint ../result/qwen3vl_local/Qwen3-VL-8B-Instruct_image/checkpoint_batch.json \
    > ../result/qwen3vl_local/logs/shard_1.log 2>&1 &

# GPU 2 - 处理 Shard 2/3
CUDA_VISIBLE_DEVICES=2 nohup python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --batch_size 1 \
    --use_flash_attn \
    --use_model_parallel \
    --resume \
    --shard_id 2 \
    --num_shards 3 \
    --skip_checkpoint ../result/qwen3vl_local/Qwen3-VL-8B-Instruct_image/checkpoint_batch.json \
    > ../result/qwen3vl_local/logs/shard_2.log 2>&1 &

echo "All shards launched! Monitor with: tail -f ../result/qwen3vl_local/logs/shard_*.log"
```

#### 步骤2: 监控进度

```bash
# 查看GPU使用情况（应该看到3个GPU都在计算）
watch -n 1 gpustat

# 查看各分片日志
tail -f ../result/qwen3vl_local/logs/shard_0.log
tail -f ../result/qwen3vl_local/logs/shard_1.log
tail -f ../result/qwen3vl_local/logs/shard_2.log

# 查看所有分片日志
tail -f ../result/qwen3vl_local/logs/shard_*.log

# 检查进程状态
ps aux | grep inference_qwen3vl_local
```

#### 步骤3: 合并结果

```bash
# 等待所有分片完成后，合并结果
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

python merge_shards.py \
    --output_dir ../result/qwen3vl_local \
    --model_name Qwen3-VL-8B-Instruct \
    --modality image \
    --num_shards 3

# 查看合并结果
cat ../result/qwen3vl_local/Qwen3-VL-8B-Instruct_image_merged/checkpoint_merged.json | jq '.aggregate_metrics'
```

### 7.4 分片参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--shard_id` | 当前分片ID（从0开始） | `0`, `1`, `2` |
| `--num_shards` | 总分片数（通常等于GPU数） | `3` |
| `--skip_checkpoint` | 已有checkpoint文件路径，跳过已处理的ID | `path/to/checkpoint_batch.json` |

**分片逻辑**：
1. 加载所有queries
2. 从 `--skip_checkpoint` 加载已处理的IDs并过滤
3. 将剩余queries平均分成 `num_shards` 份
4. 当前进程只处理第 `shard_id` 份数据
5. 结果保存到独立目录：`model_name_modality_shard{id}/`

### 7.5 输出目录结构

```
result/qwen3vl_local/
├── Qwen3-VL-8B-Instruct_image/              # 原始单进程结果（1314个已完成）
│   └── checkpoint_batch.json
├── Qwen3-VL-8B-Instruct_image_shard0/       # Shard 0结果
│   └── checkpoint_batch.json
├── Qwen3-VL-8B-Instruct_image_shard1/       # Shard 1结果
│   └── checkpoint_batch.json
├── Qwen3-VL-8B-Instruct_image_shard2/       # Shard 2结果
│   └── checkpoint_batch.json
├── Qwen3-VL-8B-Instruct_image_merged/       # 合并后的最终结果
│   └── checkpoint_merged.json               # 包含所有3071个queries
└── logs/
    ├── shard_0_20260128_012440.log
    ├── shard_1_20260128_012442.log
    └── shard_2_20260128_012444.log
```

### 7.6 性能对比

| 方式 | GPU利用 | 时间（1700 queries） | 加速比 |
|------|---------|---------------------|--------|
| 单GPU | 1个 @100% | ~2-3小时 | 1x |
| batch_size=3 | 1个 @100%，2个闲置 | ~2-3小时 | ❌ 无效 |
| DataParallel | 1个 @100%，2个闲置 | ~2-3小时 | ❌ 无效 |
| **多进程分片** | 3个 @30-40% | **~30-45分钟** | ✅ **3x** |

### 7.7 注意事项

⚠️ **重要**：
- 使用 `--skip_checkpoint` 可避免重复处理已完成的数据
- 每个GPU需要 ~17GB 显存（模型并行模式）
- 启动脚本 `run_multi_gpu.sh` 已自动配置conda环境
- 确保日志目录存在：`mkdir -p ../result/qwen3vl_local/logs`
- 分片数量建议等于可用GPU数量

### 7.8 其他modality的多进程并行

#### Text-HTML 模态（3个GPU并行）

```bash
# 修改 run_multi_gpu.sh 中的参数或手动启动
CUDA_VISIBLE_DEVICES=0 python inference_qwen3vl_local.py \
    --modality text --format html \
    --shard_id 0 --num_shards 3 \
    [其他参数...] &

CUDA_VISIBLE_DEVICES=1 python inference_qwen3vl_local.py \
    --modality text --format html \
    --shard_id 1 --num_shards 3 \
    [其他参数...] &

CUDA_VISIBLE_DEVICES=2 python inference_qwen3vl_local.py \
    --modality text --format html \
    --shard_id 2 --num_shards 3 \
    [其他参数...] &

# 合并结果
python merge_shards.py --modality text --format html --num_shards 3
```

---

## 8. 参数说明

### 8.1 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model_dir` | 本地模型路径 | `/mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct` |
| `--data_path` | RealHiTBench数据路径 | `/mnt/data1/users/4xin/RealHiTBench` |
| `--qa_path` | QA JSON文件目录 | `/ltstorage/home/4xin/image_table/RealHiTBench/data` |
| `--modality` | 输入模态 | `image` / `text` / `mix` |

### 8.2 模态相关参数

| 参数 | 适用模态 | 说明 | 可选值 |
|------|---------|------|--------|
| `--format` | text, mix | 表格文本格式 | `html` / `latex` / `markdown` / `csv` |

### 8.3 性能优化参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--batch_size` | 1 | 批量大小（>1开启批量推理） |
| `--use_flash_attn` | True | 使用Flash Attention 2 |
| `--no_flash_attn` | - | 禁用Flash Attention |
| `--use_model_parallel` | False | 使用模型并行（device_map='auto'） |

### 8.4 多进程分片参数（新增）

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--shard_id` | None | 当前分片ID（0, 1, 2, ...） |
| `--num_shards` | None | 总分片数（通常等于GPU数） |
| `--skip_checkpoint` | None | 已有checkpoint文件路径（跳过已处理ID） |

### 8.5 生成参数（Qwen3-VL官方推荐）

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--temperature` | 0.7 | 生成温度（0=greedy） |
| `--top_p` | 0.8 | Top-p采样 |
| `--top_k` | 20 | Top-k采样 |
| `--repetition_penalty` | 1.0 | 重复惩罚 |
| `--presence_penalty` | 1.5 | 存在惩罚 |
| `--max_tokens` | 32768 | 最大生成token数 |

### 8.6 测试与调试参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--max_queries` | -1 | 最大处理任务数（-1=全部） |
| `--question_type` | None | 过滤特定问题类型 |
| `--use_long` | False | 使用QA_long.json |
| `--resume` | False | 从checkpoint恢复 |

### 7.6 使用greedy decoding（确定性输出）

```bash
# 适用于需要可复现结果的场景
python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --temperature 0
```

---

## 8. 监控与管理

### 8.1 查看运行进程

```bash
# 查看所有qwen推理进程
ps aux | grep inference_qwen3vl_local.py | grep -v grep

# 查看GPU使用情况
gpustat -i 1
```

### 8.2 查看日志

```bash
# 实时查看日志
tail -f ../result/qwen3vl_local/image_full_batch3.log

# 查看最后50行
tail -50 ../result/qwen3vl_local/image_full_batch3.log

# 查看错误信息
grep -i error ../result/qwen3vl_local/image_full_batch3.log
```

### 8.3 终止进程

```bash
# 查找PID
ps aux | grep inference_qwen3vl_local.py | grep -v grep

# 优雅终止（推荐）
kill <PID>

# 强制终止
kill -9 <PID>
```

### 8.4 查看checkpoint进度

```bash
# 批量模式
jq '.processed_ids | length' ../result/qwen3vl_local/Qwen3-VL-8B-Instruct_image/checkpoint_batch.json

# 单任务模式
jq '.processed_ids | length' ../result/qwen3vl_local/Qwen3-VL-8B-Instruct_image/checkpoint.json
```

---

## 9. 完整评估流程（推荐）

### 9.1 Image-Only完整评估（修复后代码）

```bash
#!/bin/bash
# 1. 清理旧checkpoint
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference
rm -f ../result/qwen3vl_local/Qwen3-VL-8B-Instruct_image/checkpoint*.json

# 2. 启动完整评估
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --batch_size 3' \
    > ../result/qwen3vl_local/image_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Process started with PID: $!"
echo "Monitor log: tail -f ../result/qwen3vl_local/image_full_*.log"
```

### 9.2 完整评估所有模态

```bash
#!/bin/bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

# Image-only
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image --batch_size 3' \
    > ../result/qwen3vl_local/image_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Text-HTML
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality text --format html --batch_size 3' \
    > ../result/qwen3vl_local/text_html_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Text-LaTeX
nohup bash -c 'CUDA_VISIBLE_DEVICES=2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality text --format latex --batch_size 3' \
    > ../result/qwen3vl_local/text_latex_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Mix-LaTeX
nohup bash -c 'CUDA_VISIBLE_DEVICES=4 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality mix --format latex --batch_size 3' \
    > ../result/qwen3vl_local/mix_latex_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "All evaluations started!"
echo "Monitor: gpustat -i 1"
```

---

## 10. 常见问题

### Q1: Flash Attention加载失败

**错误**：`undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs`

**解决**：
```bash
# 自动回退到默认attention，无需处理
# 或显式禁用Flash Attention：
python inference_qwen3vl_local.py --no_flash_attn ...
```

### Q2: OOM错误

**症状**：`torch.cuda.OutOfMemoryError`

**解决方案**：
1. 减小batch_size：`--batch_size 1`
2. 使用更多GPU：`CUDA_VISIBLE_DEVICES=0,1,2`
3. 减小max_pixels（修改代码中的processor配置）

### Q3: 如何验证图片格式修复生效

查看日志中是否有：
```
Processor configured with dynamic resolution: min_pixels=200704, max_pixels=1605632
```

没有PIL Image相关的错误信息即为正常。

### Q4: Checkpoint何时保存？

- **batch_size > 1**：每个batch后保存
- **batch_size = 1**：每10个任务后保存

### Q5: 如何选择batch_size？

| GPU配置 | 推荐batch_size | 说明 |
|---------|---------------|------|
| 1x A6000 (47GB) | 3-5 | 单GPU适中设置 |
| 2x A6000 | 5-8 | 双GPU可增大 |
| 3x A6000 | 8-10 | 多GPU最优 |

建议先测试小batch确认不OOM后再增大。

---

## 11. 结果文件

### 11.1 输出目录结构

```
result/qwen3vl_local/
├── Qwen3-VL-8B-Instruct_image/
│   ├── checkpoint_batch.json        # 批量模式checkpoint
│   ├── checkpoint.json              # 单任务模式checkpoint
│   └── results_20260127_092759.json # 最终结果
├── Qwen3-VL-8B-Instruct_text_html/
│   └── results_*.json
├── Qwen3-VL-8B-Instruct_text_latex/
│   └── results_*.json
└── Qwen3-VL-8B-Instruct_mix_latex/
    └── results_*.json
```

### 11.2 结果文件内容

```json
{
  "config": {
    "model_dir": "...",
    "modality": "image",
    "batch_size": 3,
    "total_queries": 3071,
    "duration_seconds": 3456.78,
    "throughput": 0.89
  },
  "aggregate_metrics": {
    "Fact Checking": {"F1": 0.85, "EM": 0.78, ...},
    "Numerical Reasoning": {...},
    ...
  },
  "results": [
    {
      "id": 1,
      "Question": "...",
      "Prediction": "...",
      "Metrics": {...},
      "ProcessingTime": 12.34
    },
    ...
  ]
}
```

---

## 📞 联系与支持

- **代码问题**：检查日志文件中的错误信息
- **性能优化**：调整batch_size和GPU配置
- **结果验证**：对比不同模态的aggregate_metrics

---

**最后更新**：2026-01-28  
**适用版本**：修复后的inference_qwen3vl_local.py  
**新增功能**：
- 图片格式修复（PIL大图片支持）
- Qwen官方推荐参数
- 多进程分片并行（`--shard_id`, `--num_shards`, `--skip_checkpoint`）
- 自动化启动脚本（`run_multi_gpu.sh`）
- 结果合并工具（`merge_shards.py`）
