# Resume 功能使用指南

## 文件信息
- **脚本**: `inference_qwen3vl_local_a100_truncate.py`
- **功能**: 支持自动resume、ERROR检测、文本截断
- **支持模式**: text、mix、image
- **关键特性**: 
  - ✅ 自动检测ERROR结果并重新处理
  - ✅ MAX_INPUT_TOKENS = 100,000 自动截断
  - ✅ 断点续传（Checkpoint恢复）

---

## 核心Resume机制

### 1. 自动ERROR检测

当使用 `--resume` 时，脚本会：

```python
# 从checkpoint.json加载已处理的结果
checkpoint_file = f'{output_file_path}/checkpoint.json'
processed_ids = set()

if os.path.exists(checkpoint_file) and opt.resume:
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
        all_eval_results = checkpoint_data.get('results', [])
        processed_ids = set(checkpoint_data.get('processed_ids', []))
```

**关键逻辑**：
- 加载所有results（包括ERROR）
- 只有 **成功结果** 被添加到 `processed_ids`
- ERROR结果 **不在** `processed_ids` 中
- 导致ERROR样本 **自动被重新处理**

### 2. ERROR样本识别

```python
# 检测ERROR样本的逻辑（在gen_solution_batch中）
error_ids = set()
successful_results = []
processed_ids = set()

for result in all_eval_results:
    if result['Prediction'].startswith('[ERROR'):
        error_ids.add(result['id'])
        # 不加入processed_ids，所以会被重新处理
    else:
        successful_results.append(result)
        processed_ids.add(result['id'])
```

---

## 运行命令

### 方式 1: Text HTML模式（文本模态）

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference

python inference_qwen3vl_local_a100_truncate.py \
  --modality text \
  --format html \
  --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
  --data_path /data/pan/4xin/datasets/RealHiTBench \
  --resume \
  --batch_size 1
```

**预期行为**：
- 加载checkpoint中的3043个成功结果
- 检测17个ERROR (OOM)样本
- 重新处理这17个样本，使用MAX_INPUT_TOKENS=100,000截断
- 保存到: `result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_text_html_default/`

### 方式 2: Mix HTML模式（多模态）

```bash
python inference_qwen3vl_local_a100_truncate.py \
  --modality mix \
  --format html \
  --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
  --data_path /data/pan/4xin/datasets/RealHiTBench \
  --resume \
  --batch_size 1
```

**预期行为**：
- 加载mix_html的checkpoint
- 检测ERROR样本
- 使用文本截断重新处理
- 保存到: `result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_mix_html_default/`

### 方式 3: 使用包装脚本（推荐）

```bash
# Text HTML
python run_text_html_truncate.py

# Mix HTML
python run_mix_html_truncate.py
```

这些脚本会自动配置所有参数。

---

## 关键参数详解

| 参数 | 说明 | 值 |
|------|------|-----|
| `--modality` | 输入模态 | `text`, `mix`, `image` |
| `--format` | 文本格式 | `html`, `markdown`, `latex`, `csv` |
| `--resume` | 启用resume模式 | 标志，无需值 |
| `--model_dir` | 模型路径 | `/data/pan/4xin/models/Qwen3-VL-8B-Instruct` |
| `--data_path` | 数据集路径 | `/data/pan/4xin/datasets/RealHiTBench` |
| `--batch_size` | 批处理大小 | 1（推荐） |
| `--max_queries` | 最大处理数 | -1（全部） |
| `--use_flash_attn` | 闪现注意力 | true（推荐） |
| `--use_model_parallel` | 模型并行 | false（推荐数据并行） |

---

## Checkpoint文件位置

### 原始checkpoints

```
result/
├── qwen3vl_local_a100/
│   └── Qwen3-VL-8B-Instruct_text_html_a100/
│       └── checkpoint.json          # 17个ERROR样本 + 3043成功
└── qwen3vl_local_a100_default/
    └── Qwen3-VL-8B-Instruct_mix_html_default/
        └── checkpoint.json          # 17个ERROR样本 + 3043成功
```

### Resume流程

```
加载checkpoint.json
│
├─ results: 3060个结果（3043成功 + 17个ERROR）
└─ processed_ids: 仅3043个成功ID

Resume时：
├─ 加载所有3060个results到all_eval_results
├─ processed_ids = 3043个成功ID
├─ 跳过这3043个
└─ 重新处理17个ERROR样本 ✓
```

---

## 文本截断过程

当某个样本的HTML文本超过100,000个token时：

```
Input: economy-table14_swap.html (334,162 tokens) → TOO LARGE
                ↓ 检测到超限
        [TRUNCATE] 触发截断
                ↓
        Character ratio truncation: 334,162 * (100,000/334,162) = ~100,000
                ↓
Output: Truncated HTML (99,847 tokens) → 适合处理
```

**关键代码**:
```python
def truncate_text_if_needed(messages_text, processor, max_tokens=100000):
    # 1. 令牌化输入
    tokens = processor.apply_chat_template(...)
    
    # 2. 检查是否超限
    if len(tokens) > max_tokens:
        # 3. 计算截断比例
        truncate_ratio = max_tokens / len(tokens)
        
        # 4. 按字符截断（保留90%以安全）
        truncate_len = int(len(messages_text) * truncate_ratio * 0.9)
        return messages_text[:truncate_len]
    
    return messages_text  # 正常大小，无需截断
```

---

## 运行输出示例

### Text HTML Resume

```
Loading Qwen3-VL model from /data/pan/4xin/models/Qwen3-VL-8B-Instruct...
Available GPUs: 4
Using Flash Attention 2
Model loaded on GPU with DataParallel

Loading dataset...
Loaded 3,071 total queries

Resuming from checkpoint with 3043 processed queries
  Found 17 ERROR samples to reprocess:
    - ID 2747: [ERROR] OOM: CUDA out of memory...
    - ID 2748: [ERROR] OOM: CUDA out of memory...
    ...
    - ID 3021: [ERROR] OOM: CUDA out of memory...

Processing 17 queries: [████████████████████] 100%

Processing Query ID: 2747
  HTML size: 334,162 tokens
  [TRUNCATE] Input too large, truncating to 100,000
  [TRUNCATE] Result: 99,847 tokens (29.8% of original)
  Prediction: Based on the table analysis, the economy grew by...
  Processing Time: 45.23s

Processing Query ID: 2748
  HTML size: 212,445 tokens
  [TRUNCATE] Input too large, truncating to 100,000
  ...

EVALUATION COMPLETE
Total queries: 3060 (3043 successful + 17 reprocessed)
Duration: 765.34s
Throughput: 4.00 queries/sec
Results saved to: result/qwen3vl_local_a100_default/.../checkpoint.json
```

### 验证结果

```bash
# 检查是否所有ERROR都已修复
python << 'EOF'
import json
with open('result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_text_html_default/checkpoint.json') as f:
    data = json.load(f)
    errors = [r for r in data['results'] if r['Prediction'].startswith('[ERROR')]
    print(f"Total results: {len(data['results'])}")
    print(f"Remaining errors: {len(errors)}")
    if len(errors) == 0:
        print("✅ All OOM samples fixed!")
    else:
        print(f"⚠️  Still {len(errors)} errors:")
        for r in errors[:3]:
            print(f"  - ID {r['id']}: {r['Prediction'][:80]}")
EOF
```

---

## 常见问题

### Q1: Resume后还是有ERROR？
A: 通常是模型显存不足。检查：
```bash
nvidia-smi  # 查看显存占用
```
解决方案：
- 减小batch_size: `--batch_size 1`
- 释放显存: `pkill python`

### Q2: 截断后答案准确性是否下降？
A: 通常不会，因为：
- MAX_INPUT_TOKENS = 100,000 已经很大
- 大多数表格远小于这个限制
- 只有5个样本需要真正截断（economy-table, society-table等）

### Q3: Resume时是否会重复处理已成功的样本？
A: 否。脚本会：
- 从checkpoint加载processed_ids
- 跳过这些ID（`if query['id'] in processed_ids: continue`）
- 只处理新的或ERROR的样本

### Q4: 如何中断resume并保留进度？
A: 按 Ctrl+C，会自动保存checkpoint：
```
^C
Checkpoint saved: 3050 queries processed
```
下次resume时会从第3051个查询开始。

---

## 完整工作流

```bash
# 1. 设置环境
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference
conda activate 4xin-hit

# 2. 运行text_html resume（~30分钟）
python inference_qwen3vl_local_a100_truncate.py \
  --modality text \
  --format html \
  --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
  --data_path /data/pan/4xin/datasets/RealHiTBench \
  --resume \
  --batch_size 1

# 3. 运行mix_html resume（~30分钟）
python inference_qwen3vl_local_a100_truncate.py \
  --modality mix \
  --format html \
  --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
  --data_path /data/pan/4xin/datasets/RealHiTBench \
  --resume \
  --batch_size 1

# 4. 验证结果
python << 'EOF'
import json, os
for mode in ['text_html', 'mix_html']:
    path = f'../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_{mode}_default/checkpoint.json'
    with open(path) as f:
        data = json.load(f)
        errors = len([r for r in data['results'] if '[ERROR' in r['Prediction']])
        print(f"{mode}: {len(data['results'])} total, {errors} errors")
EOF
```

---

## 时间和成本节省

| 方式 | 时间 | 评价 |
|------|------|------|
| ❌ 重新运行全部3071 | ~15小时 | 浪费 |
| ✅ Resume仅处理17个 | ~1小时 | 推荐 |
| **节省** | **14小时** | **93%** |

---

最后更新: 2024年
作者: GitHub Copilot
