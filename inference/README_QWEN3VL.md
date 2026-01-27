# Qwen3-VL RealHiTBench 评测使用指南

本项目支持两种推理方式：
1. **API推理**：使用阿里云百炼平台 Qwen3-VL API（`inference_qwen3vl.py`）
2. **本地推理**：使用本地部署的开源模型（`inference_qwen3vl_local.py`）⭐ **推荐**

## 快速开始

### 1. 环境准备

#### 核心依赖（已验证安装）

```bash
PyTorch: 2.9.1+cu128
Transformers: 4.57.6
Qwen-VL-Utils: 0.0.14
NumPy: >=1.23,<2.0
```

#### 完整依赖列表

**必需包**：
- transformers >= 4.55.0
- accelerate >= 1.10.0
- qwen-vl-utils == 0.0.14
- torch >= 2.0 (当前: 2.9.1+cu128)
- numpy >= 1.23,<2.0

**评估指标**：
- evaluate >= 0.4.2
- sacrebleu >= 2.5.1
- rouge_score >= 0.1.2
- datasets >= 3.6.0

**其他**：
- pillow, tqdm, PyYAML, requests, beautifulsoup4, html5lib, openpyxl

#### 依赖安装

```bash
# 推荐：仅安装核心包（最快）
pip install transformers accelerate qwen-vl-utils evaluate rouge_score sacrebleu datasets pillow beautifulsoup4

# 或从 requirements.txt 安装（已修复版本冲突）
pip install -r requirements.txt
```

#### ✅ 依赖版本修复说明

本项目修复了 requirements.txt 中的版本冲突：

| 包名 | 原问题 | 修复方案 | 说明 |
|------|--------|---------|------|
| numpy | `==2.0.2` 与 gradio/opencv 冲突 | `>=1.23,<2.0` | 允许兼容的 numpy 1.x |
| numba | `==0.60.0` 与 vllm 0.10.0 冲突 | `>=0.60.0` | vllm 需要 0.61.2+ |
| gradio | `==4.16.0` 固定过严 | `>=4.16.0` | 允许灵活版本选择 |
| opencv | `==4.12.0.88` 需要 numpy>=2 | `>=4.10.0` | 使用兼容版本 |
| pandas | `==2.3.0` 固定过严 | `>=2.0.0` | 允许灵活版本选择 |

**安装建议**：
```bash
# 如果 pip install -r requirements.txt 失败，使用核心包安装
pip install transformers accelerate qwen-vl-utils evaluate \
            rouge_score sacrebleu datasets pillow torch torchvision

# 如需 Flash Attention 2（可选，加快推理）
pip install flash-attn --no-build-isolation
```
#### 下载数据集
```
huggingface-cli download spzy/RealHiTBench \
  --repo-type dataset \
  --local-dir /mnt/data1/users/4xin/RealHiTBench \
  --local-dir-use-symlinks False
```
#### 下载Qwen3-vl-8B-Instruct模型
```
python -m pip install -U hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'PY'
from huggingface_hub import snapshot_download

local_dir = "/data/pan/4xin/models/Qwen3-VL-8B-Instruct"
snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    repo_type="model",
    local_dir=local_dir,
    # 不写 max_workers 就是默认并发（更快）
)
print("✅ Downloaded to:", local_dir)
PY
```

### 2. 数据集结构

确保数据集位于以下位置：

```
数据主目录: /mnt/data1/users/4xin/RealHiTBench/
├── image/                    # 表格图片（.png）
├── html/                     # HTML 格式表格
├── csv/                      # CSV 格式表格
├── markdown/                 # Markdown 格式表格
├── latex/                    # LaTeX 格式表格
├── tables/                   # Excel 文件（可视化任务）
└── data/                     # QA 数据文件（若分离）

QA数据目录: /ltstorage/home/4xin/image_table/RealHiTBench/data/
├── QA_final.json             # 主评测集（3071条）
├── QA_long.json              # 长序列评测集
└── QA_structure.json         # 结构理解评测集
```

### 3. API 配置（仅用于 API 模式）

使用阿里云百炼平台 Qwen3-VL API：
- **API Key**: `sk-xxx`（Singapore 地区）
- **Base URL**: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- **模型**: `qwen3-vl-flash` 或 `qwen3-vl-plus`

---

## 使用方法

### 快速测试

#### 1. 本地模型测试（推荐）

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

# 自动测试：5 个样本，3 种模态，所有 GPU
./test_qwen3vl_local.sh

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 ./test_qwen3vl_local.sh
```

#### 2. API 模式测试

```bash
./test_qwen3vl.sh sk-YOUR_API_KEY
```

### 单独运行某种模态

#### 方式1：本地推理（推荐）⭐

**Image-only（纯图像输入）**

```bash
python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --max_queries 100 \
    --temperature 0

# 推荐命令（后台运行 + 保存日志）
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --batch_size 3 \
    --temperature 0' \
    > ../result/qwen3vl_local/image_batch3.log 2>&1 &

# 然后监控进度
tail -f ../result/qwen3vl_local/image_batch3.log
```

**Text-only（纯文本输入）**

```bash
python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality text \
    --format html \
    --max_queries 100
```

**Mix（图像+文本输入）**

```bash
python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality mix \
    --format latex \
    --max_queries 100
```

#### 方式2：API 推理

**Image-only**

```bash
python inference_qwen3vl.py \
    --model qwen3-vl-flash \
    --api_key sk-YOUR_API_KEY \
    --modality image \
    --max_queries 100
```

**Text-only**

```bash
python inference_qwen3vl.py \
    --model qwen3-vl-flash \
    --api_key sk-YOUR_API_KEY \
    --modality text \
    --format html \
    --max_queries 100
```

**Mix**

```bash
python inference_qwen3vl.py \
    --model qwen3-vl-flash \
    --api_key sk-YOUR_API_KEY \
    --modality mix \
    --format latex \
    --max_queries 100
```

### 完整评测（全部数据集）

#### 本地模型完整评测

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

# Image 模态
CUDA_VISIBLE_DEVICES=0 nohup python -u inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality image \
    --temperature 0 > ../result/eval_image.log 2>&1 &

# Text-HTML
CUDA_VISIBLE_DEVICES=1 nohup python -u inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality text --format html \
    --batch_size 1 --temperature 0 > ../result/eval_text_html.log 2>&1 &

# Text-LaTeX
CUDA_VISIBLE_DEVICES=2 nohup python -u inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality text --format latex \
    --batch_size 1 --temperature 0 > ../result/eval_text_latex.log 2>&1 &

# Mix-LaTeX
CUDA_VISIBLE_DEVICES=4 nohup python -u inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --data_path /mnt/data1/users/4xin/RealHiTBench \
    --qa_path /ltstorage/home/4xin/image_table/RealHiTBench/data \
    --modality mix --format latex \
    --batch_size 1 --temperature 0 > ../result/eval_mix_latex.log 2>&1 &
```

#### API 完整评测

```bash
./run_full_eval.sh sk-YOUR_API_KEY
```

---

## 本地模型详细参数

### 模型下载

模型已下载到：`/mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct`

如需重新下载：
```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct
```

### 推理参数说明

| 参数 | 必需 | 类型 | 说明 | 默认值 |
|------|------|------|------|--------|
| `--model_dir` | **是** | str | 本地模型路径 | 必需 |
| `--data_path` | 否 | str | 数据集路径 | `/mnt/data1/users/4xin/RealHiTBench` |
| `--qa_path` | 否 | str | QA 文件目录 | `{data_path}/data` |
| `--modality` | **是** | str | 输入模态 | 必需 |
| `--format` | 条件* | str | 文本格式 | `html` (仅 text/mix) |
| `--temperature` | 否 | float | 生成温度 | `0` (贪婪解码) |
| `--top_p` | 否 | float | Top-p 采样 | `0.8` |
| `--top_k` | 否 | int | Top-k 采样 | `20` |
| `--max_tokens` | 否 | int | 最大生成 token | `4096` |
| `--batch_size` | 否 | int | 批处理大小 | `1` (text/mix) |
| `--question_type` | 否 | str | 筛选问题类型 | 全部 |
| `--max_queries` | 否 | int | 最大样本数 | `-1` (全部) |
| `--resume` | 否 | bool | 断点续传 | `False` |
| `--use_flash_attn` | 否 | bool | 使用 Flash Attn 2 | `True` |
| `--no_flash_attn` | 否 | bool | 禁用 Flash Attn | - |

\* `--format` 在 `text` 或 `mix` 模态下必需

### GPU 多卡配置

```bash
# 单 GPU
CUDA_VISIBLE_DEVICES=0 python inference_qwen3vl_local.py ...

# 多 GPU（自动分配）
CUDA_VISIBLE_DEVICES=0,1,2 python inference_qwen3vl_local.py ...

# 禁用 Flash Attention（如有兼容性问题）
python inference_qwen3vl_local.py --model_dir /path --no_flash_attn
```

### 显存要求

| 配置 | 显存需求 | 推荐 GPU | 备注 |
|------|---------|---------|------|
| BF16 (标准) | ~18-20GB | 1x RTX 3090/4090/A6000 | - |
| 启用 Flash Attn 2 | ~15-17GB | 更高效 | 需额外安装 |
| 长序列 (>4K tokens) | +4-8GB | 2x GPU | 自动分配 |
| 禁用 Flash Attn | ~20-22GB | - | `--no_flash_attn` |

### 性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| 模型大小 | 8B 参数 | 开源版本 |
| 动态分辨率 | 256×28² ~ 2048×28² | 自适应输入 |
| 最大 token 长度 | 128K | 支持超长文本 |
| 单 GPU 推理速度 | ~2-5 sec/sample | 取决于输入长度 |
| 100 样本预计时间 | ~5-10 分钟 | A6000 GPU |
| 完整数据集耗时 | ~4-6 小时 | 3071 样本单 GPU |

---

## 新增功能：处理时间追踪

### ✨ ProcessingTime 字段

每个查询的结果中新增 `ProcessingTime` 字段，记录单个样本的处理时间（秒）：

```json
{
  "QA_ID": 1,
  "Question": "...",
  "User_Answer": "...",
  "ProcessingTime": 2.34,
  ...
}
```

此字段用于：
- 识别性能瓶颈（长序列处理）
- 评估推理延迟分布
- 优化 batch 处理配置
- 预估完整数据集耗时

---

## 关键 Bug 修复

### ✅ Fix 1: OOM 错误处理

**问题**：文本超长时导致 CUDA OOM，程序崩溃。

**修复位置**：`inference_qwen3vl_local.py` 第 128-178 行

**实现**：
```python
try:
    response = model.generate(...)
except torch.cuda.OutOfMemoryError:
    response = "[ERROR] OOM: Text too large (X tokens)"
    torch.cuda.empty_cache()
```

**结果**：超大文本返回错误标记而非崩溃，允许继续处理。

### ✅ Fix 2: 评估提示关键字修复

**问题**：KeyError 'User_Answer' vs 'Predicted_Answer'，导致 Summary/Anomaly 查询被跳过。

**影响**：
- 前次运行：HTML 损失 ~32/400 (8%)，LaTeX 损失 ~133/1170 (11%)
- 总计：~165 条查询未保存

**修复位置**：`inference_qwen3vl_local.py` 第 1225 行

**修复**：
```python
# 之前
eval_prompt = prompt.format(Predicted_Answer=user_answer, ...)

# 之后
eval_prompt = prompt.format(User_Answer=user_answer, ...)
```

**验证**：新检查点在 10 条查询后已包含 Summary Analysis (1) 和 Anomaly Analysis (1)

### ✅ Fix 3: 依赖版本冲突

**问题**：requirements.txt 中多个包版本约束过严，导致无法安装。

**修复**：见[环境准备](#1-环境准备)部分的版本修复表。

---

## 评估指标

### 标准 QA 指标（所有任务）

- **F1**：词级 F1 分数
- **EM**：精确匹配率
- **ROUGE-L**：最长公共子序列
- **SacreBLEU**：BLEU 分数

### 任务特定指标

| 任务类型 | 评估方式 | 指标 |
|---------|---------|------|
| Fact Checking | 自动匹配 | F1, EM, ROUGE-L |
| Numerical Reasoning | 自动匹配 | F1, EM, ROUGE-L |
| Data Analysis | GPT-4o 评分 | GPT_EVAL (0-10) |
| Visualization | 代码执行 | ECR, Pass |
| Structure Comprehension | 自动匹配 | F1, EM |

---

## 结果输出

### 本地模型结果存储

```
result/qwen3vl_local/
├── Qwen3-VL-8B-Instruct_image/
│   ├── results_20260127_211234.json
│   ├── checkpoint.json
│   └── Qwen3-VL-8B-Instruct_image.log
├── Qwen3-VL-8B-Instruct_text_html/
│   └── results_20260127_213456.json
├── Qwen3-VL-8B-Instruct_text_latex/
│   └── results_20260127_215678.json
└── Qwen3-VL-8B-Instruct_mix_latex/
    └── results_20260127_221234.json
```

### API 模式结果存储

```
result/qwen3vl/
├── qwen3-vl-flash_image/
│   └── results_20260127_*.json
├── qwen3-vl-flash_text_html/
│   └── results_20260127_*.json
└── qwen3-vl-flash_mix_latex/
    └── results_20260127_*.json
```

### 结果文件格式

```json
{
  "config": {
    "model": "Qwen3-VL-8B-Instruct",
    "modality": "image",
    "total_queries": 770,
    "processed_queries": 770,
    "duration_seconds": 2345.67
  },
  "summary": {
    "avg_F1": 0.7856,
    "avg_EM": 0.6234,
    "avg_ROUGE-L": 0.7123,
    "Fact Checking_avg_F1": 0.8234,
    "Numerical Reasoning_avg_F1": 0.7456,
    "avg_ProcessingTime": 3.04
  },
  "results": [
    {
      "QA_ID": 1,
      "Question": "...",
      "User_Answer": "...",
      "ProcessingTime": 2.34,
      "F1": 0.8,
      ...
    }
  ]
}
```

---

## 断点续传

支持断点续传，使用 `--resume` 参数自动加载前次进度：

```bash
# 运行被中断的评测
python inference_qwen3vl_local.py \
    --model_dir /mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct \
    --modality image \
    --resume

# 检查点每处理 10 条样本自动保存
# 从最后保存点继续（无需重新处理已完成样本）
```

### 检查点文件

```json
{
  "results": [...],
  "processed_ids": [1, 2, 3, ..., 50]
}
```

---

## 常见问题

### 1. API 速率限制

```bash
--sleep_time 2.0  # 每次调用间隔 2 秒
```

### 2. 文本过长导致 OOM

已修复！超长文本自动返回 `[ERROR]` 标记，不再崩溃。

```
若仍遇问题，尝试：
- 减小 --max_tokens
- 使用 --no_flash_attn
- 分批处理 --max_queries 1000
```

### 3. CUDA OOM

```bash
# 尝试以下方案
python inference_qwen3vl_local.py --no_flash_attn ...
python inference_qwen3vl_local.py --max_queries 500 ...
CUDA_VISIBLE_DEVICES=0 python ...
```

### 4. 模型权重加载失败

```bash
# 重新下载模型
huggingface-cli delete-cache
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir /path
```

### 5. 查看实时进度

```bash
# 进度条显示：使用 tqdm 实时更新
tail -f ../result/eval_image.log
```

### 6. GPT 评估失败

数据分析任务需要 OpenAI API key，若未配置，GPT_EVAL 字段为 `N/A`。

---

## 工作流示例

### 完整评测流程

```bash
cd /ltstorage/home/4xin/image_table/RealHiTBench/inference

# Step 1: 快速测试验证环境
./test_qwen3vl_local.sh
# 预期：~15 秒，5 个样本，3 种模态

# Step 2: 查看结果
ls -lh ../result/qwen3vl_local/Qwen3-VL-8B-Instruct_image/

# Step 3: 运行完整评测
# (使用上面的多 GPU 后台运行命令)

# Step 4: 监控进度
watch -n 5 'ps aux | grep inference_qwen3vl_local'
tail -f ../result/eval_*.log

# Step 5: 汇总结果
python << 'EOF'
import json, glob
from pathlib import Path

results_dir = Path("../result/qwen3vl_local")
for result_file in sorted(results_dir.glob("*/results_*.json")):
    with open(result_file) as f:
        data = json.load(f)
    cfg = data["config"]
    summary = data["summary"]
    print(f"{cfg['modality']:5s} {cfg.get('format', 'N/A'):8s} | "
          f"F1: {summary.get('avg_F1', 0):.4f} | "
          f"Samples: {cfg['processed_queries']} | "
          f"Time: {summary.get('avg_ProcessingTime', 0):.2f}s/sample")
EOF
```

### 预期耗时

- **测试 (5 样本)**：~15 秒
- **100 样本**：~5-10 分钟
- **770 样本 (image)**：~40-60 分钟
- **完整数据集 (3071+ 样本)**：~4-6 小时（单 GPU）

### 并行加速

使用多 GPU 并行评测可加速 3-4 倍：

```bash
# 同时在 4 个 GPU 上运行
CUDA_VISIBLE_DEVICES=0 python ... --modality image > eval_image.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python ... --modality text --format html > eval_html.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python ... --modality text --format latex > eval_latex.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python ... --modality mix --format latex > eval_mix.log 2>&1 &

# 总耗时：~2-3 小时（从 4-6 小时）
```

---

## 本地 vs API 对比

| 特性 | 本地模型 | API 模式 |
|------|---------|---------|
| **推理速度** | ~2-5 sec/样本 | ~2-5 sec/样本 |
| **显存需求** | ~18-20GB | 无 |
| **离线使用** | ✅ | ❌ |
| **并发能力** | 受 GPU 限制 | 受 API 限制 |
| **成本** | 仅电费 | 按 token 计费 |
| **模型版本** | 8B 开源版 | flash/plus 专有版 |
| **质量** | 优秀 | 更优秀（flash/plus） |
| **稳定性** | 100% | 依赖网络 |

---

## 技术栈

- **推理框架**：Hugging Face Transformers
- **加速库**：Flash Attention 2 (可选)
- **多 GPU**：Accelerate
- **评估**：Evaluate + Custom Metrics
- **监控**：TQDM + 日志文件
- **文本解析**：BeautifulSoup4 (HTML/XML)

---

## 获取帮助

遇到问题？检查清单：

- [ ] PyTorch 版本 >= 2.0
- [ ] Transformers >= 4.55.0
- [ ] 模型文件存在：`/mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct`
- [ ] 数据集路径正确
- [ ] GPU 显存 >= 18GB
- [ ] CUDA 环境正确配置

---

**最后更新**：2026-01-27
**验证环境**：PyTorch 2.9.1+cu128, Transformers 4.57.6
**Bug 修复**：OOM 处理、评估提示关键字、依赖版本冲突
**新增功能**：ProcessingTime 追踪、BeautifulSoup4 支持
