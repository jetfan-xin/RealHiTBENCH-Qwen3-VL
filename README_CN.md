# RealHiTBench - 评测原理详解

<div align="left" style="line-height: 1;">
  <a href="" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53%3F?color=green" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="" style="margin: 2px;">
    <img alt="Data License" src="https://img.shields.io/badge/Data_License-cc--by--nc--4.0-blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

Official repository for paper: [**RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis**](https://arxiv.org/abs/2506.13405)

---

## 📖 目录

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Task Types & Evaluation Logic](#task-types--evaluation-logic)
  - [1. Fact Checking](#1-fact-checking)
  - [2. Numerical Reasoning](#2-numerical-reasoning)
  - [3. Data Analysis](#3-data-analysis)
  - [4. Visualization (Chart Generation)](#4-visualization-chart-generation)
  - [5. Structure Comprehending](#5-structure-comprehending)
- [Evaluation Metrics Deep Dive](#evaluation-metrics-deep-dive)
  - [QA Metrics (F1, EM, ROUGE-L, SacreBLEU)](#qa-metrics-text-based-tasks)
  - [Chart Generation Metrics (ECR, Pass@1)](#chart-generation-metrics-visualization-task)
- [Code Execution Flow](#code-execution-flow-visualization-task)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

---

## Overview

**RealHiTBench** 是一个全面评估大型语言模型 (LLM) 和多模态大模型 (MLM) 在复杂层级表格理解与推理能力的基准测试。

### 核心特点

| 维度 | 规模 |
|------|------|
| **表格数量** | 708 张 |
| **领域覆盖** | 24 个领域 |
| **QA 对数量** | 3,752 对 |
| **任务类型** | 5 大类、16 子类 |
| **输入模态** | Image / Text (LaTeX, HTML, CSV, Markdown) / Mix |

### 表格复杂结构类别

- **Hierarchical Column Header**: 多级列头合并
- **Hierarchical Row Header**: 缩进或合并的行头层级
- **Nested Sub-Tables**: 全宽分隔行划分的子表格
- **Multi-Table Join**: 结构相似的子表格隐式组合
- **Miscellaneous**: 备注文本、单元格颜色等辅助信息

---

## Dataset Structure

```
data/
├── QA_final.json          # 主数据集 (3,752 QA pairs)
├── QA_structure.json      # Structure Comprehending 专用
├── QA_long.json           # 长文本测试集
├── image/                 # PNG 表格图片
├── latex/                 # LaTeX 格式表格
├── html/                  # HTML 格式表格
├── csv/                   # CSV 格式表格
├── tables/                # Excel 文件 (Visualization 任务专用)
└── markdown/              # Markdown 格式表格
```

---

## Task Types & Evaluation Logic

### 1. Fact Checking

**任务目标**: 从表格中检索并验证事实性信息

| 子类型 | 描述 | 示例问题 |
|--------|------|----------|
| **Value-Matching** | 直接定位单元格值 | "What was the unemployment rate in 2020?" |
| **Multi-hop Fact Checking** | 多跳推理定位 | "Find the year with highest agriculture employment and its total population" |
| **Inference-based** | 基于规则推断 | "Is the growth rate positive for all years?" |

**Ground Truth 格式**:
```json
{
  "FinalAnswer": "1955, 62170",
  "ProcessedAnswer": "1955, 62170"
}
```

**评测指标**: F1, EM, ROUGE-L, SacreBLEU

**判定逻辑**: 将模型输出与 `ProcessedAnswer` 进行文本匹配，经标准化后计算各指标分数。

---

### 2. Numerical Reasoning

**任务目标**: 对表格数据进行数值计算与推理

| 子类型 | 描述 | 示例问题 |
|--------|------|----------|
| **Ranking** | 排序比较 | "Rank age groups by employment percentage" |
| **Comparison** | 数值比较 | "Which year had higher GDP?" |
| **Calculation** | 算术运算 | "Calculate the difference between 2020 and 2019" |
| **Counting** | 计数统计 | "How many years exceeded 10%?" |
| **Multi-hop** | 多步数值推理 | "Sum the top 3 values and divide by total" |

**Ground Truth 格式**:
```json
{
  "FinalAnswer": "35 to 39 years, 35 to 44 years",
  "ProcessedAnswer": "35 to 39 years, 35 to 44 years"
}
```

**评测指标**: F1, EM, ROUGE-L, SacreBLEU

---

### 3. Data Analysis

**任务目标**: 对表格数据进行统计分析与洞察

| 子类型 | 描述 | 示例问题 |
|--------|------|----------|
| **Rudimentary Analysis** | 基础统计 | "What is the mean and standard deviation?" |
| **Summary Analysis** | 概括总结 | "Summarize the main trends in this table" |
| **Predictive Analysis** | 趋势预测 | "Predict the value for next year" |
| **Exploratory Analysis** | 相关性探索 | "Find correlations between columns" |
| **Anomaly Analysis** | 异常检测 | "Identify any outliers in the data" |

**Ground Truth 格式**:
```json
{
  "FinalAnswer": "5.80, 1.62",
  "ProcessedAnswer": "5.80, 1.62"
}
```

**评测指标**: F1, EM, ROUGE-L, SacreBLEU + **GPT_EVAL** (0-100分)

> **GPT_EVAL**: 对于开放性分析题（Summary、Predictive、Exploratory、Anomaly），使用 GPT-4o 作为评判者，评估答案的正确性与完整性。

---

### 4. Visualization (Chart Generation)

**任务目标**: 根据表格数据生成可视化图表代码

| 子类型 | 描述 |
|--------|------|
| **BarChart Generation** | 柱状图生成 |
| **LineChart Generation** | 折线图生成 |
| **PieChart Generation** | 饼图生成 |
| **ScatterChart Generation** | 散点图生成 |

**⚠️ 关键区别**: 这是唯一需要**生成可执行代码**的任务，评测方式与其他任务完全不同。

#### Ground Truth 格式

| 字段 | 内容 | 示例 |
|------|------|------|
| `FinalAnswer` | **完整的 Python matplotlib 代码** | `"import pandas as pd\nimport matplotlib.pyplot as plt\n..."` |
| `ProcessedAnswer` | **从代码执行结果中提取的 Y 轴数值** | `"[[56787, 59091], [6260, 4744]]"` |

**完整 Ground Truth 示例**:

```json
{
  "QuestionType": "Visualization",
  "SubQType": "LineChart Generation",
  "Question": "Please create a line chart comparing employed vs unemployed population...",
  "FinalAnswer": "import pandas as pd\nimport matplotlib.pyplot as plt\ndf = pd.read_excel('table.xlsx')\n...\nplt.show()",
  "ProcessedAnswer": "[[56787, 59091, 59891], [6260, 4744, 4521]]"
}
```

**评测指标**: **ECR** (代码可执行率) + **Pass@1** (数据正确率)

详见下方 [Chart Generation Metrics](#chart-generation-metrics-visualization-task)

---

### 5. Structure Comprehending

**任务目标**: 理解复杂表格结构并回答相关问题（通常需要理解合并单元格、层级关系等）

**Ground Truth 格式**: 与 Fact Checking 相同

**评测指标**: F1, EM, ROUGE-L, SacreBLEU

> ⚠️ **注意**: 当前数据集中 Structure Comprehending 任务的 `FinalAnswer` 字段为空，导致所有指标为 0。这是数据标注问题，非代码 bug。

---

## Evaluation Metrics Deep Dive

### QA Metrics (Text-based Tasks)

适用于: Fact Checking, Numerical Reasoning, Data Analysis, Structure Comprehending

#### 预处理流程

在计算所有指标之前，答案会经过标准化处理：

```python
def normalize_answer(s):
    """标准化答案文本"""
    s = s.lower()                      # 1. 转小写
    s = remove_articles(s)             # 2. 移除冠词 (a, an, the)
    s = remove_punctuation(s)          # 3. 移除标点
    s = collapse_whitespace(s)         # 4. 合并空白字符
    return s

def process_decimal(s):
    """小数标准化：保留1位小数"""
    # "3.14159" → "3.1"
    # "10.567" → "10.6"
```

#### Exact Match (EM)

**定义**: 标准化后的答案是否完全匹配

```python
EM = 1.0 if normalize_answer(reference) == normalize_answer(prediction) else 0.0
```

**报告格式**: 百分比 (EM × 100)

**示例**:
| Reference | Prediction | EM |
|-----------|------------|-----|
| "1955, 62170" | "1955, 62170" | 100% |
| "1955, 62170" | "1955,62170" | 100% (标点移除后相同) |
| "1955, 62170" | "62170, 1955" | 0% (顺序不同) |

---

#### Word-level F1 Score

**定义**: 基于**词级别** (word-level) 的 F1 分数，而非字符级别

```python
def word_f1(reference, prediction):
    ref_words = normalize_answer(reference).split()
    pred_words = normalize_answer(prediction).split()
    
    common = Counter(ref_words) & Counter(pred_words)
    num_same = sum(common.values())
    
    precision = num_same / len(pred_words) if pred_words else 0
    recall = num_same / len(ref_words) if ref_words else 0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1
```

**报告格式**: 百分比 (F1 × 100)

**示例**:
| Reference | Prediction | Precision | Recall | F1 |
|-----------|------------|-----------|--------|-----|
| "35 to 39 years" | "35 to 39 years" | 1.0 | 1.0 | 100% |
| "35 to 39 years" | "35 to 44 years" | 0.6 | 0.6 | 60% |
| "apple banana" | "banana cherry" | 0.5 | 0.5 | 50% |

---

#### ROUGE-L

**定义**: 基于**最长公共子序列 (LCS)** 的 ROUGE 分数

```python
def rouge_l(reference, prediction):
    ref_words = normalize_answer(reference).split()
    pred_words = normalize_answer(prediction).split()
    
    lcs_len = lcs_length(ref_words, pred_words)
    
    precision = lcs_len / len(pred_words) if pred_words else 0
    recall = lcs_len / len(ref_words) if ref_words else 0
    
    rouge_l = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return rouge_l
```

**报告格式**: 百分比 (ROUGE-L × 100)

**与 F1 的区别**: ROUGE-L 考虑词序（通过 LCS），F1 只看词袋重合

---

#### SacreBLEU

**定义**: 使用 `evaluate` 库的标准 SacreBLEU 实现，4-gram BLEU + brevity penalty

**评分范围**: 0-100

**配置**: 默认参数（tokenize='13a'）

---

### Chart Generation Metrics (Visualization Task)

#### ECR (Executable Code Rate / 代码可执行率)

**定义**: 生成的代码能否**无错误执行**

```
ECR = True   # 代码执行成功（无异常）
ECR = False  # 代码执行失败（抛出任何异常）
```

**计算公式**:

$$ECR = \frac{\text{成功执行的样本数}}{\text{总样本数}} \times 100\%$$

**判定条件**:

| 情况 | ECR 值 | 说明 |
|------|--------|------|
| 代码正常执行完成 | `True` | 包括输出警告但不报错 |
| 语法错误 (SyntaxError) | `False` | 代码无法解析 |
| 运行时错误 (RuntimeError) | `False` | 除零、索引越界等 |
| 导入失败 (ImportError) | `False` | 缺少依赖包 |
| 文件未找到 (FileNotFoundError) | `False` | Excel 路径错误 |
| 超时 (15秒) | `False` | 代码执行时间过长 |

**⚠️ 安全注意**: 代码执行使用 Python `exec()` 直接运行，**无沙盒隔离**

> 原始项目未实现或提及任何安全隔离机制，仅依赖 15 秒超时保护。

---

#### Pass@1 (数据正确率)

**定义**: 生成图表的**数据值**是否与标准答案**完全匹配**

**判定条件**:

| ECR | Y值匹配 | Pass@1 |
|-----|---------|--------|
| True | 匹配 | `True` |
| True | 不匹配 | `False` |
| False | — | `None` (不参与计算) |

**计算公式**:

$$Pass@1 = \frac{\text{Pass=True 的样本数}}{\text{总样本数}} \times 100\%$$

> **重要**: 分母是**所有样本数**，不是仅成功执行的样本数。`Pass=None` 的样本视为失败。

---

#### Y 值提取与对比流程

**Step 1: 代码提取**

从模型输出中用正则表达式提取 Python 代码：

```python
pattern1 = r"import pandas as pd.*?plt\.show\(\)"
pattern2 = r"import matplotlib.pyplot as plt.*?plt\.show\(\)"
```

**Step 2: 路径替换**

将 `table.xlsx` 替换为实际文件路径：
```python
code = code.replace("table.xlsx", f"data/tables/{filename}.xlsx")
```

**Step 3: 代码执行**

```python
@timeout(15)  # 15秒超时
def execute(code):
    exec(code)  # 直接执行，无沙盒
```

**Step 4: Y 值提取**

根据图表类型，从 matplotlib 对象中提取 Y 轴数据：

| 图表类型 | 提取方法 | 返回格式 |
|----------|----------|----------|
| **LineChart** | `line.get_ydata()` | `[[series1], [series2], ...]` |
| **BarChart** | `patch.get_height()` | `[bar1, bar2, bar3, ...]` |
| **PieChart** | `(theta2-theta1)/360` | `[0.25, 0.35, 0.40]` (比例) |
| **ScatterChart** | `collection.get_offsets()[:,1]` | `[[y_values]]` |

```python
def get_bar_y_predictions(plt):
    return [patch.get_height() for patch in plt.gca().patches]

def get_line_y_predictions(plt):
    return [list(line.get_ydata()) for line in plt.gca().get_lines()]

def get_pie_y_predictions(plt):
    return [round((p.theta2 - p.theta1) / 360.0, 2) for p in plt.gca().patches]

def get_scatter_y_predictions(plt):
    return [[item[1] for item in coll.get_offsets()] for coll in plt.gca().collections]
```

**Step 5: 数值对比**

```python
def std_digit(values):
    """四舍五入到2位小数"""
    return [round(x, 2) for x in values]

def compare(list1, list2):
    """排序后逐一精确比较"""
    list1.sort()
    list2.sort()
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if list1[i] != list2[i]:  # 精确匹配，无容差
            return False
    return True
```

**PieChart 特殊处理**:

饼图的 Ground Truth 是原始数值，需先归一化为比例：

```python
def compute_pie_chart_metric(references, predictions):
    # 归一化为比例 (sum = 1.0)
    total = sum(references)
    normalized_refs = [round(r / total, 2) for r in references]
    # 然后比较
    return compare(normalized_refs, predictions)
```

---

## Code Execution Flow (Visualization Task)

### 完整评测流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      模型生成原始输出                                 │
│  "Here's the code:\n```python\nimport pandas as pd..."             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: 正则提取代码                                                │
│  pattern = r"import pandas as pd.*?plt\.show\(\)"                   │
│  → 得到纯 Python 代码                                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: 路径替换                                                    │
│  "table.xlsx" → "/path/to/data/tables/employment-table02.xlsx"      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: 包装为 if __name__ == '__main__':                          │
│  (防止模块级代码意外执行)                                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: 执行代码 (15秒超时)                                         │
│                                                                      │
│  try:                                                                │
│      exec(python_code)  # ⚠️ 无沙盒！                                │
│      ECR = True                                                      │
│  except Exception:                                                   │
│      ECR = False                                                     │
│      Pass = None  # 跳过数据比较                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                      ┌─────────────┴─────────────┐
                      │                           │
                ECR = False                   ECR = True
                      │                           │
                      ▼                           ▼
            ┌─────────────────┐     ┌─────────────────────────────────┐
            │ Pass = None     │     │ Step 5: Y值提取                  │
            │ (统计为失败)     │     │ get_bar_y_predictions(plt)      │
            └─────────────────┘     │ get_line_y_predictions(plt)     │
                                    │ ...                              │
                                    └─────────────────────────────────┘
                                                  │
                                                  ▼
                                    ┌─────────────────────────────────┐
                                    │ Step 6: 数值标准化               │
                                    │ - 四舍五入到2位小数              │
                                    │ - 展平嵌套列表                   │
                                    │ - PieChart: 归一化为比例         │
                                    └─────────────────────────────────┘
                                                  │
                                                  ▼
                                    ┌─────────────────────────────────┐
                                    │ Step 7: 排序后精确比较           │
                                    │                                  │
                                    │ pred = sorted(std_digit(pred))  │
                                    │ ref = sorted(std_digit(ref))    │
                                    │                                  │
                                    │ Pass = (pred == ref)            │
                                    └─────────────────────────────────┘
                                                  │
                                    ┌─────────────┴─────────────┐
                                    │                           │
                              匹配成功                        匹配失败
                                    │                           │
                                    ▼                           ▼
                            ┌───────────┐               ┌───────────┐
                            │ Pass=True │               │ Pass=False│
                            └───────────┘               └───────────┘
```

### 安全警告

⚠️ **代码直接使用 `exec()` 执行，无沙盒隔离**

> **重要说明**: 原始项目代码**未提及或实现任何沙盒/安全隔离机制**。这是该项目在 Chart Generation 评测中的已知特性（设计特点），并非 bug 或疏漏。项目代码直接使用 Python `exec()` 执行生成的代码，仅依赖 15 秒超时保护。

- 不要在生产环境运行未经审查的代码
- 仅用于本地评测可信模型
- 建议在隔离的虚拟环境或容器中运行

### 超时机制

| 阶段 | 超时时间 |
|------|----------|
| 单次代码执行 | 15 秒 |
| 完整评测函数 | 20 秒 |

### 错误处理

| 异常类型 | 处理方式 | 结果 |
|----------|----------|------|
| `SyntaxError` | 捕获并记录 | ECR=False, Pass=None |
| `NameError` | 捕获并记录 | ECR=False, Pass=None |
| `FileNotFoundError` | 捕获并记录 | ECR=False, Pass=None |
| `TimeoutError` | 强制终止 | ECR=False, Pass=None |
| Y值提取失败 | 捕获并记录 | ECR=True, Pass=False |

---

## Quick Start

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

从 [Hugging Face](https://huggingface.co/datasets/spzy/RealHiTBench) 下载到 `data/` 目录

### 3. 运行本地推理 (Qwen3-VL 示例)

```bash
# Image-only 模态
CUDA_VISIBLE_DEVICES=0,1,2 python inference/inference_qwen3vl_local.py \
  --model_dir /path/to/Qwen3-VL-8B-Instruct \
  --modality image \
  --batch_size 3

# Text-only 模态 (LaTeX格式)
CUDA_VISIBLE_DEVICES=0,1,2 python inference/inference_qwen3vl_local.py \
  --model_dir /path/to/Qwen3-VL-8B-Instruct \
  --modality text \
  --format latex \
  --batch_size 3

# Mix 模态 (Image + Text)
CUDA_VISIBLE_DEVICES=0,1,2 python inference/inference_qwen3vl_local.py \
  --model_dir /path/to/Qwen3-VL-8B-Instruct \
  --modality mix \
  --format latex \
  --batch_size 3
```

### 4. 查看评测结果

```bash
# 可视化聚合指标
python utils/visualize_aggregate_metrics.py result/qwen3vl_local/*/results_*.json

# 重新计算已有 checkpoint 的指标 (修复 Pass/ECR 统计)
python utils/recompute_aggregate_metrics.py --recursive result/
```

详细命令参考: [inference/COMMANDS_REFERENCE.md](inference/COMMANDS_REFERENCE.md)

---

## Aggregate Metrics Calculation

最终指标按 `QuestionType` 分组聚合：

```python
# 数值指标 (F1, EM, ROUGE-L, SacreBLEU, GPT_EVAL)
aggregate[metric] = mean(values)

# 布尔指标 (Pass, ECR)
aggregate[metric] = count(True) / total_samples
# 注: Pass=None 视为 False，计入分母
```

---

## Project Structure

```
RealHiTBench/
├── data/                          # 数据集
│   ├── QA_final.json              # 主数据集
│   ├── image/                     # PNG 图片
│   ├── latex/html/csv/markdown/   # 文本格式表格
│   └── tables/                    # Excel (Visualization用)
├── inference/                     # 推理脚本
│   ├── inference_qwen3vl_local.py # 本地 Qwen3-VL
│   ├── inference_llm.py           # 开源 LLM
│   ├── inference_mlm.py           # 开源 MLM (图片)
│   ├── inference_mix.py           # MLM (图片+文本)
│   ├── inference_close.py         # 闭源 API
│   ├── qa_metrics_simple.py       # QA 指标计算
│   ├── answer_prompt_mlm.py       # Prompt 模板
│   └── COMMANDS_REFERENCE.md      # 命令参考
├── utils/                         # 工具函数
│   ├── chart_metric_util.py       # 图表 Y值提取/对比
│   ├── chart_process.py           # 代码执行流程
│   ├── recompute_aggregate_metrics.py  # 重算聚合指标
│   └── visualize_aggregate_metrics.py  # 结果可视化
├── metrics/                       # 评测指标实现
├── result/                        # 输出结果
└── requirements.txt
```

---

## Data Format Specification

### QA_final.json 结构

```json
{
  "queries": [
    {
      "id": 1,
      "FileName": "employment-table01",
      "CompStrucCata": "ColumnHeaderMerge",
      "Source": "Bureau of Labor Statistics",
      "Question": "Match the year where...",
      "QuestionType": "Fact Checking",
      "SubQType": "Multi-hop Fact Checking",
      "COT": [
        {"planning": "First, we need to..."},
        {"planning": "Next, we identify..."}
      ],
      "FinalAnswer": "1955, 62170",
      "ProcessedAnswer": "1955, 62170"
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 描述 | 评测使用 |
|------|------|------|----------|
| `id` | int | 唯一标识符 | 追踪用 |
| `FileName` | string | 表格文件名（无后缀） | ✅ 定位文件 |
| `CompStrucCata` | string | 复杂结构类别 | 分析用 |
| `Source` | string | 数据来源 | 元数据 |
| `Question` | string | 问题文本 | ✅ 输入 |
| `QuestionType` | string | 主任务类型 | ✅ 评测分组 |
| `SubQType` | string | 子任务类型 | ✅ Visualization 需要 |
| `COT` | array | 人工标注的推理步骤 | ❌ 不发送给模型 |
| `FinalAnswer` | string | 原始标准答案 | Visualization: 完整代码 |
| `ProcessedAnswer` | string | 处理后的答案 | ✅ 评测基准 |

---

## Citation

```bibtex
@misc{wu2025realhitbenchcomprehensiverealistichierarchical,
      title={RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis}, 
      author={Pengzuo Wu and Yuhang Yang and Guangcheng Zhu and Chao Ye and Hong Gu and Xu Lu and Ruixuan Xiao and Bowen Bao and Yijing He and Liangyu Zha and Wentao Ye and Junbo Zhao and Haobo Wang},
      year={2025},
      eprint={2506.13405},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.13405}, 
}
```

---

## License

- **Code**: MIT License
- **Data**: CC-BY-NC-4.0
