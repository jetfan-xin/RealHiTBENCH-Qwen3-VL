# RealHiTBench 结果可视化指南

## 概述

`result_visualization.ipynb` 是一个用于可视化 RealHiTBench 项目结果的 Jupyter Notebook，支持多个模态的数据分析和图表生成。

## 功能特性

该notebook包含以下功能：

### 1. **数据加载与统计**
   - 从 `/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/result/complied/qwen3vl_default_pic/` 目录加载各个模态的数据
   - 支持的模态：`mix_html`, `mix_json`, `mix_latex`, `mix_markdown`, `image`
   - 自动计算各任务各指标的平均得分

### 2. **图表类型**

为每个模态生成以下类型的图表，保存到相应目录：

#### 2.1 **任务平均得分柱状图** (`{modality}_task_comparison.png`)
   - 展示各任务（Fact Checking、Numerical Reasoning等）的平均得分
   - 包含总体平均得分（Overall）
   - 彩色标记不同任务

#### 2.2 **聚合指标热力图** (`{modality}_aggregate_heatmap.png`)
   - 展示所有任务和所有指标的得分热力分布
   - 颜色深度表示得分高低
   - 包含具体数值标注

#### 2.3 **域名难度分析图** (`{modality}_domain_analysis.png`)
   - 从FileName中自动提取数据域名
   - 按得分排序展示各域名的难度
   - 颜色渐变表示难度（绿=简单，红=困难）

#### 2.4 **单表vs多表对比图** (`{modality}_single_vs_multi_table.png`)
   - 对比单表数据和多表数据在各任务上的表现
   - 基于 `CompStrucCata` 列分类
   - 如果数据不包含此列，则自动跳过

### 3. **汇总统计**
   - 生成 `modality_summary.csv` 文件，汇总所有模态的指标
   - 打印各任务的最佳模态排名
   - 支持跨模态比对分析

## 目录结构

```
result/complied/
├── qwen3vl_default_pic/
│   ├── mix_html/
│   │   ├── mix_html_task_comparison.png
│   │   ├── mix_html_aggregate_heatmap.png
│   │   ├── mix_html_domain_analysis.png
│   │   └── mix_html_single_vs_multi_table.png
│   ├── mix_json/
│   │   └── [同上]
│   ├── mix_latex/
│   │   └── [同上]
│   ├── mix_markdown/
│   │   └── [同上]
│   ├── image/
│   │   └── [同上]
│   └── modality_summary.csv
```

## 使用方法

### 方法 1：在 VS Code 中运行

1. 打开文件：`utils/result_visualization.ipynb`
2. 确保已安装必要的库（pandas, matplotlib, numpy）
3. 依次执行每个单元格（Ctrl+Enter）
4. 或者点击"Run All"运行整个notebook

### 方法 2：在命令行中运行

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL
jupyter nbconvert --to notebook --execute utils/result_visualization.ipynb
```

### 方法 3：转换为Python脚本后运行

```bash
# 如果需要，可以将notebook转换为Python脚本
jupyter nbconvert --to script utils/result_visualization.ipynb
python utils/result_visualization.py
```

## 数据说明

### 支持的任务类型

- **Fact Checking**：事实核查（指标：F1, EM）
- **Numerical Reasoning**：数值推理（指标：F1, EM）
- **Structure Comprehending**：结构理解（指标：F1, EM）
- **Data Analysis**：数据分析（指标：ROUGE-L, F1, EM）
- **Visualization**：可视化（指标：ECR, Pass）

### 指标说明

- **F1**：F1分数（0-100）
- **EM**：完全匹配率（0-100）
- **ROUGE-L**：ROUGE-L分数（0-100）
- **Pass**：通过率（百分比）
- **ECR**：元素匹配率（百分比）

## 输出说明

### 控制台输出
- 显示数据加载进度
- 显示每个模态的统计信息
- 显示各任务的最佳模态排名

### 图表输出
- 所有图表以PNG格式保存，分辨率为300 dpi
- 每个模态的图表保存在该模态的目录下
- 图表包含标题、标签、图例和数值标注

### CSV输出
- `modality_summary.csv`：所有模态的汇总数据
- 行为各模态，列为各任务和Overall

## 自定义

### 修改模态列表

编辑第一个代码单元格中的 `SUPPORTED_MODALITIES`：

```python
SUPPORTED_MODALITIES = ['mix_html', 'mix_json', 'mix_latex', 'mix_markdown', 'image']
```

### 修改任务和指标

编辑 `TASK_METRICS` 字典：

```python
TASK_METRICS = {
    'Fact Checking': ['F1', 'EM'],
    # ... 其他任务
}
```

### 修改图表样式

在各图表生成函数中调整：
- 颜色：`color='#XXXXXX'`
- 字体大小：`fontsize=XX`
- 图表尺寸：`figsize=(width, height)`
- 分辨率：`dpi=XXX`

## 常见问题

### Q1：为什么某些模态的数据为空？
A：检查 `result/complied/qwen3vl_default_pic/{modality}/results.json` 文件是否存在且包含有效数据。

### Q2：如何只生成特定模态的图表？
A：修改第四部分的代码，将 `for modality, df in modality_data.items():` 改为循环特定模态。

### Q3：单表vs多表对比图没有显示？
A：检查数据中是否包含 `CompStrucCata` 或 `Metrics.CompStrucCata` 列。

### Q4：如何修改输出目录？
A：修改 `MODEL_DIR` 的定义（第一个代码单元格）。

## 依赖库

- `pandas`：数据处理
- `matplotlib`：绘图
- `numpy`：数值计算
- `json`：JSON文件读取
- `pathlib`：路径处理

## 版本信息

- 创建日期：2026年2月
- 支持 Python 3.8+
- 适配 Jupyter Notebook / JupyterLab

## 扩展功能建议

1. 添加交互式图表（使用 Plotly）
2. 生成PDF格式的报告
3. 支持时间序列分析（如果有模型训练过程数据）
4. 添加统计测试（如 T-test, ANOVA）
5. 生成自动化的分析报告

---

**更多帮助**：编辑 `result_visualization.ipynb` 的各个单元格以查看详细的执行步骤和输出。
