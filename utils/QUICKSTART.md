# 快速开始指南 - 结果可视化

## 简介

`result_visualization.ipynb` 是一个Jupyter Notebook，可以自动从你的RealHiTBench结果生成各种分析图表。

## 快速步骤

### 1. 打开Notebook

```bash
# 方式一：直接在VS Code中打开
# 在VS Code中导航到: utils/result_visualization.ipynb

# 方式二：通过Jupyter Lab
jupyter lab utils/result_visualization.ipynb
```

### 2. 运行Notebook

**选项A：逐单元执行（推荐）**
- 点击每个单元格，按 `Ctrl+Enter` 执行
- 这样可以查看每步的进度和输出

**选项B：一次性全部执行**
- 在菜单中选择 "Run All Cells"
- 或使用快捷键 `Ctrl+Shift+Enter`

### 3. 查看结果

所有生成的图表会保存到：
```
/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/result/complied/qwen3vl_default_pic/
```

每个模态有一个子目录，包含该模态的所有图表：
- `mix_html/`
- `mix_json/`
- `mix_latex/`
- `mix_markdown/`
- `image/`

## 生成的文件

### 对于每个模态，会生成以下图表：

1. **{modality}_task_comparison.png**
   - 各任务平均得分的柱状图
   - 包含Overall总体得分

2. **{modality}_aggregate_heatmap.png**
   - 所有任务和指标的热力图
   - 可视化性能分布

3. **{modality}_domain_analysis.png**
   - 按数据域名分析难度
   - 帮助识别具体挑战

4. **{modality}_single_vs_multi_table.png**
   - 对比单表和多表数据表现
   - 如果数据包含此信息

### 全局文件：

- **modality_summary.csv**
  - CSV格式的汇总表
  - 包含所有模态的所有指标
  - 方便在Excel中进一步分析

## 常见命令

### 仅测试是否能运行
```bash
python utils/test_visualization.py
```

### 转换为Python脚本
```bash
jupyter nbconvert --to script utils/result_visualization.ipynb
python utils/result_visualization.py
```

### 转换为HTML报告
```bash
jupyter nbconvert --to html utils/result_visualization.ipynb
```

## 输出示例

运行时会输出类似：
```
================================================================================
项目根目录: /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL

【MIX_HTML】统计
  Fact Checking (样本数: 1214)
    F1: 59.01
    EM: 50.33
  ...
  总平均得分: 48.54
```

## 自定义

### 修改支持的模态

编辑第一个代码单元格：
```python
SUPPORTED_MODALITIES = ['mix_html', 'mix_json', 'mix_latex', 'mix_markdown', 'image']
```

### 修改任务定义

编辑TASK_METRICS字典来改变任务或指标：
```python
TASK_METRICS = {
    'Fact Checking': ['F1', 'EM'],
    # 添加更多任务...
}
```

### 修改图表样式

在各图表生成函数中调整参数：
```python
fig, ax = plt.subplots(figsize=(14, 8))  # 修改尺寸
ax.set_title(..., fontsize=16)            # 修改字体大小
bars = ax.bar(..., color='#4C78A8')       # 修改颜色
```

## 故障排除

### Q: 提示模块找不到？
A: 确保在项目根目录运行notebook，或修改第一个单元格的路径。

### Q: 某些模态没有数据？
A: 检查 `result/complied/qwen3vl_default_pic/{modality}/results.json` 是否存在。

### Q: 图表没有保存？
A: 检查输出目录是否存在或有写入权限。

### Q: 出现MemoryError？
A: 数据可能太大，尝试注释掉某些不需要的模态。

## 相关文件

- `VISUALIZATION_GUIDE.md` - 详细的使用文档
- `test_visualization.py` - 快速测试脚本
- `result_visualization.ipynb` - 主体notebook
- `visualize_aggregate_metrics.py` - 使用的工具函数库

## 下一步

1. 运行notebook生成初始图表
2. 查看 `modality_summary.csv` 了解各模态表现
3. 根据需要修改图表样式或添加新分析
4. 将结果导出为PDF或HTML用于报告

---

**提示**：如果遇到问题，先运行 `python utils/test_visualization.py` 来诊断环境是否正确配置。
