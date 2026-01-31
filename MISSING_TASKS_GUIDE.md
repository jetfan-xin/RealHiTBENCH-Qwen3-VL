# 缺失任务分析与重新推理 - 完整流程

## 📋 总体概述

本系统提供完整的工作流程，用于：
1. **分析**所有results.json中的缺失任务
2. **识别**缺失原因（文件依赖、OOM错误、未完成运行）
3. **生成**针对性的重新推理脚本
4. **执行**自动化的任务补全

## 🔄 完整工作流程

```
QA_final_sc_filled.json (3,071 tasks)
         ↓
    ┌────────────────────────────────┐
    │  analyze_missing_tasks.py      │ ← 步骤1: 分析缺失任务
    └────────────────────────────────┘
         ↓
    skip_ids.json (每个结果目录一个)
         ↓
    ┌────────────────────────────────┐
    │ generate_missing_task_inf...py │ ← 步骤2: 生成推理脚本
    └────────────────────────────────┘
         ↓
    rerun_*.py (9个配置脚本)
         ↓
    ┌────────────────────────────────┐
    │  run_all_missing_tasks.sh      │ ← 步骤3: 执行推理
    └────────────────────────────────┘
         ↓
    更新的results.json (缺失任务补全)
```

## 📁 文件结构

```
RealHiTBENCH-Qwen3-VL/
├── data/
│   └── QA_final_sc_filled.json          # 主任务列表 (3,071 tasks)
│
├── result/complied/                      # 结果目录
│   ├── qwen3vl_default_pic/
│   │   ├── image/
│   │   │   ├── results.json             # 原始结果
│   │   │   └── skip_ids.json            # 缺失任务分析 ← 步骤1生成
│   │   ├── mix_html/
│   │   │   ├── results.json
│   │   │   └── skip_ids.json
│   │   └── ...
│   ├── qwen3vl_resize_pic/
│   └── qwen3vl_text/
│
├── utils/
│   ├── analyze_missing_tasks.py         # 步骤1: 分析脚本
│   ├── generate_missing_task_inference.py  # 步骤2: 生成脚本
│   └── missing_tasks_summary.json       # 总体统计
│
└── inference/
    ├── inference_qwen3vl_local_a100_truncate.py
    ├── inference_qwen3vl_local_a100_truncate_with_task_ids.py  # Wrapper支持task_ids
    │
    └── rerun_missing_tasks/             # 步骤2生成的目录
        ├── README.md                    # 详细使用指南
        ├── run_all_missing_tasks.sh    # 主运行脚本
        ├── missing_tasks_summary.txt   # 摘要
        └── rerun_*.py                   # 9个单独脚本
```

## 🚀 快速开始

### 完整流程（三步走）

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL

# 步骤1: 分析缺失任务
python utils/analyze_missing_tasks.py

# 步骤2: 生成推理脚本
python utils/generate_missing_task_inference.py

# 步骤3: 运行所有推理
bash inference/rerun_missing_tasks/run_all_missing_tasks.sh
```

**预计总时间**: 分析5分钟 + 推理6-8分钟 = ~10-15分钟

### 仅重新分析（不重新推理）

```bash
# 如果只想查看当前缺失情况
python utils/analyze_missing_tasks.py

# 查看总体报告
cat utils/missing_tasks_summary.json | jq .

# 查看特定配置的缺失任务
cat result/complied/qwen3vl_text/text_html/skip_ids.json | jq .
```

## 📊 当前统计（初始分析）

### 总体情况
- **总任务数**: 3,071
- **结果文件**: 10
- **平均完成率**: 99.73%
- **总缺失任务**: 84 (跨所有10个文件)
- **总错误任务**: 44 (OOM等)

### 缺失原因分布
- **文件依赖问题**: 不处理（源文件不存在）
- **Incomplete runs**: 45 tasks（文件存在但未处理）
- **Error tasks**: 44 tasks（OOM错误）
- **需要重新运行**: 89 tasks

### 重点配置

| 配置 | 缺失数 | 错误数 | 优先级 |
|------|--------|--------|--------|
| qwen3vl_text/text_html | 8 | 17 | 🔥 高 |
| qwen3vl_default_pic/mix_html | 8 | 17 | 🔥 高 |
| qwen3vl_resize_pic/mix_html | 33 | 0 | 🔥 高 |
| qwen3vl_text/text_csv | 18 | 0 | ⚠️ 中 |
| qwen3vl_resize_pic/mix_csv | 4 | 10 | ⚠️ 中 |
| 其他配置 | 1-4 | 0 | ✅ 低 |

## 🔧 详细使用说明

### 步骤1: 分析缺失任务

**工具**: `utils/analyze_missing_tasks.py`

**功能**:
- 加载QA_final_sc_filled.json (3,071任务)
- 扫描所有results.json文件
- 识别缺失的任务ID
- 分析缺失原因（文件依赖vs处理错误）
- 生成skip_ids.json (每个结果目录)
- 生成总体报告missing_tasks_summary.json

**输出文件**:
```
result/complied/*/skip_ids.json           # 每个配置一个
utils/missing_tasks_summary.json          # 总体报告
```

**skip_ids.json结构**:
```json
{
  "metadata": {
    "result_file": "qwen3vl_text/text_html/results.json",
    "config": "qwen3vl_text",
    "modality": "text",
    "format": "html"
  },
  "statistics": {
    "total_tasks": 3071,
    "completed": 3063,
    "success": 3046,
    "error": 17,
    "missing": 8
  },
  "skip_ids": [2216, 2217, ...],        # 所有缺失的ID
  "error_ids": [2747, 2748, ...],       # 有ERROR的ID
  "skip_reasons": {
    "2216": "Missing source file(s): labor-table68.html",
    "2747": "OOM error or processing failure"
  },
  "categorized": {
    "file_dependency_issues": [...],    # 文件不存在（不重新运行）
    "incomplete_runs": [...]            # 未完成（需要重新运行）
  }
}
```

### 步骤2: 生成推理脚本

**工具**: `utils/generate_missing_task_inference.py`

**功能**:
- 读取所有skip_ids.json
- 提取需要重新运行的任务（排除file_dependency_issues）
- 为每个配置生成专门的Python脚本
- 自动选择合适的inference脚本（truncate vs default）
- 生成主运行脚本run_all_missing_tasks.sh

**输出文件**:
```
inference/rerun_missing_tasks/
├── README.md                           # 详细使用指南
├── run_all_missing_tasks.sh           # 主脚本
├── missing_tasks_summary.txt          # 摘要
└── rerun_*.py                          # 9个单独脚本
```

**脚本特性**:
- ✅ 自动task_ids过滤
- ✅ 自动OOM防护（text/mix使用truncate）
- ✅ Resume模式（合并到现有结果）
- ✅ batch_size=1（避免OOM）

### 步骤3: 执行推理

**主脚本**: `inference/rerun_missing_tasks/run_all_missing_tasks.sh`

**运行方式**:
```bash
# 方式1: 运行所有（推荐）
bash inference/rerun_missing_tasks/run_all_missing_tasks.sh

# 方式2: 单独运行特定配置
python inference/rerun_missing_tasks/rerun_qwen3vl_text_text_html.py

# 方式3: 并行运行（多GPU）
CUDA_VISIBLE_DEVICES=0 python rerun_qwen3vl_text_text_html.py &
CUDA_VISIBLE_DEVICES=1 python rerun_qwen3vl_default_pic_mix_html.py &
```

## 🛡️ OOM防护机制

### 自动检测与应用
脚本自动判断是否需要OOM防护：

**需要truncate的情况**:
- 有error_ids（之前出现过OOM错误）
- modality为text或mix（可能包含超大HTML）

**使用的脚本**:
```
inference_qwen3vl_local_a100_truncate_with_task_ids.py
```

### 文本截断参数
- **MAX_INPUT_TOKENS**: 100,000
- **截断策略**: 保留90%目标长度
- **适用文件**: HTML表格文本（csv/latex不受影响）

### 已知OOM样本
| Task ID | 文件名 | 原因 | 解决方案 |
|---------|--------|------|----------|
| 2747-2749 | economy-table14_swap | 1.2MB HTML, 334K tokens | 截断到100K |
| 2750-2751 | society-table02_swap | 缺HTML文件 | 已排除（文件依赖） |
| 2758-2763 | ... | 大型HTML表格 | 截断到100K |

## 📈 验证结果

### 运行后验证

```bash
# 重新分析缺失任务
python utils/analyze_missing_tasks.py

# 比较前后变化
diff -u utils/missing_tasks_summary.json.old utils/missing_tasks_summary.json

# 检查特定配置
cat result/complied/qwen3vl_text/text_html/skip_ids.json | jq '.statistics'
```

**期望结果**:
```json
{
  "statistics": {
    "total_tasks": 3071,
    "completed": 3071,      # 应该是3071
    "success": 3071,        # 应该是3071
    "error": 0,             # 应该是0
    "missing": 0            # 应该是0
  }
}
```

### 检查结果文件

```bash
# 查看结果总数
cat result/complied/qwen3vl_text/text_html/results.json | jq '.results | length'

# 检查是否有ERROR
cat result/complied/qwen3vl_text/text_html/results.json | jq '.results[] | select(.Prediction | startswith("[ERROR]"))'
```

## 🔍 故障排除

### 问题1: 分析脚本报错

**错误**: `FileNotFoundError: QA_final_sc_filled.json`

**解决**:
```bash
# 检查文件路径
ls -la /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data/QA_final_sc_filled.json

# 修改脚本中的路径
vim utils/analyze_missing_tasks.py
```

### 问题2: CUDA OOM仍然发生

**检查**:
```bash
# 确认使用了truncate脚本
grep "inference_qwen3vl_local_a100_truncate" inference/rerun_missing_tasks/rerun_*.py

# 确认batch_size=1
grep "batch_size" inference/rerun_missing_tasks/rerun_*.py
```

**临时解决**:
- 降低MAX_INPUT_TOKENS（需修改inference_qwen3vl_local_a100_truncate.py）
- 使用更小的GPU批次
- 单独运行OOM样本

### 问题3: 任务仍然缺失

**可能原因**:
1. 文件依赖问题（源文件真的不存在）
2. 新的处理错误
3. Resume没有正确合并

**检查方法**:
```bash
# 查看详细日志
python inference/rerun_missing_tasks/rerun_qwen3vl_text_text_html.py 2>&1 | tee rerun.log

# 检查checkpoint
cat result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_text_html_default/checkpoint.json | jq '.processed_ids | length'
```

## 📚 相关文档

- [inference/rerun_missing_tasks/README.md](inference/rerun_missing_tasks/README.md) - 推理脚本详细说明
- [inference/README_TRUNCATION.md](inference/README_TRUNCATION.md) - 文本截断机制
- [inference/COMMANDS_REFERENCE.md](inference/COMMANDS_REFERENCE.md) - 推理命令参考

## 🔄 更新与维护

### 重新生成脚本

如果结果发生变化（例如手动修复了一些任务），重新生成脚本：

```bash
# 删除旧脚本
rm -rf inference/rerun_missing_tasks

# 重新分析
python utils/analyze_missing_tasks.py

# 重新生成
python utils/generate_missing_task_inference.py
```

### 定期检查

建议在以下情况重新分析：
- 添加新的results.json文件
- 手动修复了一些任务
- 更新了QA_final_sc_filled.json
- 完成了推理后想验证结果

## 🎯 最佳实践

### 1. 逐步验证
```bash
# 步骤1: 分析
python utils/analyze_missing_tasks.py
# → 查看 utils/missing_tasks_summary.json

# 步骤2: 生成脚本
python utils/generate_missing_task_inference.py
# → 查看 inference/rerun_missing_tasks/README.md

# 步骤3: 先测试单个配置
python inference/rerun_missing_tasks/rerun_qwen3vl_text_text_latex.py  # 只有1个任务
# → 验证是否成功

# 步骤4: 运行所有
bash inference/rerun_missing_tasks/run_all_missing_tasks.sh
```

### 2. 优先级运行

如果时间有限，按优先级运行：

```bash
# 高优先级: OOM错误任务（需要truncate）
python rerun_qwen3vl_text_text_html.py           # 17 tasks
python rerun_qwen3vl_default_pic_mix_html.py     # 17 tasks

# 中优先级: 大量未完成任务
python rerun_qwen3vl_resize_pic_mix_html.py      # 25 tasks
python rerun_qwen3vl_text_text_csv.py            # 15 tasks

# 低优先级: 少量任务（快速完成）
python rerun_qwen3vl_*_latex.py                  # 各1 task
python rerun_qwen3vl_*_image_*.py                # 各1 task
```

### 3. 监控与日志

```bash
# 保存完整日志
bash inference/rerun_missing_tasks/run_all_missing_tasks.sh 2>&1 | tee full_rerun.log

# 实时监控进度
tail -f full_rerun.log

# 检查GPU使用
watch -n 1 nvidia-smi
```

## 📞 支持

如有问题，请检查：
1. [inference/rerun_missing_tasks/README.md](inference/rerun_missing_tasks/README.md) - 推理脚本详细文档
2. 生成的日志文件
3. skip_ids.json中的详细错误信息

---

**最后更新**: 2026-01-31  
**版本**: 1.0  
**维护**: 自动生成系统
