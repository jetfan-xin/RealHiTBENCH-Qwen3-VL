# 缺失任务重新推理指南

## 概述

本目录包含用于重新运行缺失任务的自动生成脚本。这些脚本基于 `skip_ids.json` 分析结果生成，专门处理因OOM错误或未完成运行而缺失的任务。

## 生成的文件

### 1. 单独的推理脚本（9个）
每个配置一个脚本，仅处理该配置中缺失的任务：

- `rerun_qwen3vl_default_pic_image_None.py` (1 tasks)
- `rerun_qwen3vl_default_pic_mix_html.py` (17 tasks - OOM错误)
- `rerun_qwen3vl_resize_pic_image_None.py` (1 tasks)
- `rerun_qwen3vl_resize_pic_mix_csv.py` (11 tasks)
- `rerun_qwen3vl_resize_pic_mix_html.py` (25 tasks - 未完成运行)
- `rerun_qwen3vl_resize_pic_mix_latex.py` (1 tasks)
- `rerun_qwen3vl_text_text_csv.py` (15 tasks)
- `rerun_qwen3vl_text_text_html.py` (17 tasks - OOM错误)
- `rerun_qwen3vl_text_text_latex.py` (1 tasks)

### 2. 主运行脚本
- `run_all_missing_tasks.sh` - 按顺序运行所有9个配置

### 3. 摘要文件
- `missing_tasks_summary.txt` - 详细统计信息

## 关键特性

### 自动OOM防护
脚本自动检测OOM风险并应用相应策略：

- **有OOM错误的配置**（text_html, mix_html, mix_csv）：
  - 使用 `inference_qwen3vl_local_a100_truncate_with_task_ids.py`
  - 启用文本截断（MAX_INPUT_TOKENS=100,000）
  - 防止超大HTML表格导致的内存溢出

- **无OOM风险的配置**（image, latex）：
  - 使用 `inference_qwen3vl_local_a100_default.py`
  - 标准推理流程

### 任务ID过滤
所有脚本使用 `--task_ids` 参数精确指定要处理的任务：

```python
--task_ids "2747,2748,2749,2750,2751,..."
```

这确保：
- ✅ 只处理缺失的任务
- ✅ 不重复处理已成功的任务
- ✅ 节省计算时间和资源

### Resume模式
所有脚本启用 `--resume` 参数：

- 加载已有的checkpoint文件
- 将新结果合并到现有结果中
- 不覆盖已成功的任务

## 使用方法

### 方式1：运行所有配置（推荐）

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL
bash inference/rerun_missing_tasks/run_all_missing_tasks.sh
```

**优点**：
- 一次性完成所有缺失任务
- 自动按顺序执行
- 出错时自动停止

**预计时间**：~2-3小时（89个任务总计）

### 方式2：单独运行特定配置

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL

# 示例：仅运行text_html的17个OOM任务
python inference/rerun_missing_tasks/rerun_qwen3vl_text_text_html.py

# 示例：仅运行mix_html的25个未完成任务
python inference/rerun_missing_tasks/rerun_qwen3vl_resize_pic_mix_html.py
```

**优点**：
- 可以选择性运行最重要的配置
- 方便调试和测试

### 方式3：并行运行（多GPU）

如果有多个空闲GPU，可以同时运行多个配置：

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python inference/rerun_missing_tasks/rerun_qwen3vl_text_text_html.py &

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python inference/rerun_missing_tasks/rerun_qwen3vl_default_pic_mix_html.py &

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python inference/rerun_missing_tasks/rerun_qwen3vl_resize_pic_mix_html.py &
```

**注意**：确保不要在同一GPU上运行多个推理任务！

## 任务分类统计

### 按原因分类
- **Incomplete runs**: 45 tasks（未完成运行，文件存在但未处理）
- **Error tasks**: 44 tasks（OOM错误导致失败）
- **总计**: 89 tasks

### 按配置分类

#### 优先级1：OOM错误任务（需要truncate）
| 配置 | 任务数 | 脚本 |
|------|--------|------|
| qwen3vl_text/text_html | 17 | rerun_qwen3vl_text_text_html.py |
| qwen3vl_default_pic/mix_html | 17 | rerun_qwen3vl_default_pic_mix_html.py |
| qwen3vl_resize_pic/mix_csv | 11 (10 errors) | rerun_qwen3vl_resize_pic_mix_csv.py |

#### 优先级2：未完成运行（可能的OOM或中断）
| 配置 | 任务数 | 脚本 |
|------|--------|------|
| qwen3vl_resize_pic/mix_html | 25 | rerun_qwen3vl_resize_pic_mix_html.py |
| qwen3vl_text/text_csv | 15 | rerun_qwen3vl_text_text_csv.py |

#### 优先级3：少量任务（快速完成）
| 配置 | 任务数 | 脚本 |
|------|--------|------|
| qwen3vl_default_pic/image | 1 | rerun_qwen3vl_default_pic_image_None.py |
| qwen3vl_resize_pic/image | 1 | rerun_qwen3vl_resize_pic_image_None.py |
| qwen3vl_resize_pic/mix_latex | 1 | rerun_qwen3vl_resize_pic_mix_latex.py |
| qwen3vl_text/text_latex | 1 | rerun_qwen3vl_text_text_latex.py |

## OOM防护机制

### 文本截断策略
- **阈值**: MAX_INPUT_TOKENS = 100,000
- **机制**: 超限文本自动截断到安全范围（90%目标长度）
- **适用**: text_html、mix_html、mix_csv等包含超大HTML的任务

### 已知OOM样本
基于之前的分析，以下任务已知会导致OOM：

- **economy-table14_swap** (IDs: 2747-2749)
  - 原因: 1.2MB HTML文件，334K tokens
  - 解决: 截断到100K tokens

- **society-table02_swap** (IDs: 2750-2751)
  - 原因: 缺少HTML文件但有5.00M像素图片
  - 解决: 标记为文件依赖问题（已排除）

## 结果验证

运行完成后，验证结果：

```bash
# 重新分析缺失任务
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL
python utils/analyze_missing_tasks.py

# 检查特定配置的skip_ids.json
cat result/complied/qwen3vl_text/text_html/skip_ids.json
```

**期望结果**：
- `error_ids` 数组应该为空或显著减少
- `incomplete_runs` 数组应该为空或显著减少
- `statistics.completed` 应该接近3071

## 故障排除

### 问题1：CUDA OOM仍然发生
**解决方案**：
- 确认使用了truncate版本脚本
- 检查batch_size=1（已默认设置）
- 考虑降低MAX_INPUT_TOKENS（需修改inference脚本）

### 问题2：任务仍然缺失
**可能原因**：
- 文件依赖问题（源文件不存在）
- 新的处理错误

**检查方法**：
```bash
# 查看详细日志
python inference/rerun_missing_tasks/rerun_qwen3vl_text_text_html.py 2>&1 | tee rerun.log
```

### 问题3：Resume不工作
**检查**：
- 确认checkpoint.json存在于结果目录
- 确认使用了`--resume`参数（已默认启用）

## 时间估算

基于之前的性能数据：

| 配置类型 | 平均每任务时间 | 总任务数 | 预计时间 |
|----------|----------------|----------|----------|
| image | ~3s | 2 | ~6s |
| text_html | ~4s | 17 | ~68s (~1分钟) |
| mix_html | ~4s | 42 | ~168s (~3分钟) |
| text_csv | ~4s | 15 | ~60s (~1分钟) |
| text_latex | ~3s | 1 | ~3s |
| mix_csv | ~4s | 11 | ~44s |
| mix_latex | ~3s | 1 | ~3s |

**总计**: 约6-8分钟（顺序执行）

**注意**: OOM任务可能需要更长时间（~10-15s/任务）

## 相关文件

- `/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference/README_TRUNCATION.md` - 文本截断机制详细说明
- `/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/utils/analyze_missing_tasks.py` - 缺失任务分析工具
- `/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/utils/missing_tasks_summary.json` - 总体统计报告

## 更新脚本

如果skip_ids.json发生变化，重新生成脚本：

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL
rm -rf inference/rerun_missing_tasks
python utils/generate_missing_task_inference.py
```

---

**生成时间**: 2026-01-31  
**总任务数**: 89  
**配置数量**: 9  
**预计完成时间**: ~6-8分钟
