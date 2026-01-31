# 缺失任务分析与推理 - 快速参考

## 🚀 三步完成

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL

# 1️⃣ 分析缺失任务 (~5分钟)
python utils/analyze_missing_tasks.py

# 2️⃣ 生成推理脚本 (~1秒)
python utils/generate_missing_task_inference.py

# 3️⃣ 运行所有推理 (~6-8分钟)
bash inference/rerun_missing_tasks/run_all_missing_tasks.sh
```

**总用时**: ~10-15分钟  
**处理任务**: 89个缺失任务  
**覆盖配置**: 9个不同配置

---

## 📊 当前统计

| 指标 | 数值 |
|------|------|
| 总任务数 | 3,071 |
| 完成率 | 99.73% |
| 缺失任务 | 84 (跨10个文件) |
| 需重新运行 | 89 tasks |
| - OOM错误 | 44 tasks |
| - 未完成运行 | 45 tasks |

---

## 📁 关键文件

### 输入
- `data/QA_final_sc_filled.json` - 主任务列表
- `result/complied/*/results.json` - 原始结果

### 输出
- `result/complied/*/skip_ids.json` - 每个配置的缺失分析 ✨
- `utils/missing_tasks_summary.json` - 总体统计 ✨
- `inference/rerun_missing_tasks/*.py` - 推理脚本 ✨
- `inference/rerun_missing_tasks/run_all_missing_tasks.sh` - 主脚本 ✨

---

## 🔧 单独运行特定配置

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL

# 文本HTML (17 OOM错误)
python inference/rerun_missing_tasks/rerun_qwen3vl_text_text_html.py

# Mix HTML (25 未完成)
python inference/rerun_missing_tasks/rerun_qwen3vl_resize_pic_mix_html.py

# 其他配置...
ls inference/rerun_missing_tasks/rerun_*.py
```

---

## ✅ 验证结果

```bash
# 重新分析
python utils/analyze_missing_tasks.py

# 查看特定配置
cat result/complied/qwen3vl_text/text_html/skip_ids.json | jq '.statistics'

# 期望: completed=3071, error=0, missing=0
```

---

## 🛡️ OOM防护

**自动启用** 对于:
- text_html, mix_html, mix_csv (有OOM历史)
- 使用 `inference_qwen3vl_local_a100_truncate_with_task_ids.py`
- MAX_INPUT_TOKENS = 100,000
- batch_size = 1

**不需要** 对于:
- image, latex (无OOM风险)
- 使用标准inference脚本

---

## 📚 详细文档

- **完整指南**: [MISSING_TASKS_GUIDE.md](MISSING_TASKS_GUIDE.md)
- **推理脚本说明**: [inference/rerun_missing_tasks/README.md](inference/rerun_missing_tasks/README.md)
- **文本截断机制**: [inference/README_TRUNCATION.md](inference/README_TRUNCATION.md)

---

## 🔄 重新生成脚本

```bash
# 删除旧脚本
rm -rf inference/rerun_missing_tasks

# 重新运行
python utils/analyze_missing_tasks.py
python utils/generate_missing_task_inference.py
```

---

**生成时间**: 2026-01-31  
**维护**: 自动化系统
