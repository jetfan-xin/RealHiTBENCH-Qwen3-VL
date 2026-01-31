# Resume 速查表 - 5分钟速成指南

## 🎯 核心问题回答

**Q: `inference_qwen3vl_local_a100_truncate.py` 如何运行resume的text_html & mix_html？**

### A: 三行命令搞定

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference

# 方案1: 一键运行（推荐）
bash run_resume_both.sh

# 方案2: 分别运行
python run_text_html_truncate.py    # ~30分钟
python run_mix_html_truncate.py     # ~30分钟

# 方案3: 完整参数
python inference_qwen3vl_local_a100_truncate.py \
  --modality text --format html --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
  --data_path /data/pan/4xin/datasets/RealHiTBench --resume --batch_size 1
```

---

## 🔄 Resume 机制（核心）

```
┌─ 加载checkpoint ─┐
│                  ├─ results: 3060个（3043成功 + 17个ERROR）
│                  └─ processed_ids: 3043个（仅成功的）
│
├─ 处理循环
│  for query in queries:
│    if query['id'] in processed_ids:
│      skip ✓ (3043个)
│    else:
│      process ✓ (17个ERROR - 自动重新处理！)
│
└─ 保存结果
   checkpoint.json (3060个全部成功)
```

### 关键洞察

> **ERROR样本为什么会被自动重新处理？**
> 
> 因为ERROR结果不是"成功"结果，所以不被加入`processed_ids`中。
> 
> Resume时，`if query['id'] in processed_ids` 这个check会过滤掉3043个成功的样本，
> 但过不了ERROR样本，导致ERROR样本自动进入处理流程！

---

## 📝 参数速查

| 参数 | 说明 | 值 |
|------|------|-----|
| `--modality` | 模态 | `text` / `mix` / `image` |
| `--format` | 格式 | `html` / `markdown` / `latex` / `csv` |
| `--resume` | 启用resume | 无值（标志） |
| `--batch_size` | 批大小 | `1`（推荐） |
| `--model_dir` | 模型路径 | `/data/pan/4xin/models/Qwen3-VL-8B-Instruct` |
| `--data_path` | 数据路径 | `/data/pan/4xin/datasets/RealHiTBench` |

---

## 🎬 文本截断工作流

```
Input: economy-table14_swap.html
       │
       ├─ Tokenize: 334,162 tokens
       │
       ├─ Check: 334,162 > 100,000?
       │         YES ✓
       │
       ├─ Truncate: 334,162 * (100,000/334,162) * 0.9 = ~89,814 chars
       │
       └─ Output: 99,847 tokens ✓
          [TRUNCATE] Input too large, result: 99,847 tokens
```

---

## 📊 效果对比

| 方式 | 时间 | 效率 | 推荐度 |
|------|------|------|--------|
| 重新运行全部 | 15小时 | 低 | ❌ |
| Resume处理 | 1小时 | 高 | ✅ |

---

## ✅ 一键检验

```bash
# 运行完后验证
python << 'EOF'
import json, os
for mode in ['text_html', 'mix_html']:
    path = f'../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_{mode}_truncate/checkpoint.json'
    data = json.load(open(path))
    errors = len([r for r in data['results'] if '[ERROR' in r.get('Prediction', '')])
    print(f"{mode}: {errors} errors" + (" ✅ FIXED!" if errors == 0 else ""))
EOF
```

---

## 🚨 常见问题

| 问题 | 解决方案 |
|------|--------|
| Still have ERROR after resume? | `pkill python` + `nvidia-smi` (检查显存) |
| How to continue after interruption? | Run same command with `--resume` again |
| Results saved where? | `result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_*_truncate/` |

---

## 📂 关键文件位置

```
inference/
├── inference_qwen3vl_local_a100_truncate.py  ← 核心脚本
├── run_text_html_truncate.py                ← Text包装脚本
├── run_mix_html_truncate.py                 ← Mix包装脚本
├── run_resume_both.sh                       ← 一键脚本
├── RESUME_SUMMARY.md                        ← 详细文档
├── RESUME_USAGE_GUIDE.md                    ← 使用指南
├── RESUME_DETAILED_FLOWCHART.md             ← 流程图
└── RESUME_QUICK_COMMANDS.sh                 ← 命令参考
```

---

## 🏃 快速开始（3分钟）

```bash
# 1. 进入目录
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference

# 2. 初始化（第一次）
bash DEPLOYMENT_GUIDE.sh

# 3. 运行resume（核心）
bash run_resume_both.sh

# 4. 查看结果（验证）
tail -f text_html_resume.log
tail -f mix_html_resume.log

# 5. 完成后验证
python << 'EOF'
import json
for mode in ['text_html', 'mix_html']:
    path = f'../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_{mode}_truncate/checkpoint.json'
    data = json.load(open(path))
    errors = len([r for r in data['results'] if '[ERROR' in r.get('Prediction', '')])
    print(f"✅ {mode}: Fixed!" if errors == 0 else f"⚠️ {mode}: {errors} errors")
EOF
```

---

## 🧠 理解要点

### Resume ≠ "继续处理"

Resume = **"加载已有结果 + 跳过成功的 + 重新处理失败的"**

### 为什么17个ERROR样本会自动被重新处理？

```python
# Checkpoint中
processed_ids = {2746, 2748, 2749, ..., 3070}  # 3043个
# ❌ 不包含: 2747, 2748, 2749, 2750, 2751, 2758-2763, 2966-2968, 3019-3021

# Resume时
if query['id'] in processed_ids:
    continue  # 跳过
# else:
#   process ← 这就是ERROR样本的命运！
```

### MAX_INPUT_TOKENS = 100,000 有什么用？

- 自动检测超大输入（>100K tokens）
- 自动截断到100K以内
- 防止OOM错误

---

## 📋 成功标志

运行完成后应该看到：

✅ `Total results: 3060`
✅ `Errors: 0`
✅ `Status: All fixed!`
✅ `Processing Time: ~1 hour`

---

## 🎓 三个关键概念

| 概念 | 说明 |
|------|------|
| **processed_ids** | 已成功处理的查询ID集合（只包含成功，不包含ERROR） |
| **ERROR检测** | 检查 `Prediction.startswith('[ERROR')` |
| **文本截断** | 输入超过100K tokens时自动截断 |

---

## 💾 Checkpoint格式简图

```json
{
  "results": [
    {"id": 2746, "Prediction": "答案...", ...},      ✓ 成功
    {"id": 2747, "Prediction": "[ERROR] OOM", ...},  ✗ 失败
    {"id": 2748, "Prediction": "...", ...}           ✓ 成功
  ],
  "processed_ids": [2746, 2748, ...]  ← 不包含2747！
}
```

当Resume时：
- 加载所有results（3060个）
- processed_ids = 3043个
- 跳过这3043个
- 重新处理ID 2747（以及其他16个ERROR样本）

---

## 🚀 立即开始

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference
bash run_resume_both.sh
```

**预期结果**: ~1小时后，17个OOM错误全部修复！

---

最后更新: 2024年
简化难度: ⭐⭐⭐ (中等)
建议阅读时间: 5分钟
