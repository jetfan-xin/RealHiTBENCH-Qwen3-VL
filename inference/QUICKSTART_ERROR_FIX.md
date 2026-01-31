# 🚀 快速开始：修复OOM错误样本

## ✅ 是的！新程序已经包含自动重处理ERROR样本的功能

### 核心特性

1. **自动ERROR检测** 
   - 扫描checkpoint中所有`Prediction`字段
   - 识别以`[ERROR]`开头的失败样本
   - 自动将这些样本加入重试队列

2. **智能checkpoint恢复**
   ```python
   # 旧逻辑（会跳过ERROR样本）
   processed_ids = set(checkpoint_data.get('processed_ids', []))
   
   # 新逻辑（自动排除ERROR样本）
   for result in all_eval_results:
       if result['Prediction'].startswith('[ERROR'):
           error_ids.add(result['id'])  # 标记为需重试
       else:
           processed_ids.add(result['id'])  # 仅记录成功样本
   ```

3. **文本截断防OOM**
   - MAX_INPUT_TOKENS = 100,000
   - 在重试时自动截断超长文本
   - 成功避免再次OOM

---

## 📋 当前状态分析

### Text HTML模式
- ✅ 成功样本：**3,043**个
- ⚠️ OOM错误：**17**个（需重处理）
- 错误ID：`[2747, 2748, 2749, 2750, 2751, 2758, 2759, 2760, 2761, 2762, 2763, 2966, 2967, 2968, 3019, 3020, 3021]`

### Mix HTML模式
- ✅ 成功样本：**3,043**个
- ⚠️ 图片OOM：**17**个（需重处理）
- 错误ID：同上（相同样本的不同模态）

---

## 🎯 推荐操作流程

### 方法1：一键自动重处理（最快）⭐

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference

# 1. 自动复制checkpoint并设置
bash setup_reprocess_oom.sh

# 2. 重处理text_html的17个OOM样本
python run_text_html_truncate.py

# 3. 重处理mix_html的17个ERROR样本  
python run_mix_html_truncate.py
```

**优势**：
- ⏱️ 仅处理17个失败样本（~30分钟）
- 💾 不重复处理3043个成功样本（节省~15小时）
- 🔒 原始结果保持不变

### 方法2：手动验证流程

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference

# 1. 查看OOM样本数量
python << 'EOF'
import json
with open('../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_html_a100/checkpoint.json') as f:
    data = json.load(f)
    oom = [r for r in data['results'] if r['Prediction'].startswith('[ERROR')]
    print(f"OOM samples: {len(oom)}")
    print(f"IDs: {[r['id'] for r in oom]}")
EOF

# 2. 手动复制checkpoint
mkdir -p ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/
cp ../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_html_a100/checkpoint.json \
   ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/

# 3. 运行重处理
python run_text_html_truncate.py
```

---

## 📊 运行时输出示例

```
Resuming from checkpoint:
  - Successful: 3043 queries
  - Failed (will retry): 17 queries
  Will reprocess ID 2747: [ERROR] OOM: Text too large (334159 tokens)...
  Will reprocess ID 2748: [ERROR] OOM: Text too large (334159 tokens)...
  ... (15 more)

Processing text modality:   0%|          | 0/17 [00:00<?, ?it/s]

============================================================
Query ID: 2747 | Type: Structure Comprehending
============================================================
  [TRUNCATE] Input too large (334,159 tokens), truncating to 100,000
  [TRUNCATE] Result: 99,823 tokens (original: 334,159)
  Prediction: Craft
  Reference: Craft
  Metrics: {'F1': 100.0, 'EM': 100.0, 'ROUGE-L': 100.0, 'SacreBLEU': 0.0}
  Processing Time: 45.2s

Processing text modality:   6%|▌         | 1/17 [00:45<11:23, 42.7s/it]
```

---

## ❓ 常见问题

### Q1: 为什么要复制checkpoint？
**A**: 为了利用已有的成功结果，避免重复计算3043个已处理样本。

### Q2: 会覆盖原始结果吗？
**A**: 不会。新结果保存在独立目录：`result/qwen3vl_local_a100_truncate/`

### Q3: 如果不复制checkpoint会怎样？
**A**: 会从头处理全部3071个样本（~15小时），而非仅处理17个失败样本（~30分钟）。

### Q4: checkpoint.json会被修改吗？
**A**: 会。脚本会移除ERROR结果，仅保留成功结果，然后在重处理后添加新结果。

### Q5: 如果重处理仍然失败怎么办？
**A**: 
1. 检查是否确实使用了truncate脚本（有MAX_INPUT_TOKENS限制）
2. 查看日志中的`[TRUNCATE]`消息
3. 如需降低阈值，修改`MAX_INPUT_TOKENS`为80000或60000

---

## 🔍 验证成功

运行完成后检查：

```bash
# 检查text_html结果
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL

python << 'EOF'
import json
with open('result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/checkpoint.json') as f:
    data = json.load(f)
    total = len(data['results'])
    errors = len([r for r in data['results'] if r['Prediction'].startswith('[ERROR')])
    print(f"Total results: {total}")
    print(f"Successful: {total - errors}")
    print(f"Still ERROR: {errors}")
    
    if errors == 0:
        print("✅ All samples processed successfully!")
    else:
        print(f"⚠️ {errors} samples still have errors")
EOF
```

预期输出：
```
Total results: 3060
Successful: 3060
Still ERROR: 0
✅ All samples processed successfully!
```

---

## 📝 文件清单

| 文件 | 用途 |
|------|------|
| inference_qwen3vl_local_a100_truncate.py | 带ERROR自动重处理的推理脚本 |
| run_text_html_truncate.py | text_html运行脚本 |
| run_mix_html_truncate.py | mix_html运行脚本 |
| setup_reprocess_oom.sh | ⭐ 一键设置脚本（复制checkpoint） |
| demo_error_reprocessing.py | 功能演示脚本 |
| README_TRUNCATION.md | 详细文档 |
| QUICKSTART_ERROR_FIX.md | 本文件 |

---

## ⏱️ 时间对比

| 方法 | 处理样本数 | 预计时间 |
|------|-----------|---------|
| ❌ 从头运行（不推荐） | 3071 | ~15小时 |
| ✅ 使用checkpoint重处理（推荐） | 17 | ~30分钟 |
| 节省 | -3054 | **~14.5小时** |

---

**立即开始**：
```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference
bash setup_reprocess_oom.sh
python run_text_html_truncate.py
```
