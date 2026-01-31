# Resume 流程详解

## 📊 Resume 的三个核心步骤

```
┌─────────────────────────────────────────────────────────────────┐
│                    脚本启动（--resume flag）                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 加载 Checkpoint 数据                                      │
├─────────────────────────────────────────────────────────────────┤
│ checkpoint_file = 'checkpoint.json'                             │
│ with open(checkpoint_file, 'r') as f:                           │
│     checkpoint_data = json.load(f)                              │
│     all_eval_results = checkpoint_data.get('results', [])       │
│     processed_ids = set(checkpoint_data.get('processed_ids')) │
│                                                                 │
│ 结果：                                                          │
│   ├─ all_eval_results: 3060个结果                              │
│   │  ├─ 3043个成功结果 (Prediction: "...")                    │
│   │  └─ 17个ERROR结果  (Prediction: "[ERROR] OOM: ...")      │
│   └─ processed_ids: 仅3043个成功ID的集合                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 遍历查询 - 决定skip还是处理                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ for query in all_queries:                                       │
│     if query['id'] in processed_ids:  # 这里是关键！             │
│         continue  # Skip已成功的样本                            │
│                                                                 │
│ 最终结果：                                                      │
│   ├─ 跳过: 3043个已成功的查询 ✓                                │
│   └─ 处理: 17个ERROR查询 ← 这17个被自动重新处理！            │
│                                                                 │
│ 为什么ERROR样本会被重新处理？                                  │
│   ERROR样本 NOT IN processed_ids                                 │
│   ├─ 原因: ERROR结果不是"成功"结果                            │
│   └─ 所以: 跳过逻辑不适用，自动进入处理流程                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: 处理查询 - 文本截断 + 推理                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ for each ERROR query:                                           │
│     1. 建立消息: build_messages(query, ...)                     │
│     2. 检查文本大小: tokenize(html_text)                        │
│     3. 如果过大，截断: truncate_text_if_needed(...)            │
│     4. 模型推理: get_final_answer_local(...)                    │
│     5. 保存结果: add to all_eval_results                        │
│     6. 更新已处理: processed_ids.add(query['id'])               │
│     7. 定期checkpoint保存                                       │
│                                                                 │
│ 截断逻辑（MAX_INPUT_TOKENS = 100,000）：                       │
│   Input tokens > 100,000?                                       │
│   ├─ YES: 截断到~100,000 tokens                                │
│   │       [TRUNCATE] Input too large: 334,162 → 99,847        │
│   └─ NO: 直接处理                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: 保存最终结果                                            │
├─────────────────────────────────────────────────────────────────┤
│ results_batch_TIMESTAMP.json（最终输出）                        │
│ checkpoint.json（备份，包含所有3060个结果）                     │
│                                                                 │
│ 统计：                                                          │
│   Total: 3060                                                   │
│   Success: 3060 (之前17个ERROR现在成功)                        │
│   Error: 0                                                      │
│   ✅ 完全成功！                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Text HTML vs Mix HTML Resume

### Text HTML 模式

```
checkpoint.json (text_html)
├─ results (3060个)
│  ├─ [2747]: "[ERROR] OOM: CUDA..."  ← 会重新处理
│  ├─ [2748]: "[ERROR] OOM: CUDA..."  ← 会重新处理
│  ├─ ...
│  ├─ [2751]: "[ERROR] OOM: CUDA..."  ← 会重新处理
│  ├─ [2758]: "[ERROR] OOM: CUDA..."  ← 会重新处理
│  └─ [3021]: "[ERROR] OOM: CUDA..."  ← 会重新处理
│
└─ processed_ids (3043个)
   ├─ 2746, 2752, 2753, ... (所有成功的ID)
   └─ ❌ 不包含2747-2751, 2758-2763, 2966-2968, 3019-3021

运行：
  python run_text_html_truncate.py --resume
           ↓
  加载checkpoint → all_eval_results(3060) + processed_ids(3043)
           ↓
  for query in all_queries:
    if query['id'] in processed_ids:  # 3043个
      skip ✓
    else:                              # 17个
      process with truncation → [TRUNCATE] logic
           ↓
  保存 results_batch_*.json (3060个，17个现已成功)
```

### Mix HTML 模式

```
checkpoint.json (mix_html)
├─ results (3060个)
│  ├─ [2747]: "[ERROR] CUDA..."       ← 会重新处理
│  ├─ [2748]: "[ERROR] CUDA..."       ← 会重新处理
│  ├─ ...
│  └─ [3021]: "[ERROR] CUDA..."       ← 会重新处理
│
└─ processed_ids (3043个)
   └─ ❌ 不包含ERROR IDs

运行：
  python run_mix_html_truncate.py --resume
           ↓
  same logic as text_html but with image modality
           ↓
  保存 results_batch_*.json (3060个，17个现已成功)
```

---

## 🎯 关键区别

### ❌ 错误理解

```
"Resume只是从checkpoint继续处理新的查询，ERROR样本不会被重新处理"

这是错的！因为：
  processed_ids 只包含 SUCCESS 的IDs
  ERROR样本 NOT IN processed_ids
  所以 ERROR样本会被 skip check 过滤掉
  然后自动进入处理流程
```

### ✅ 正确理解

```
"Resume 会：
  1. 加载所有3060个结果（包括17个ERROR）
  2. 标记3043个成功的ID为已处理
  3. 跳过这3043个
  4. 自动重新处理17个ERROR样本（带文本截断）
  5. 最终所有3060个查询都有成功的结果"
```

---

## 📝 核心代码流程

### 设置阶段

```python
# 1. 初始化结果列表和已处理集合
all_eval_results = []
processed_ids = set()

# 2. 加载checkpoint（如果存在且--resume）
checkpoint_file = f'{output_file_path}/checkpoint.json'
if os.path.exists(checkpoint_file) and opt.resume:
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
        all_eval_results = checkpoint_data.get('results', [])          # 3060个
        processed_ids = set(checkpoint_data.get('processed_ids', []))  # 3043个
    
    print(f"Resuming from checkpoint with {len(processed_ids)} processed queries")
    # 输出: "Resuming from checkpoint with 3043 processed queries"
```

### 处理阶段

```python
# 3. 遍历所有查询（3071个）
for query in tqdm(querys):
    # 关键skip逻辑
    if query['id'] in processed_ids:  # 3043个成功ID在这里
        continue                       # 跳过 ✓
    
    # 只有17个ERROR查询会到这里
    try:
        # 处理流程（包括文本截断）
        response = get_final_answer_local(...)
        
        # 保存结果
        result = {
            'id': query['id'],
            'Prediction': response,
            ...
        }
        all_eval_results.append(result)
        processed_ids.add(query['id'])
        
    except Exception as e:
        # 错误处理
        result['Prediction'] = f"[ERROR] {str(e)}"
        all_eval_results.append(result)
```

### 截断阶段（关键！）

```python
def truncate_text_if_needed(messages_text, processor, max_tokens=100000):
    """智能截断超大文本输入"""
    
    # 1. 令牌化
    tokens = processor.apply_chat_template(...)
    input_tokens = len(tokens)
    
    # 2. 检查是否超限
    if input_tokens > max_tokens:
        # 3. 计算截断比例（保留90%的安全margin）
        truncate_ratio = max_tokens / input_tokens
        safe_ratio = truncate_ratio * 0.9
        
        # 4. 按字符截断（不按token，以保持结构）
        truncate_len = int(len(messages_text) * safe_ratio)
        truncated_text = messages_text[:truncate_len]
        
        # 5. 日志输出
        print(f"[TRUNCATE] Input too large ({input_tokens} tokens)")
        print(f"[TRUNCATE] Truncating to ~{max_tokens} tokens")
        print(f"[TRUNCATE] Result: {len(truncated_text)} chars")
        
        return truncated_text, input_tokens, True  # 返回截断标志
    
    return messages_text, input_tokens, False  # 正常大小，无需截断
```

### 保存阶段

```python
# 5. 定期保存checkpoint
if (batch_idx + 1) % save_interval == 0:
    checkpoint_data = {
        'results': all_eval_results,      # 3060个（包括新处理的17个）
        'processed_ids': list(processed_ids),  # 3043+17=3060个
        ...
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"Checkpoint saved: {len(processed_ids)} queries processed")
```

---

## 📊 Resume 效果对比

### 场景1: 不使用Resume（完全重新运行）

```
加载checkpoint: NO
处理查询: 3071个全部处理
  ├─ 新处理的3071个
  └─ ✅ 最终: 3071个结果（包括17个ERROR的重新处理）

时间: ~15小时 ⏱️
GPU: 100% 占用 
🔴 浪费! 重复处理了3043个已成功的查询
```

### 场景2: 使用Resume（推荐）

```
加载checkpoint: YES → 3060个已有结果
处理查询: 仅17个ERROR查询
  ├─ 跳过3043个成功查询 ✓
  └─ 重处理17个ERROR → [TRUNCATE]

时间: ~1小时 ⏱️
GPU: 100% 占用（只在处理ERROR时）
🟢 高效! 节省14小时，只处理失败样本
```

### 场景3: 手工指定skip_checkpoint（多进程分片）

```
使用--skip_checkpoint指定外部checkpoint
用于多进程并行（不同GPU运行不同部分）

python ... --shard_id 0 --num_shards 4 \
           --skip_checkpoint /path/to/other/checkpoint.json
```

---

## 🚀 完整工作流示例

### Text HTML Resume 完整示例

```
Step 1: 检查当前状态
  $ ls -la ../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_html_a100/
  checkpoint.json (包含3043成功 + 17 ERROR)

Step 2: 设置截断目录（一次性）
  $ bash DEPLOYMENT_GUIDE.sh
  ✓ 创建output目录
  ✓ 复制checkpoint
  ✓ 验证17个OOM样本

Step 3: 运行resume推理
  $ python inference_qwen3vl_local_a100_truncate.py \
      --modality text \
      --format html \
      --resume

  输出:
    Resuming from checkpoint with 3043 processed queries
    Found 17 OOM errors to reprocess:
      - ID 2747: [ERROR] OOM: CUDA out of memory...
      - ID 2748: [ERROR] OOM: CUDA out of memory...
      ...
    
    Processing Query ID: 2747
      HTML size: 334,162 tokens
      [TRUNCATE] Input too large, truncating to 100,000
      [TRUNCATE] Result: 99,847 tokens
      Prediction: Based on the table analysis...
      Time: 45s
    
    Processing Query ID: 2748
      ...
    
    EVALUATION COMPLETE
    Total queries: 3060
    Duration: 765s
    Results saved to: checkpoint.json

Step 4: 验证结果
  $ python << 'EOF'
  import json
  with open(...checkpoint.json') as f:
      data = json.load(f)
      errors = len([r for r in data['results'] 
                    if '[ERROR' in r['Prediction']])
      print(f"Total: {len(data['results'])}")
      print(f"Errors: {errors}")
      if errors == 0:
          print("✅ All fixed!")
  EOF

  输出:
    Total: 3060
    Errors: 0
    ✅ All fixed!
```

---

## 🎓 学习要点

1. **Resume = 加载 + Skip已处理 + 处理新/失败的**
   - 不只是"继续"，是"智能跳过"

2. **ERROR样本自动被重新处理**
   - 因为ERROR结果不在processed_ids中
   - Skip逻辑过滤不了它们

3. **文本截断是关键**
   - MAX_INPUT_TOKENS = 100,000
   - 只有5个样本真的需要截断
   - 其他17个可能通过截断修复OOM

4. **效率提升巨大**
   - 仅处理17个失败样本 vs 重新处理3071个
   - 节省93%的时间和计算资源

---

最后更新: 2024年
