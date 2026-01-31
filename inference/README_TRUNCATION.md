# Text Truncation Inference Scripts

## 新增文件说明

### 1. inference_qwen3vl_local_a100_truncate.py
基于 `inference_qwen3vl_local_a100_default.py` 改进的推理脚本，**增加自动文本截断功能**以防止 OOM。

**核心改进：**
- **文本截断阈值**：`MAX_INPUT_TOKENS = 100,000`（基于缺失ID数据分析）
- **智能截断机制**：
  - 自动检测输入token数，超限时截断到安全范围
  - 保留90%目标长度作为安全边界
  - 添加截断标记提示模型
- **自动ERROR样本重处理**：⭐ 新功能
  - 自动识别checkpoint中的`[ERROR] OOM:`样本
  - 排除ERROR样本，仅保留成功结果
  - 使用`--resume`时自动重新处理这些失败样本
  - 支持text和mix两种模式的OOM错误
- **适用场景**：text_html、mix_html等包含超大HTML文件的任务

**关键函数：**
```python
def truncate_text_if_needed(messages_text: str, processor, max_tokens: int = MAX_INPUT_TOKENS):
    """
    检查并截断超长文本
    返回: (truncated_text, original_tokens, was_truncated)
    """
```

### 2. run_text_html_truncate.py
运行 text_html 模式的便捷脚本，启用文本截断。

**用法：**
```bash
cd inference
python run_text_html_truncate.py
```

### 3. run_mix_html_truncate.py
运行 mix_html 模式的便捷脚本，启用文本截断。

**用法：**
```bash
cd inference
python run_mix_html_truncate.py
```

## 数据分析结果

### 缺失ID分析（共11个）

**Text HTML模式和Mix HTML模式共同缺失：**
- 2216-2220: labor-table68（无HTML文件，49.36M像素图片）
- 2960-2962: society-table10_swap（无HTML文件，4.96M像素图片）
- 3069-3071: science-table72_swap（小HTML文件~7.5K tokens，2.10M像素图片）

**关键发现：**
1. **labor-table68 和 society-table10_swap 本身缺少HTML文件** → 数据集问题，非OOM
2. **science-table72_swap HTML正常但被跳过** → 可能是其他原因（待检查）
3. **真正的OOM问题样本**（从日志发现）：
   - Query 2747-2749: economy-table14_swap（1.2MB HTML, 334K tokens, 11.96M像素）
   - Query 2750-2751: society-table02_swap（缺HTML，5.00M像素）

### 截断阈值选择

基于分析，选择 **MAX_INPUT_TOKENS = 100,000**：

**理由：**
- ✅ 大部分正常表格远低于此阈值（<10K tokens）
- ✅ 允许处理较大的复杂表格（10K-100K tokens）
- ✅ 截断极端异常值（如economy-table14_swap的334K tokens）
- ✅ 防止CUDA OOM的同时最大化信息保留

**预期效果：**
- 正常样本：无影响
- 超大样本：截断到100K tokens，仍保留约30%原始内容（对economy-table14_swap）
- OOM率：降低至接近0

## 使用建议

### ⭐ 场景1：自动重处理已有checkpoint中的OOM错误（推荐）
```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference

# 将已有checkpoint复制到truncate目录
mkdir -p ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/
cp ../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_html_a100/checkpoint.json \
   ../result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/

# 运行时会自动识别并重处理17个OOM样本
python run_text_html_truncate.py
```

**优势**：
- ✅ 不重复处理已成功的3043个样本
- ✅ 仅重新处理17个OOM失败样本
- ✅ 节省时间（~1小时 vs ~15小时）

### 场景2：从头运行text_html（处理缺失ID）
```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference
python run_text_html_truncate.py
```
- 输出目录：`result/qwen3vl_local_a100_truncate/Qwen3-VL-8B-Instruct_text_html_truncate/`
- 将处理之前OOM的样本（economy-table14_swap等）

### 场景3：重新运行mix_html（处理缺失ID）
```bash
### 启动时显示ERROR样本检测：
```
Resuming from checkpoint:
  - Successful: 3043 queries
  - Failed (will retry): 17 queries
  Will reprocess ID 2747: [ERROR] OOM: Text too large (334159 tokens)...
  Will reprocess ID 2748: [ERROR] OOM: Text too large (334159 tokens)...
  Will reprocess ID 2749: [ERROR] OOM: Text too large (334162 tokens)...
  ... (14 more)
```

### 运行时显示截断信息：
```
Query ID: 2749 | File: economy-table14_swap
============================================================
  [TRUNCATE] Input too large (334,162 tokens), truncating to 100,000
  [TRUNCATE] Result: 99,847 tokens (original: 334,162)
  Warning: Large text input (99,847 tokens)
  Prediction: growing
  Reference: growing
  Metrics: {'F1': 100.0, 'EM': 100.0, 'ROUGE-L': 100.0, 'SacreBLEU': 100.0}
  Processing Time: 45.23
    --format html \
    --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
    --data_path /data/pan/4xin/datasets/RealHiTBench \
    --resume
```

## 输出示例

运行时会显示截断信息：
```
Query ID: 2749 | File: economy-table14_swap
============================================================
  [TRUNCATE] Input too large (334,162 tokens), truncating to 100,000
  [TRUNCATE] Result: 99,847 tokens (original: 334,162)
  Warning: Large text input (99,847 tokens)
  Prediction: [模型输出]
  Metrics: {'F1': XX, 'EM': XX, 'ROUGE-L': XX, 'SacreBLEU': XX}
  Processing Time: XX.XXs
```

## 与原脚本对比

| 特性 | inference_qwen3vl_local_a100_default.py | inference_qwen3vl_local_a100_truncate.py |
|------|----------------------------------------|----------------------------------------|
| 文本处理 | 直接传入，超限OOM | 自动检测&截断 |
| MAX_INPUT_TOKENS | 无限制 | 100,000 |
| OOM处理 | try-catch捕获 | 主动预防+捕获 |
| 输出目录 | qwen3vl_local_a100_default | qwen3vl_local_a100_truncate |
| 适用场景 | 标准benchmark | 包含超大文本的数据集 |

## 注意事项

1. **不修改原脚本**：新脚本独立存在，不影响已有实验
2. **截断可能影响精度**：对于被截断的极端样本，答案准确性可能下降
3. **输出目录隔离**：结果保存在单独目录，便于对比分析
4. **checkpoint兼容**：支持--resume从断点恢复
5. **自动ERROR重处理**：⭐ 使用--resume时会自动检测并重试所有`[ERROR]`样本
6. **已知OOM样本**：text_html和mix_html各有17个OOM样本会被自动重处理

## 后续优化建议

如果100K仍然导致部分OOM，可调整：
1. 降低阈值到80K或60K
2. 增加图片分辨率限制（对mix模式）
3. 使用更激进的HTML压缩策略
