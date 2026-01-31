#!/usr/bin/env bash
# QUICK REFERENCE - Resume Commands for Text/Mix HTML

# ============================================================================
# OPTION 1: 使用包装脚本（最简单，推荐）
# ============================================================================

cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference

# Text HTML - 重新处理17个OOM样本（使用MAX_INPUT_TOKENS=100,000截断）
python run_text_html_truncate.py

# Mix HTML - 重新处理17个ERROR样本（多模态）
python run_mix_html_truncate.py


# ============================================================================
# OPTION 2: 直接调用推理脚本（完整控制）
# ============================================================================

# Text HTML with resume
python inference_qwen3vl_local_a100_truncate.py \
  --modality text \
  --format html \
  --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
  --data_path /data/pan/4xin/datasets/RealHiTBench \
  --resume \
  --batch_size 1

# Mix HTML with resume
python inference_qwen3vl_local_a100_truncate.py \
  --modality mix \
  --format html \
  --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
  --data_path /data/pan/4xin/datasets/RealHiTBench \
  --resume \
  --batch_size 1


# ============================================================================
# OPTION 3: 后台运行（长时间任务）
# ============================================================================

# 方式A：nohup后台运行（不依赖终端）
nohup python run_text_html_truncate.py > text_html.log 2>&1 &
nohup python run_mix_html_truncate.py > mix_html.log 2>&1 &

# 查看运行进度
tail -f text_html.log
tail -f mix_html.log

# 方式B：tmux分屏运行（推荐）
tmux new-session -d -s qwen3vl
tmux send-keys -t qwen3vl 'cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference && python run_text_html_truncate.py' Enter
tmux new-window -t qwen3vl -n mix
tmux send-keys -t qwen3vl:mix 'python run_mix_html_truncate.py' Enter

# 查看tmux窗口
tmux list-windows -t qwen3vl
tmux capture-pane -t qwen3vl -p        # 查看text_html窗口
tmux capture-pane -t qwen3vl:mix -p    # 查看mix_html窗口

# 重新连接tmux
tmux attach-session -t qwen3vl


# ============================================================================
# 工作流解释
# ============================================================================

# Resume机制（自动处理）：
#
# 1. 加载checkpoint.json
#    ├─ results: 3060个结果（3043成功 + 17个ERROR）
#    └─ processed_ids: 仅3043个成功ID
#
# 2. 初始化
#    ├─ 加载all_eval_results = 3060个results
#    └─ processed_ids = 3043个成功ID
#
# 3. 处理循环
#    for query in all_queries:
#      if query['id'] in processed_ids:
#        skip ✓
#      else:
#        process query（包括17个ERROR样本）
#
# 4. 文本截断
#    if tokens > 100,000:
#      truncate to 100,000 tokens
#      [TRUNCATE] Input too large, truncating to 100,000
#      [TRUNCATE] Result: 99,847 tokens
#
# 5. 保存结果
#    checkpoint.json（继续save，包含所有结果）
#    results_batch_TIMESTAMP.json（最终结果）


# ============================================================================
# 验证结果
# ============================================================================

# 检查ERROR是否已修复
python << 'EOF'
import json
import os

modes = {
    'text_html': '../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_text_html_default/checkpoint.json',
    'mix_html': '../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_mix_html_default/checkpoint.json'
}

for mode, path in modes.items():
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            errors = [r for r in data['results'] if '[ERROR' in r.get('Prediction', '')]
            print(f"\n{mode}:")
            print(f"  Total: {len(data['results'])}")
            print(f"  Errors: {len(errors)}")
            print(f"  Status: {'✅ All fixed!' if len(errors) == 0 else f'⚠️ {len(errors)} remaining'}")
    else:
        print(f"\n{mode}: ❌ Checkpoint not found")
EOF


# ============================================================================
# 日志位置
# ============================================================================

# 推理脚本日志
../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_text_html_default/checkpoint.json
../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_mix_html_default/checkpoint.json

# 最终结果
../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_text_html_default/results_batch_*.json
../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_mix_html_default/results_batch_*.json


# ============================================================================
# 常用命令
# ============================================================================

# 查看GPU使用情况
nvidia-smi

# 查看运行中的Python进程
ps aux | grep python

# 杀死推理进程（释放显存）
pkill -f "inference_qwen3vl"

# 查看checkpoint统计
python << 'EOF'
import json
path = '../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_text_html_default/checkpoint.json'
with open(path) as f:
    data = json.load(f)
    print(f"Total results: {len(data['results'])}")
    print(f"Processed IDs: {len(data.get('processed_ids', []))}")
    
    # 统计各类型结果
    success = sum(1 for r in data['results'] if not r['Prediction'].startswith('[ERROR'))
    oom = sum(1 for r in data['results'] if '[ERROR] OOM' in r['Prediction'])
    other_error = sum(1 for r in data['results'] if r['Prediction'].startswith('[ERROR') and '[ERROR] OOM' not in r['Prediction'])
    
    print(f"Successful: {success}")
    print(f"OOM errors: {oom}")
    print(f"Other errors: {other_error}")
EOF


# ============================================================================
# 常见问题排查
# ============================================================================

# Q: Resume后仍然有ERROR？
# A: 可能显存不足，尝试：
#    1. nvidia-smi 查看显存
#    2. pkill python 释放显存
#    3. 减小batch_size: --batch_size 1

# Q: 如何监控运行进度？
# A: 
#    1. 实时查看（nohup）: tail -f text_html.log
#    2. 查看checkpoint大小: ls -lh result/.../checkpoint.json
#    3. 统计已处理的ID数: python -c "import json; print(len(json.load(open(...))['processed_ids']))"

# Q: 中断后如何继续？
# A: 直接运行相同命令，加上--resume即可自动继续

# Q: 如何回滚到原始状态？
# A: 
#    rm -rf ../result/qwen3vl_local_a100_default/Qwen3-VL-8B-Instruct_*_default/
#    然后重新设置（见DEPLOYMENT_GUIDE.sh）
