#!/usr/bin/env python3
"""
分析 QA_final_sc_filled.json 中所有答案的长度，
以确定合适的 max_tokens 值
"""

import json
import sys
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np

def analyze_qa_file(qa_file, model_name="Qwen/Qwen3-VL-8B-Instruct"):
    """分析 QA 文件中所有答案的 token 长度"""
    
    print(f"Loading tokenizer from {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Falling back to approximate token counting (1 token ≈ 4 chars)")
        tokenizer = None
    
    print(f"Loading QA file: {qa_file}")
    with open(qa_file, 'r') as f:
        data = json.load(f)
    
    queries = data.get('queries', [])
    print(f"Total queries: {len(queries)}\n")
    
    # 按类型收集答案长度
    answers_by_type = defaultdict(list)
    
    for query in queries:
        qtype = query.get('QuestionType', 'Unknown')
        
        # 获取答案（优先使用 FinalAnswer）
        if 'FinalAnswer' in query:
            answer = query['FinalAnswer']
        elif 'ProcessedAnswer' in query:
            answer = query['ProcessedAnswer']
        else:
            continue
        
        # 计算 token 数
        if tokenizer:
            token_count = len(tokenizer.encode(str(answer)))
        else:
            # 粗略估计：1 token ≈ 4 个字符
            token_count = max(1, len(str(answer)) // 4)
        
        answers_by_type[qtype].append({
            'tokens': token_count,
            'chars': len(str(answer)),
            'answer': str(answer)[:100] if isinstance(answer, str) else str(answer)[:100],
            'id': query.get('id'),
            'subtype': query.get('SubQType', '')
        })
    
    # 分析统计信息
    print("=" * 80)
    print("ANSWER LENGTH ANALYSIS BY QUESTION TYPE")
    print("=" * 80)
    
    overall_tokens = []
    
    for qtype in sorted(answers_by_type.keys()):
        items = answers_by_type[qtype]
        tokens = [item['tokens'] for item in items]
        
        overall_tokens.extend(tokens)
        
        print(f"\n{qtype}:")
        print(f"  Count: {len(items)}")
        print(f"  Token count:")
        print(f"    Min: {min(tokens)}")
        print(f"    Max: {max(tokens)}")
        print(f"    Mean: {np.mean(tokens):.1f}")
        print(f"    Median: {np.median(tokens):.1f}")
        print(f"    P95: {np.percentile(tokens, 95):.1f}")
        print(f"    P99: {np.percentile(tokens, 99):.1f}")
        
        # 显示最长的几个答案
        sorted_items = sorted(items, key=lambda x: x['tokens'], reverse=True)
        print(f"  Longest answers:")
        for i, item in enumerate(sorted_items[:3], 1):
            preview = item['answer'].replace('\n', ' ')[:60]
            print(f"    {i}. [{item['tokens']} tokens] {preview}...")
    
    # 全局统计
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total answers: {len(overall_tokens)}")
    print(f"\nToken count distribution:")
    print(f"  Min: {min(overall_tokens)}")
    print(f"  Max: {max(overall_tokens)}")
    print(f"  Mean: {np.mean(overall_tokens):.1f}")
    print(f"  Median: {np.median(overall_tokens):.1f}")
    print(f"  P90: {np.percentile(overall_tokens, 90):.1f}")
    print(f"  P95: {np.percentile(overall_tokens, 95):.1f}")
    print(f"  P99: {np.percentile(overall_tokens, 99):.1f}")
    
    # 分布统计
    print(f"\nToken count ranges:")
    ranges = [(0, 50), (50, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000), (10000, float('inf'))]
    for low, high in ranges:
        count = sum(1 for t in overall_tokens if low <= t < high)
        pct = 100 * count / len(overall_tokens)
        if high == float('inf'):
            print(f"  >= {low}: {count} ({pct:.1f}%)")
        else:
            print(f"  {low}-{high}: {count} ({pct:.1f}%)")
    
    # 推荐值
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    p99_value = np.percentile(overall_tokens, 99)
    p95_value = np.percentile(overall_tokens, 95)
    max_value = max(overall_tokens)
    
    print(f"\nBased on analysis:")
    print(f"  P99 value: {p99_value:.0f} tokens (99% of answers fit)")
    print(f"  P95 value: {p95_value:.0f} tokens (95% of answers fit)")
    print(f"  Max value: {max_value} tokens")
    
    # 推荐的 max_tokens 值
    recommendations = [
        ("Ultra-conservative (cover all)", max_value + 100),
        ("Very conservative (P99 + buffer)", int(p99_value * 1.2)),
        ("Conservative (P99 + 50%)", int(p99_value * 1.5)),
        ("Balanced (P99 * 2)", int(p99_value * 2)),
        ("Aggressive (P95 * 1.5)", int(p95_value * 1.5)),
    ]
    
    print(f"\nRecommended max_tokens values:")
    for desc, value in recommendations:
        # 调整到合理的值
        if value < 100:
            value = 100
        elif value < 512:
            value = 512
        elif value < 1024:
            value = 1024
        else:
            # 向上取整到 256 的倍数
            value = ((value + 255) // 256) * 256
        
        print(f"  {desc}: {value}")
    
    print(f"\n💡 SUGGESTED VALUE: {int(p99_value * 1.5)}")
    print(f"   This covers ~99% of answers with 50% buffer")
    print(f"   Balances memory efficiency with safety margin")

if __name__ == "__main__":
    qa_file = "/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data/QA_final_sc_filled.json"
    analyze_qa_file(qa_file)
