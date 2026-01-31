#!/usr/bin/env python3
"""
合并image/mix模态的shard结果，并添加Structure Comprehending任务结果。

工作流:
1. 从每个shard目录中找到最新的results_*.json文件
2. 合并所有shard的results，排除Structure Comprehending任务
3. 从sc专用文件中提取所有Structure Comprehending结果
4. 合并所有结果到最终JSON文件（仅保留configs和results）

用法:
    python merge_resize_shards_with_sc.py --modality image
    python merge_resize_shards_with_sc.py --modality mix --format html
    python merge_resize_shards_with_sc.py --modality mix --format markdown
"""

import json
import os
import glob
import argparse
from datetime import datetime
from pathlib import Path


def find_latest_result_file(shard_dir):
    """在shard目录中找到最新的results_*.json文件"""
    pattern = os.path.join(shard_dir, 'results_*.json')
    result_files = glob.glob(pattern)
    
    if not result_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_file = max(result_files, key=os.path.getmtime)
    return latest_file


def load_json_file(filepath):
    """加载JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载文件失败 {filepath}: {e}")
        return None


def find_latest_sc_file(sc_dir):
    """在SC目录中找到最新的results_*.json文件"""
    pattern = os.path.join(sc_dir, 'results_*.json')
    result_files = glob.glob(pattern)
    
    if not result_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_file = max(result_files, key=os.path.getmtime)
    return latest_file


def prepare_configs(configs_list):
    """准备configs数组，保留所有原始config并添加合并信息"""
    if not configs_list:
        return []
    
    # 为每个config添加来源标识
    enhanced_configs = []
    for i, cfg in enumerate(configs_list):
        enhanced_cfg = cfg.copy()
        # 添加来源信息（如果尚未存在）
        if 'source_identifier' not in enhanced_cfg:
            if i < len(configs_list) - 1:
                enhanced_cfg['source_identifier'] = f'shard_{i}'
            else:
                enhanced_cfg['source_identifier'] = 'structure_comprehending'
        enhanced_configs.append(enhanced_cfg)
    
    return enhanced_configs


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='合并shard结果并添加Structure Comprehending结果')
    parser.add_argument('--modality', type=str, required=True, 
                        choices=['image', 'mix'],
                        help='模态类型: image 或 mix')
    parser.add_argument('--format', type=str, default='html',
                        help='格式类型 (默认: html)')
    args = parser.parse_args()
    
    # 路径配置
    base_result_dir = '/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/result/qwen3vl_local_a100_resize_pic'
    
    # 根据模态和格式构建路径
    if args.modality == 'image':
        merged_dir_name = 'Qwen3-VL-8B-Instruct_image_a100_merged'
        shard_prefix = 'Qwen3-VL-8B-Instruct_image_a100_shard'
        sc_dir_name = 'Qwen3-VL-8B-Instruct_image_a100_sc'
        output_filename = 'results_image_merged_with_sc.json'
    else:  # mix
        merged_dir_name = f'Qwen3-VL-8B-Instruct_mix_{args.format}_a100_merged'
        shard_prefix = f'Qwen3-VL-8B-Instruct_mix_{args.format}_a100_shard'
        sc_dir_name = f'Qwen3-VL-8B-Instruct_mix_{args.format}_a100_sc'
        output_filename = f'results_mix_{args.format}_merged_with_sc.json'
    
    base_dir = os.path.join(base_result_dir, merged_dir_name)
    sc_dir = os.path.join(base_dir, sc_dir_name)
    output_file = os.path.join(base_dir, output_filename)
    
    print("=" * 80)
    print(f"{args.modality.upper()}模态Shard合并 + Structure Comprehending结果整合")
    print("=" * 80)
    print(f"模态: {args.modality}")
    if args.modality == 'mix':
        print(f"格式: {args.format}")
    print()
    
    # Step 1: 找到所有shard目录
    print("[1/6] 扫描shard目录...")
    shard_pattern = os.path.join(base_dir, f'{shard_prefix}*')
    shard_dirs = sorted(glob.glob(shard_pattern))
    
    if not shard_dirs:
        print(f"❌ 未找到shard目录: {shard_pattern}")
        return
    
    print(f"✓ 找到 {len(shard_dirs)} 个shard目录:")
    for shard_dir in shard_dirs:
        print(f"  - {os.path.basename(shard_dir)}")
    print()
    
    # Step 2: 查找最新的SC文件
    print("[2/6] 查找Structure Comprehending文件...")
    if not os.path.exists(sc_dir):
        print(f"❌ SC目录不存在: {sc_dir}")
        return
    
    sc_file = find_latest_sc_file(sc_dir)
    if not sc_file:
        print(f"❌ 未在SC目录中找到results_*.json文件: {sc_dir}")
        return
    
    print(f"  ✓ 找到最新SC文件:")
    print(f"    - 目录: {sc_dir_name}")
    print(f"    - 文件: {os.path.basename(sc_file)}")
    print(f"    - 修改时间: {datetime.fromtimestamp(os.path.getmtime(sc_file)).strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 3: 加载所有shard的结果
    print("[3/6] 加载shard结果文件...")
    all_results = []
    all_configs = []
    
    for shard_dir in shard_dirs:
        shard_name = os.path.basename(shard_dir)
        latest_file = find_latest_result_file(shard_dir)
        
        if not latest_file:
            print(f"  ⚠️  {shard_name}: 未找到results_*.json文件")
            continue
        
        data = load_json_file(latest_file)
        if not data:
            print(f"  ⚠️  {shard_name}: 文件加载失败")
            continue
        
        results = data.get('results', [])
        config = data.get('config', {})
        
        # 过滤掉Structure Comprehending任务
        non_sc_results = [
            r for r in results 
            if r.get('QuestionType') != 'Structure Comprehending'
        ]
        
        sc_count = len(results) - len(non_sc_results)
        
        print(f"  ✓ {shard_name}:")
        print(f"    - 文件: {os.path.basename(latest_file)}")
        print(f"    - 总结果: {len(results)}")
        print(f"    - 非SC结果: {len(non_sc_results)}")
        print(f"    - SC结果(将被替换): {sc_count}")
        
        all_results.extend(non_sc_results)
        all_configs.append(config)
    
    print(f"\n  合并后非SC结果总数: {len(all_results)}")
    print()
    
    # Step 4: 加载Structure Comprehending结果
    print("[4/6] 加载Structure Comprehending结果...")
    sc_data = load_json_file(sc_file)
    if not sc_data:
        print("❌ SC文件加载失败")
        return
    
    sc_results = sc_data.get('results', [])
    
    # 提取Structure Comprehending结果
    sc_only_results = [
        r for r in sc_results 
        if r.get('QuestionType') == 'Structure Comprehending'
    ]
    
    print(f"  ✓ 从SC文件中提取:")
    print(f"    - 文件: {os.path.basename(sc_file)}")
    print(f"    - 总结果: {len(sc_results)}")
    print(f"    - SC结果: {len(sc_only_results)}")
    
    # 添加SC config
    sc_config = sc_data.get('config', {})
    all_configs.append(sc_config)
    print()
    
    # Step 5: 合并所有结果
    print("[5/6] 合并所有结果...")
    all_results.extend(sc_only_results)
    
    # 按ID排序
    all_results.sort(key=lambda x: x.get('id', 0))
    
    # 统计各类型数量
    type_counts = {}
    for r in all_results:
        qtype = r.get('QuestionType', 'Unknown')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    print(f"  ✓ 合并完成:")
    print(f"    - 总结果数: {len(all_results)}")
    print(f"    - 按类型统计:")
    for qtype, count in sorted(type_counts.items()):
        print(f"      · {qtype}: {count}")
    print()
    
    # Step 6: 准备configs并保存
    print("[6/6] 保存最终结果...")
    configs = prepare_configs(all_configs)
    
    # 添加合并元信息到第一个config
    if configs:
        configs[0]['merged_total_queries'] = len(all_results)
        configs[0]['merge_description'] = "Merged from shards (non-SC) + SC results"
        configs[0]['merge_source_shards'] = len(shard_dirs)
        configs[0]['merge_sc_source'] = os.path.basename(sc_file)
        configs[0]['merge_time'] = datetime.now().isoformat()
    
    # 构建最终JSON（只包含configs和results）
    final_data = {
        'configs': configs,
        'results': all_results
    }
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ 保存成功:")
    print(f"    - 输出文件: {output_file}")
    print(f"    - 文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print(f"    - Configs数量: {len(configs)}")
    print()
    
    # 验证结果
    print("=" * 80)
    print("验证最终结果")
    print("=" * 80)
    
    # 检查ID唯一性
    ids = [r.get('id') for r in all_results]
    unique_ids = set(ids)
    
    print(f"✓ ID检查:")
    print(f"  - 总结果数: {len(all_results)}")
    print(f"  - 唯一ID数: {len(unique_ids)}")
    if len(ids) == len(unique_ids):
        print(f"  - 状态: ✅ 无重复ID")
    else:
        duplicates = len(ids) - len(unique_ids)
        print(f"  - 状态: ⚠️  发现 {duplicates} 个重复ID")
    print()
    
    # 检查ID范围
    if ids:
        print(f"✓ ID范围:")
        print(f"  - 最小ID: {min(ids)}")
        print(f"  - 最大ID: {max(ids)}")
        print()
    
    # 按类型统计
    print(f"✓ 类型统计:")
    for qtype, count in sorted(type_counts.items()):
        percentage = count / len(all_results) * 100
        print(f"  - {qtype}: {count} ({percentage:.1f}%)")
    print()
    
    print("=" * 80)
    print("✅ 合并完成！")
    print("=" * 80)
    print(f"\n最终文件: {output_file}")
    print(f"总结果数: {len(all_results)}")
    print()


if __name__ == '__main__':
    main()
