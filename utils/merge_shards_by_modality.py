#!/usr/bin/env python3
"""
Merge results from multiple shards of a specific modality.
Finds the latest results_*.json files from each shard and merges them.

Usage:
    python merge_shards_by_modality.py --result_dir ../result/qwen3vl_local_a100_default --modality text_json --num_shards 3
    python merge_shards_by_modality.py --result_dir ../result/qwen3vl_local_a100_default --modality image --num_shards 3
    python merge_shards_by_modality.py --result_dir ../result/qwen3vl_local_a100_default --modality mix_html --num_shards 3
"""

import os
import json
import argparse
import glob
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def find_latest_results_file(shard_dir):
    """Find the latest results_*.json file in a shard directory."""
    pattern = os.path.join(shard_dir, 'results_*.json')
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by modification time, return the latest
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def merge_shard_results(result_base_dir, model_name, modality_suffix, num_shards):
    """
    Merge results from multiple shards into a single result file.
    
    Args:
        result_base_dir: Base directory containing shard results
        model_name: Model name prefix (e.g., 'Qwen3-VL-8B-Instruct')
        modality_suffix: Modality suffix (e.g., 'text_json', 'mix_html', 'image')
        num_shards: Number of shards to merge
    """
    
    print(f"\n{'='*60}")
    print(f"Merging {num_shards} shards of {modality_suffix}")
    print(f"{'='*60}\n")
    
    # Collect all results
    all_results = []
    all_processed_count = 0
    shard_info = {}
    
    for shard_id in range(num_shards):
        # Build shard directory name
        # Try both patterns: with and without '_default'
        shard_dir_candidates = [
            os.path.join(result_base_dir, f'{model_name}_{modality_suffix}_default_shard{shard_id}'),
            os.path.join(result_base_dir, f'{model_name}_{modality_suffix}_shard{shard_id}'),
        ]
        
        shard_dir = None
        for candidate in shard_dir_candidates:
            if os.path.exists(candidate):
                shard_dir = candidate
                break
        
        if not shard_dir:
            print(f"⚠️  Shard {shard_id} directory not found")
            print(f"   Tried: {shard_dir_candidates[0]}")
            print(f"   Tried: {shard_dir_candidates[1]}")
            continue
        
        # Find latest results file
        latest_file = find_latest_results_file(shard_dir)
        if not latest_file:
            print(f"⚠️  Shard {shard_id}: No results_*.json found in {shard_dir}")
            continue
        
        # Load results
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            shard_results = data.get('results', [])
            all_results.extend(shard_results)
            all_processed_count += len(shard_results)
            
            shard_info[shard_id] = {
                'file': os.path.basename(latest_file),
                'count': len(shard_results),
                'path': latest_file
            }
            
            print(f"✓ Shard {shard_id}: {len(shard_results)} results from {os.path.basename(latest_file)}")
        
        except json.JSONDecodeError as e:
            print(f"✗ Shard {shard_id}: Error decoding JSON from {latest_file}")
            print(f"  Error: {e}")
            continue
        except Exception as e:
            print(f"✗ Shard {shard_id}: Error loading {latest_file}")
            print(f"  Error: {e}")
            continue
    
    if not all_results:
        print("\n❌ ERROR: No results found in any shard!")
        return None
    
    # Create output directory
    merged_dir = os.path.join(result_base_dir, f'{model_name}_{modality_suffix}_default_merged')
    os.makedirs(merged_dir, exist_ok=True)
    
    # Calculate aggregate metrics
    print(f"\nCalculating aggregate metrics...")
    
    task_metric_totals = defaultdict(lambda: defaultdict(float))
    task_metric_counts = defaultdict(lambda: defaultdict(int))
    
    for result in all_results:
        task_type = result.get('QuestionType', 'Unknown')
        
        # Process metrics from the 'Metrics' sub-dictionary
        if 'Metrics' in result and isinstance(result['Metrics'], dict):
            for metric_name, metric_value in result['Metrics'].items():
                # Convert boolean to numeric
                if isinstance(metric_value, bool):
                    metric_value = 1.0 if metric_value else 0.0
                # Convert string to numeric
                elif isinstance(metric_value, str):
                    if metric_value.lower() == 'true':
                        metric_value = 1.0
                    elif metric_value.lower() == 'false':
                        metric_value = 0.0
                    else:
                        try:
                            metric_value = float(metric_value)
                        except (ValueError, TypeError):
                            continue
                
                if isinstance(metric_value, (int, float)):
                    task_metric_totals[task_type][metric_name] += metric_value
                    task_metric_counts[task_type][metric_name] += 1
        
        # Also process ProcessingTime
        if 'ProcessingTime' in result:
            try:
                pt = float(result['ProcessingTime'])
                task_metric_totals[task_type]['ProcessingTime'] += pt
                task_metric_counts[task_type]['ProcessingTime'] += 1
            except (ValueError, TypeError):
                pass
    
    # Calculate averages
    avg_metrics = {}
    for task_type in task_metric_totals:
        avg_metrics[task_type] = {}
        for metric_name in task_metric_totals[task_type]:
            if task_metric_counts[task_type][metric_name] > 0:
                avg_value = (
                    task_metric_totals[task_type][metric_name] / 
                    task_metric_counts[task_type][metric_name]
                )
                # Convert ECR and Pass to percentage (0-100)
                if metric_name in ['ECR', 'Pass']:
                    avg_value = avg_value * 100
                avg_metrics[task_type][metric_name] = avg_value
    
    # Create merged result file
    merged_result = {
        'config': {
            'model_name': model_name,
            'modality': modality_suffix,
            'num_shards': num_shards,
            'merged_timestamp': datetime.now().isoformat(),
            'shard_info': shard_info
        },
        'aggregate_metrics': avg_metrics,
        'results': all_results
    }
    
    # Save merged results
    output_file = os.path.join(merged_dir, f'results_merged_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Merge completed successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {merged_dir}")
    print(f"Output file:      {os.path.basename(output_file)}")
    print(f"Total results:    {len(all_results)}")
    print(f"Unique shards:    {len(shard_info)}")
    
    if avg_metrics:
        print(f"\nAggregate metrics by question type:")
        for task_type in sorted(avg_metrics.keys()):
            print(f"\n  {task_type}:")
            for metric_name, metric_value in sorted(avg_metrics[task_type].items()):
                if isinstance(metric_value, float):
                    print(f"    {metric_name}: {metric_value:.4f}")
                else:
                    print(f"    {metric_name}: {metric_value}")
    
    return merged_result


def main():
    parser = argparse.ArgumentParser(
        description='Merge results from multiple shards of a specific modality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_shards_by_modality.py --result_dir ../result/qwen3vl_local_a100_default --modality text_json --num_shards 3
  python merge_shards_by_modality.py --result_dir ../result/qwen3vl_local_a100_default --modality image --num_shards 3
  python merge_shards_by_modality.py --result_dir ../result/qwen3vl_local_a100_default --modality mix_html --num_shards 3
        """
    )
    
    parser.add_argument('--result_dir', type=str, 
                        default='../result/qwen3vl_local_a100_default',
                        help='Base directory containing shard results')
    parser.add_argument('--model_name', type=str, 
                        default='Qwen3-VL-8B-Instruct',
                        help='Model name (default: Qwen3-VL-8B-Instruct)')
    parser.add_argument('--modality', type=str, required=True,
                        help='Modality suffix (e.g., text_json, mix_html, image, text_csv)')
    parser.add_argument('--num_shards', type=int, default=3,
                        help='Number of shards to merge (default: 3)')
    
    args = parser.parse_args()
    
    # Convert to absolute path if needed
    result_dir = os.path.abspath(args.result_dir)
    
    if not os.path.exists(result_dir):
        print(f"❌ ERROR: Result directory not found: {result_dir}")
        return 1
    
    result = merge_shard_results(
        result_dir,
        args.model_name,
        args.modality,
        args.num_shards
    )
    
    return 0 if result else 1


if __name__ == '__main__':
    exit(main())
