#!/usr/bin/env python3
"""
Merge results from multiple shards back into a single result file.
Usage: python merge_shards.py --output_dir ../result/qwen3vl_local --model_name Qwen3-VL-8B-Instruct --modality image --num_shards 3
"""

import os
import json
import argparse
from collections import defaultdict


def merge_shard_results(output_base, model_name, modality, num_shards, format_type=None):
    """Merge results from multiple shards into a single result."""
    
    # Build modality suffix
    modality_suffix = modality
    if modality != 'image' and format_type:
        modality_suffix += f"_{format_type}"
    
    # Find all shard directories
    all_results = []
    all_processed_ids = set()
    aggregate_metrics = defaultdict(list)
    
    for shard_id in range(num_shards):
        shard_dir = f'{output_base}/{model_name}_{modality_suffix}_shard{shard_id}'
        
        if not os.path.exists(shard_dir):
            print(f"Warning: Shard {shard_id} directory not found: {shard_dir}")
            continue
        
        # Load checkpoint
        checkpoint_file = f'{shard_dir}/checkpoint_batch.json'
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                processed_ids = data.get('processed_ids', [])
                results = data.get('results', [])
                
                # Filter to only include results for IDs in processed_ids
                for r in results:
                    if r.get('id') in processed_ids and r.get('id') not in all_processed_ids:
                        all_results.append(r)
                        all_processed_ids.add(r['id'])
                
                print(f"Shard {shard_id}: {len(processed_ids)} processed, {len(results)} results")
        else:
            # Try non-batch checkpoint
            checkpoint_file = f'{shard_dir}/checkpoint.json'
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    processed_ids = data.get('processed_ids', [])
                    results = data.get('results', [])
                    
                    for r in results:
                        if r.get('id') in processed_ids and r.get('id') not in all_processed_ids:
                            all_results.append(r)
                            all_processed_ids.add(r['id'])
                    
                    print(f"Shard {shard_id}: {len(processed_ids)} processed, {len(results)} results")
    
    # Create merged output directory
    merged_dir = f'{output_base}/{model_name}_{modality_suffix}_merged'
    os.makedirs(merged_dir, exist_ok=True)
    
    # Calculate aggregate metrics grouped by task type (QuestionType)
    task_metric_totals = defaultdict(lambda: defaultdict(float))
    task_metric_counts = defaultdict(lambda: defaultdict(int))
    
    for result in all_results:
        # Get task type, default to 'Unknown' if not present
        task_type = result.get('QuestionType', 'Unknown')
        
        # Process metrics from the 'Metrics' sub-dictionary if present
        if 'Metrics' in result and isinstance(result['Metrics'], dict):
            for metric_name, metric_value in result['Metrics'].items():
                # Convert boolean to numeric (True->1, False->0)
                if isinstance(metric_value, bool):
                    metric_value = 1.0 if metric_value else 0.0
                # Convert string 'true'/'false' to numeric
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
        
        # Also process top-level numeric fields (like ProcessingTime)
        for key, value in result.items():
            if key in ['ProcessingTime'] and isinstance(value, (int, float)):
                task_metric_totals[task_type][key] += value
                task_metric_counts[task_type][key] += 1
    
    # Calculate averages for each task type
    avg_metrics = {}
    for task_type in task_metric_totals:
        avg_metrics[task_type] = {}
        for metric_name in task_metric_totals[task_type]:
            if task_metric_counts[task_type][metric_name] > 0:
                avg_metrics[task_type][metric_name] = (
                    task_metric_totals[task_type][metric_name] / 
                    task_metric_counts[task_type][metric_name]
                )
    
    # Save merged checkpoint
    merged_checkpoint = {
        'processed_ids': list(all_processed_ids),
        'results': all_results,
        'aggregate_metrics': avg_metrics
    }
    
    with open(f'{merged_dir}/checkpoint_merged.json', 'w') as f:
        json.dump(merged_checkpoint, f, indent=2, ensure_ascii=False)
    
    print(f"\nMerged results:")
    print(f"  Total processed: {len(all_processed_ids)}")
    print(f"  Total results: {len(all_results)}")
    print(f"  Output: {merged_dir}/checkpoint_merged.json")
    
    if avg_metrics:
        print(f"\nAggregate metrics by task type:")
        for task_type in sorted(avg_metrics.keys()):
            print(f"\n  {task_type}:")
            for metric_name, metric_value in sorted(avg_metrics[task_type].items()):
                print(f"    {metric_name}: {metric_value:.4f}")
    
    return merged_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Merge results from multiple shards')
    parser.add_argument('--output_dir', type=str, default='../result/qwen3vl_local',
                        help='Base output directory containing shard results')
    parser.add_argument('--model_name', type=str, default='Qwen3-VL-8B-Instruct',
                        help='Model name used in directory naming')
    parser.add_argument('--modality', type=str, default='image',
                        help='Modality (image, text, mix)')
    parser.add_argument('--format', type=str, default=None,
                        help='Format type for text/mix modalities')
    parser.add_argument('--num_shards', type=int, default=3,
                        help='Number of shards to merge')
    
    args = parser.parse_args()
    
    merge_shard_results(
        args.output_dir,
        args.model_name,
        args.modality,
        args.num_shards,
        args.format
    )


if __name__ == '__main__':
    main()
