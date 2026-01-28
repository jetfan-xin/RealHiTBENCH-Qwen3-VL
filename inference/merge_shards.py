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
    
    # Calculate aggregate metrics
    metric_totals = defaultdict(float)
    metric_counts = defaultdict(int)
    
    for result in all_results:
        for key, value in result.items():
            if isinstance(value, (int, float)) and key not in ['id']:
                metric_totals[key] += value
                metric_counts[key] += 1
    
    avg_metrics = {}
    for key in metric_totals:
        if metric_counts[key] > 0:
            avg_metrics[key] = metric_totals[key] / metric_counts[key]
    
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
        print(f"\nAggregate metrics:")
        for key, value in sorted(avg_metrics.items()):
            print(f"  {key}: {value:.4f}")
    
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
