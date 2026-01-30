#!/usr/bin/env python3
"""
Merge Script for Structure Comprehending Results

This script merges newly generated Structure Comprehending (SC) results with existing 
checkpoint_merged.json files. It replaces SC items by matching IDs and recomputes 
aggregate metrics.

Usage:
    python merge_sc_results.py

Input:
    - New SC results: result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_<modality>_a100/checkpoint.json
    - Existing merged: result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_<modality>_a100_merged/checkpoint_merged.json

Output:
    - Updated results: result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_<modality>_a100_merged/results_sc_filled.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add parent directory to path for imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

MODALITIES = [
    "image",
    "mix_csv", 
    "mix_html",
    "mix_latex",
    "mix_markdown",
    "text_csv",
    "text_html",
    "text_latex",
    "text_markdown"
]

BASE_DIR = Path("/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/result/qwen3vl_local_a100")
MODEL_NAME = "Qwen3-VL-8B-Instruct"


def load_json_file(filepath: Path) -> Dict:
    """Load JSON file with error handling."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Dict, filepath: Path):
    """Save JSON file with pretty formatting."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filepath}")


def compute_aggregate_metrics(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Compute aggregate metrics grouped by QuestionType.
    
    Args:
        results: List of result dictionaries with 'QuestionType' and 'Metrics' fields
        
    Returns:
        Dictionary mapping QuestionType to averaged metrics
    """
    # Group results by QuestionType
    grouped = defaultdict(list)
    for result in results:
        qtype = result.get('QuestionType', 'Unknown')
        metrics = result.get('Metrics', {})
        if metrics:  # Only include if metrics exist
            grouped[qtype].append(metrics)
    
    # Compute averages for each question type
    aggregated = {}
    for qtype, metrics_list in grouped.items():
        if not metrics_list:
            continue
            
        # Get all metric keys from first item
        metric_keys = set()
        for metrics in metrics_list:
            metric_keys.update(metrics.keys())
        
        # Compute average for each metric
        avg_metrics = {}
        for key in metric_keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        aggregated[qtype] = avg_metrics
    
    return aggregated


def merge_modality_results(modality: str) -> bool:
    """
    Merge Structure Comprehending results for a specific modality.
    
    Args:
        modality: Modality name (e.g., 'image', 'mix_csv')
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Processing modality: {modality}")
    print(f"{'='*80}")
    
    # Determine folder names based on modality
    if modality == "image":
        checkpoint_folder = f"{MODEL_NAME}_{modality}_a100"
        merged_folder = f"{MODEL_NAME}_{modality}_a100_merged"
    else:  # mix_* modalities
        checkpoint_folder = f"{MODEL_NAME}_{modality}_a100"
        merged_folder = f"{MODEL_NAME}_{modality}_a100_merged"
    
    # File paths
    new_sc_file = BASE_DIR / checkpoint_folder / "checkpoint.json"
    existing_merged_file = BASE_DIR / merged_folder / "checkpoint_merged.json"
    output_file = BASE_DIR / merged_folder / "results_sc_filled.json"
    
    # Check if new SC results exist
    if not new_sc_file.exists():
        print(f"  ⚠ Warning: New SC checkpoint not found: {new_sc_file}")
        print(f"  Skipping {modality}")
        return False
    
    # Check if existing merged file exists
    if not existing_merged_file.exists():
        print(f"  ⚠ Warning: Existing merged file not found: {existing_merged_file}")
        print(f"  Skipping {modality}")
        return False
    
    try:
        # Load files
        print(f"  Loading new SC results: {new_sc_file.name}")
        new_sc_data = load_json_file(new_sc_file)
        
        print(f"  Loading existing merged: {existing_merged_file.name}")
        existing_data = load_json_file(existing_merged_file)
        
        # Extract results
        new_results = new_sc_data.get('results', [])
        existing_results = existing_data.get('results', [])
        
        print(f"  New SC results: {len(new_results)} items")
        print(f"  Existing results: {len(existing_results)} items")
        
        # Build ID to result mapping for new SC results
        new_sc_by_id = {r['id']: r for r in new_results}
        print(f"  New SC IDs: {sorted(new_sc_by_id.keys())[:10]}..." if len(new_sc_by_id) > 10 else f"  New SC IDs: {sorted(new_sc_by_id.keys())}")
        
        # Merge: Replace SC items in existing results
        merged_results = []
        replaced_count = 0
        kept_count = 0
        
        for result in existing_results:
            result_id = result['id']
            question_type = result.get('QuestionType', '')
            
            # If this is an SC item and we have new data for it, replace it
            if question_type == 'Structure Comprehending' and result_id in new_sc_by_id:
                merged_results.append(new_sc_by_id[result_id])
                replaced_count += 1
            else:
                merged_results.append(result)
                kept_count += 1
        
        print(f"  Replaced {replaced_count} SC items")
        print(f"  Kept {kept_count} non-SC items")
        print(f"  Total merged results: {len(merged_results)}")
        
        # Recompute aggregate metrics
        print(f"  Recomputing aggregate metrics...")
        aggregate_metrics = compute_aggregate_metrics(merged_results)
        
        print(f"  Aggregate metrics computed for {len(aggregate_metrics)} question types:")
        for qtype, metrics in aggregate_metrics.items():
            print(f"    - {qtype}: {len(metrics)} metrics")
        
        # Build output data
        output_data = {
            'processed_ids': sorted([r['id'] for r in merged_results]),
            'results': merged_results,
            'aggregate_metrics': aggregate_metrics,
            'metadata': {
                'total_results': len(merged_results),
                'sc_results_replaced': replaced_count,
                'non_sc_results_kept': kept_count,
                'merge_timestamp': existing_data.get('metadata', {}).get('merge_timestamp', 'unknown'),
                'sc_rerun_applied': True
            }
        }
        
        # Save merged results
        save_json_file(output_data, output_file)
        
        print(f"  ✓ Successfully merged {modality}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {modality}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("Structure Comprehending Results Merge Script")
    print("="*80)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Modalities to process: {', '.join(MODALITIES)}")
    
    # Process each modality
    results = {}
    for modality in MODALITIES:
        success = merge_modality_results(modality)
        results[modality] = success
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    successful = [m for m, success in results.items() if success]
    failed = [m for m, success in results.items() if not success]
    
    print(f"\nSuccessful ({len(successful)}):")
    for modality in successful:
        print(f"  ✓ {modality}")
    
    if failed:
        print(f"\nFailed or Skipped ({len(failed)}):")
        for modality in failed:
            print(f"  ✗ {modality}")
    
    print(f"\n{'='*80}")
    print("Merge Complete!")
    print(f"{'='*80}\n")
    
    # Exit with appropriate code
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()
