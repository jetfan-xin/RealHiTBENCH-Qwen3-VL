#!/usr/bin/env python3
"""
Wrapper script for inference_qwen3vl_local_a100_truncate.py with --task_ids support.

This script filters the QA file before running inference, allowing you to specify
specific task IDs to process.

Usage:
    python inference_qwen3vl_local_a100_truncate_with_task_ids.py \\
        --modality text --format html --task_ids "2747,2748,2749"
"""

import sys
import os
import json
import argparse
import tempfile
import subprocess

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run Qwen3-VL inference on specific task IDs')
    parser.add_argument('--task_ids', type=str, required=True,
                        help='Comma-separated list of task IDs (e.g., "1,2,3,100")')
    parser.add_argument('--modality', type=str, required=True,
                        choices=['image', 'text', 'mix'])
    parser.add_argument('--format', type=str, default='html',
                        choices=['html', 'csv', 'markdown', 'latex'])
    parser.add_argument('--model_dir', type=str, 
                        default='/data/pan/4xin/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--data_path', type=str,
                        default='/data/pan/4xin/datasets/RealHiTBench')
    parser.add_argument('--qa_path', type=str,
                        default='/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data')
    parser.add_argument('--use_sc_filled', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--resume', action='store_true', default=True)
    
    args, unknown_args = parser.parse_known_args()
    
    # Parse task IDs
    task_id_list = [int(tid.strip()) for tid in args.task_ids.split(',')]
    task_id_set = set(task_id_list)
    
    print("=" * 80)
    print(f"Running inference on {len(task_id_list)} specific tasks")
    print("=" * 80)
    print(f"Task IDs: {sorted(task_id_list)}")
    print(f"Modality: {args.modality}")
    if args.modality != 'image':
        print(f"Format: {args.format}")
    print()
    
    # Load QA file
    qa_file = 'QA_final_sc_filled.json' if args.use_sc_filled else 'QA_final.json'
    qa_path_full = os.path.join(args.qa_path, qa_file)
    
    with open(qa_path_full, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data['queries'])
    
    # Filter queries by task IDs
    data['queries'] = [q for q in data['queries'] if q['id'] in task_id_set]
    
    print(f"Loaded {original_count} queries from {qa_file}")
    print(f"Filtered to {len(data['queries'])} queries")
    print()
    
    # Create temporary filtered QA file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, 
                                      encoding='utf-8') as tmp_file:
        json.dump(data, tmp_file, indent=2, ensure_ascii=False)
        tmp_qa_file = tmp_file.name
    
    try:
        # Build command for inference script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        inference_script = os.path.join(script_dir, "inference_qwen3vl_local_a100_truncate.py")
        
        # Create temporary directory for filtered QA file
        tmp_qa_dir = os.path.dirname(tmp_qa_file)
        tmp_qa_filename = os.path.basename(tmp_qa_file)
        
        cmd = [
            sys.executable,
            inference_script,
            "--modality", args.modality,
            "--model_dir", args.model_dir,
            "--data_path", args.data_path,
            "--qa_path", tmp_qa_dir,  # Use temp directory
            "--batch_size", str(args.batch_size),
        ]
        
        if args.modality != 'image':
            cmd.extend(["--format", args.format])
        
        if args.resume:
            cmd.append("--resume")
        
        # Add any unknown args (pass-through)
        cmd.extend(unknown_args)
        
        # Temporarily rename the QA file to expected name
        expected_qa_name = 'QA_final_sc_filled.json' if args.use_sc_filled else 'QA_final.json'
        final_tmp_qa = os.path.join(tmp_qa_dir, expected_qa_name)
        
        # Copy to expected name
        import shutil
        shutil.copy(tmp_qa_file, final_tmp_qa)
        
        print(f"Running: {' '.join(cmd)}")
        print("-" * 80)
        print()
        
        # Run inference
        result = subprocess.run(cmd, cwd=script_dir)
        
        if result.returncode == 0:
            print()
            print("=" * 80)
            print(f"✓ Successfully completed inference on {len(task_id_list)} tasks")
            print("=" * 80)
        else:
            print()
            print("=" * 80)
            print(f"✗ Inference failed with exit code {result.returncode}")
            print("=" * 80)
            sys.exit(result.returncode)
    
    finally:
        # Cleanup
        if os.path.exists(tmp_qa_file):
            os.unlink(tmp_qa_file)
        if os.path.exists(final_tmp_qa):
            os.unlink(final_tmp_qa)

if __name__ == '__main__':
    main()
