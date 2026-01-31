#!/usr/bin/env python3
"""
Run text_html inference with text truncation enabled.
This script uses the new inference_qwen3vl_local_a100_truncate.py with automatic text truncation.
"""

import subprocess
import sys
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_script = os.path.join(script_dir, "inference_qwen3vl_local_a100_truncate.py")
    
    # Configuration
    modality = "text"
    format_type = "html"
    batch_size = 1  # Use batch_size=1 for large HTML tables
    model_dir = "/data/pan/4xin/models/Qwen3-VL-8B-Instruct"
    data_path = "/data/pan/4xin/datasets/RealHiTBench"
    qa_path = "/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data"
    
    # Build command
    cmd = [
        sys.executable,
        inference_script,
        "--modality", modality,
        "--format", format_type,
        "--model_dir", model_dir,
        "--data_path", data_path,
        "--qa_path", qa_path,
        "--use_sc_filled",
        "--batch_size", str(batch_size),
        "--resume"
    ]
    
    print(f"Running text_html inference with text truncation...")
    print(f"MAX_INPUT_TOKENS = 100,000")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the inference
    result = subprocess.run(cmd, cwd=script_dir)
    
    if result.returncode == 0:
        print(f"\n✓ Successfully completed {modality}_{format_type} inference with truncation")
    else:
        print(f"\n✗ Failed {modality}_{format_type} inference with exit code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
