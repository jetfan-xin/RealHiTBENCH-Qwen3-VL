#!/usr/bin/env python3
"""
Run inference for TEXT_HTML modality (all question types).
This script runs on a single GPU without sharding.
"""

import subprocess
import sys
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_script = os.path.join(script_dir, "inference_qwen3vl_local_a100.py")
    
    # Configuration
    modality = "text"
    format_type = "html"
    batch_size = 1
    model_dir = "/data/pan/4xin/models/Qwen3-VL-8B-Instruct"
    data_path = "/data/pan/4xin/datasets/RealHiTBench"
    # QA JSON files are in the project's data directory
    qa_path = os.path.join(os.path.dirname(script_dir), "data")
    
    # Build command - use QA_final_sc_filled.json for all tasks
    cmd = [
        sys.executable,
        inference_script,
        "--modality", modality,
        "--format", format_type,
        "--model_dir", model_dir,
        "--data_path", data_path,
        "--qa_path", qa_path,
        "--batch_size", str(batch_size),
        "--use_sc_filled",  # Use QA_final_sc_filled.json
        "--resume"
    ]
    
    print(f"Running inference for {modality}_{format_type} (all question types)...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the inference
    result = subprocess.run(cmd, cwd=script_dir)
    
    if result.returncode == 0:
        print(f"\n✓ Successfully completed {modality}_{format_type} inference")
    else:
        print(f"\n✗ Failed {modality}_{format_type} inference with exit code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
