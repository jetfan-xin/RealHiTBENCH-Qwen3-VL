#!/usr/bin/env python3
"""
Re-run Structure Comprehending inference for TEXT_HTML modality with updated QA data.
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
    question_type = "Structure Comprehending"
    batch_size = 8
    model_dir = "/data/pan/4xin/models/Qwen3-VL-8B-Instruct"
    data_path = "/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL"
    output_dir = f"/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/result/qwen3vl_local_a100"
    
    # Build command
    cmd = [
        sys.executable,
        inference_script,
        "--modality", modality,
        "--format", format_type,
        "--model_dir", model_dir,
        "--data_path", data_path,
        "--output_dir", output_dir,
        "--question_type", question_type,
        "--batch_size", str(batch_size),
        "--use_sc_filled",  # Use updated QA data
        "--no_resume"  # Start fresh for clean re-inference
    ]
    
    print(f"Running Structure Comprehending inference for {modality}_{format_type}...")
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
