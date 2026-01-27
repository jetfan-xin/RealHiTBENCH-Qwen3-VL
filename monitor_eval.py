#!/usr/bin/env python
"""Real-time monitoring dashboard for Qwen3-VL evaluations."""

import json
import time
from pathlib import Path
from datetime import datetime
import sys

def get_eval_progress(folder_name):
    """Get evaluation progress from checkpoint."""
    result_dir = Path("/ltstorage/home/4xin/image_table/RealHiTBench/result/qwen3vl_local")
    checkpoint = result_dir / folder_name / "checkpoint.json"
    
    if not checkpoint.exists():
        return None, None, None
    
    try:
        with open(checkpoint) as f:
            data = json.load(f)
        processed = len(data.get('processed_ids', []))
        total = len(data.get('results', []))
        
        # Calculate average processing time
        times = []
        for result in data.get('results', []):
            if isinstance(result.get('ProcessingTime'), (int, float)):
                times.append(result['ProcessingTime'])
        
        avg_time = sum(times) / len(times) if times else 0
        
        return processed, total, avg_time
    except Exception as e:
        print(f"Error reading {checkpoint}: {e}")
        return None, None, None

def print_dashboard():
    """Print monitoring dashboard."""
    print("\033[2J\033[H", end="")  # Clear screen
    
    print("=" * 90)
    print(f"QWEN3-VL EVALUATION MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    
    evaluations = {
        "IMAGE": ("Qwen3-VL-8B-Instruct_image", "GPU 0"),
        "TEXT-HTML": ("Qwen3-VL-8B-Instruct_text_html", "GPU 1"),
        "TEXT-LATEX": ("Qwen3-VL-8B-Instruct_text_latex", "GPU 2"),
        "MIX-LATEX": ("Qwen3-VL-8B-Instruct_mix_latex", "GPU 4"),
    }
    
    est_total = 3071
    
    for name, (folder, gpu) in evaluations.items():
        processed, total, avg_time = get_eval_progress(folder)
        
        if processed is None:
            print(f"{name:12s} ({gpu:6s}) | NOT STARTED")
        else:
            pct = int((processed / est_total) * 100)
            progress_bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            
            # Estimate remaining time
            if avg_time > 0:
                remaining = (est_total - processed) * avg_time
                remaining_str = f"{remaining/3600:.1f}h" if remaining > 3600 else f"{remaining/60:.1f}m"
            else:
                remaining_str = "calculating..."
            
            print(f"{name:12s} ({gpu:6s}) | {progress_bar} | {processed:4d}/{est_total:4d} ({pct:3d}%) | "
                  f"Avg: {avg_time:.2f}s/q | ETA: {remaining_str}")
    
    print("=" * 90)
    print("Commands:")
    print("  • watch -n 5 'python monitor_eval.py'  # Refresh every 5 seconds")
    print("  • tail -f ../result/eval_*.log          # View logs")
    print("  • ps aux | grep inference_qwen3vl_local # Check processes")
    print("=" * 90)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        # Continuous monitoring mode
        while True:
            try:
                print_dashboard()
                time.sleep(5)
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
                break
    else:
        # Single print mode
        print_dashboard()
