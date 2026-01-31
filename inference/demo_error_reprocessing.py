#!/usr/bin/env python3
"""
Demonstrate the ERROR sample reprocessing feature.
Shows how the truncate script handles OOM errors from previous runs.
"""

import json
import os

print("="*80)
print("ERROR SAMPLE REPROCESSING DEMONSTRATION")
print("="*80)

# Simulate loading an existing checkpoint with OOM errors
print("\n1. Simulating checkpoint with OOM errors...")

# Create a mock checkpoint similar to the real one
mock_checkpoint = {
    "results": [
        {"id": 1, "Prediction": "Normal answer", "Metrics": {"F1": 100.0}},
        {"id": 2, "Prediction": "Another normal answer", "Metrics": {"F1": 95.0}},
        {"id": 2747, "Prediction": "[ERROR] OOM: Text too large (334159 tokens)", "Metrics": {"F1": 0.0}},
        {"id": 2748, "Prediction": "[ERROR] OOM: Text too large (334159 tokens)", "Metrics": {"F1": 0.0}},
        {"id": 2749, "Prediction": "[ERROR] OOM: Text too large (334162 tokens)", "Metrics": {"F1": 0.0}},
        {"id": 3, "Prediction": "Normal answer", "Metrics": {"F1": 88.0}},
    ],
    "processed_ids": [1, 2, 3, 2747, 2748, 2749]
}

print(f"   Total results in checkpoint: {len(mock_checkpoint['results'])}")
print(f"   Processed IDs: {mock_checkpoint['processed_ids']}")

# Simulate the new checkpoint loading logic
print("\n2. Applying new checkpoint loading logic...")

processed_ids = set()
error_ids = set()
successful_results = []

for result in mock_checkpoint['results']:
    if result['Prediction'].startswith('[ERROR'):
        error_ids.add(result['id'])
        print(f"   ⚠️  Will reprocess ID {result['id']}: {result['Prediction'][:60]}...")
    else:
        successful_results.append(result)
        processed_ids.add(result['id'])

print(f"\n3. Results after processing:")
print(f"   ✅ Successful results: {len(successful_results)} (IDs: {sorted(processed_ids)})")
print(f"   ⚠️  Failed results (will retry): {len(error_ids)} (IDs: {sorted(error_ids)})")

# Show what happens when processing
print("\n4. During inference:")
print("   - IDs 1, 2, 3 will be skipped (already processed successfully)")
print("   - IDs 2747, 2748, 2749 will be REPROCESSED with text truncation")
print("   - Expected outcome: truncate 334K tokens → 100K tokens → successful inference")

print("\n" + "="*80)
print("REAL-WORLD IMPACT")
print("="*80)

real_oom_ids = [2747, 2748, 2749, 2750, 2751, 2758, 2759, 2760, 2761, 2762, 2763, 
                2966, 2967, 2968, 3019, 3020, 3021]

print(f"\nFrom actual checkpoint analysis:")
print(f"  - text_html has {len(real_oom_ids)} OOM errors")
print(f"  - mix_html has {len(real_oom_ids)} OOM errors")
print(f"  - Total samples to reprocess: {len(real_oom_ids)}")
print(f"\nThese samples will be automatically retried when running:")
print(f"  python run_text_html_truncate.py --resume")
print(f"  python run_mix_html_truncate.py --resume")

print("\n" + "="*80)
print("KEY FEATURES")
print("="*80)
print("""
✅ Automatic ERROR detection
   - Scans checkpoint for results starting with '[ERROR'
   
✅ Intelligent reprocessing
   - Removes ERROR results from successful list
   - Adds their IDs to retry queue
   
✅ Text truncation on retry
   - MAX_INPUT_TOKENS = 100,000
   - Prevents OOM during reprocessing
   
✅ Independent output
   - Saves to: result/qwen3vl_local_a100_truncate/
   - Does not overwrite original results
   
✅ Progress tracking
   - Shows successful vs failed counts on resume
   - Clear logging of reprocessed samples
""")

print("="*80)
