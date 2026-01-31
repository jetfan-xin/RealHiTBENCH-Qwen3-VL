#!/usr/bin/env python3
"""
Test script to verify text truncation functionality.
Tests the truncation on known problematic samples before running full inference.
"""

import sys
import os

# Add parent directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Import the truncation function
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("TEXT TRUNCATION FUNCTIONALITY TEST")
print("="*80)

# Test 1: Load processor and test truncation function
print("\n1. Loading processor...")
try:
    from transformers import AutoProcessor
    model_dir = "/data/pan/4xin/models/Qwen3-VL-8B-Instruct"
    processor = AutoProcessor.from_pretrained(model_dir)
    print("   ✓ Processor loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load processor: {e}")
    sys.exit(1)

# Test 2: Import truncation function
print("\n2. Importing truncation function...")
try:
    from inference_qwen3vl_local_a100_truncate import truncate_text_if_needed, MAX_INPUT_TOKENS
    print(f"   ✓ Truncation function imported (MAX_INPUT_TOKENS = {MAX_INPUT_TOKENS:,})")
except Exception as e:
    print(f"   ✗ Failed to import: {e}")
    sys.exit(1)

# Test 3: Test with known problematic HTML file
print("\n3. Testing with economy-table14_swap.html (334K tokens expected)...")
try:
    html_file = "/data/pan/4xin/datasets/RealHiTBench/html/economy-table14_swap.html"
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print(f"   - Original size: {len(html_content):,} chars")
    
    # Test truncation
    test_prompt = f"Analyze this HTML table:\n{html_content}\n\nQuestion: test"
    truncated_text, original_tokens, was_truncated = truncate_text_if_needed(
        test_prompt, processor, MAX_INPUT_TOKENS
    )
    
    if was_truncated:
        print(f"   ✓ Truncation triggered: {original_tokens:,} → {len(truncated_text)} chars")
        print(f"   ✓ Truncation successful (should be ~100K tokens)")
    else:
        print(f"   ✗ Truncation NOT triggered (tokens: {original_tokens:,})")
        
except Exception as e:
    print(f"   ✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test with normal-sized HTML
print("\n4. Testing with normal-sized HTML (science-table72_swap.html)...")
try:
    html_file = "/data/pan/4xin/datasets/RealHiTBench/html/science-table72_swap.html"
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print(f"   - Original size: {len(html_content):,} chars")
    
    test_prompt = f"Analyze this HTML table:\n{html_content}\n\nQuestion: test"
    truncated_text, original_tokens, was_truncated = truncate_text_if_needed(
        test_prompt, processor, MAX_INPUT_TOKENS
    )
    
    if not was_truncated:
        print(f"   ✓ No truncation needed (tokens: {original_tokens:,})")
    else:
        print(f"   ✗ Unexpectedly truncated (tokens: {original_tokens:,})")
        
except Exception as e:
    print(f"   ✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Verify output directory structure
print("\n5. Checking output directory structure...")
output_base = os.path.abspath('../result/qwen3vl_local_a100_truncate')
if not os.path.exists(output_base):
    os.makedirs(output_base, exist_ok=True)
    print(f"   ✓ Created output directory: {output_base}")
else:
    print(f"   ✓ Output directory exists: {output_base}")

print("\n"+"="*80)
print("ALL TESTS COMPLETED")
print("="*80)
print("\nNext steps:")
print("  1. Run: python run_text_html_truncate.py")
print("  2. Run: python run_mix_html_truncate.py")
print("  3. Check results in: result/qwen3vl_local_a100_truncate/")
print("="*80)
