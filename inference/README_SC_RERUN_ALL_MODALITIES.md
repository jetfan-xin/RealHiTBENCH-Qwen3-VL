# Structure Comprehending Re-inference Guide

This directory contains scripts for re-running Structure Comprehending inference with updated QA data and merging the results.

## Overview

The updated QA data is in `/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data/QA_final_sc_filled.json`. This workflow:

1. **Re-runs inference** across 9 modalities, where image and mix_* are SC-only while text_* runs all question types
2. **Runs in parallel** on 4 GPUs (0, 1, 2, 3)
3. **Merges new results** with existing checkpoint_merged.json files (SC items only)
4. **Saves output** to results_sc_filled.json with updated SC items

## Quick Start

### Step 1: Run Structure Comprehending Inference

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference
./run_all_sc_parallel.sh
```

This will:
- Run **image** (SC-only) on GPU 0
- Run **mix_csv** (SC-only) on GPU 1
- Run **mix_html** (SC-only) on GPU 2
- Run **mix_latex** (SC-only) on GPU 3, followed by **mix_markdown** (SC-only) on GPU 3 (sequential)
- Then run **text_latex** and **text_markdown** (all question types) on GPU 0 (sequential after image)
- Then run **text_csv** (all question types) on GPU 1 (sequential after mix_csv)
- Then run **text_html** (all question types) on GPU 2 (sequential after mix_html)

Logs will be saved to: `../result/qwen3vl_local_a100/sc_rerun_logs/`

### Step 1b: Run on GPUs 1/2/3 Only

If only GPUs **1, 2, 3** are available:

```bash
cd /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference
./run_all_sc_parallel_3gpu.sh
```

GPU allocation (sequential on each GPU):
- GPU 1: **image → text_latex → text_html**
- GPU 2: **mix_latex → mix_csv → text_csv**
- GPU 3: **mix_html → mix_markdown → text_markdown**

### Step 2: Merge Results

After all inference completes successfully:

```bash
python merge_sc_results.py
```

This will create `results_sc_filled.json` in each modality's merged folder.

## File Structure

### Inference Scripts (Individual Modality)

- `run_sc_image.py` - Image modality
- `run_sc_mix_csv.py` - Mix CSV modality
- `run_sc_mix_html.py` - Mix HTML modality
- `run_sc_mix_latex.py` - Mix LaTeX modality
- `run_sc_mix_markdown.py` - Mix Markdown modality
- `run_text_csv.py` - Text CSV modality
- `run_text_html.py` - Text HTML modality
- `run_text_latex.py` - Text LaTeX modality
- `run_text_markdown.py` - Text Markdown modality

Each script:
- For image and mix_* scripts:
  - Uses `--use_sc_filled` flag to read QA_final_sc_filled.json (which has `_swap` suffix filenames)
  - Filters to `QuestionType == "Structure Comprehending"`
- For text_* scripts:
  - Uses `QA_final.json` (all question types, without `_swap` suffix)
  - Runs all question types (no `QuestionType` filter)
- Uses `--qa_path` to specify the QA JSON file location
- Saves to modality-specific checkpoint.json

### Launcher Script

- `run_all_sc_parallel.sh` - Orchestrates parallel execution across GPUs
- `run_all_sc_parallel_3gpu.sh` - Orchestrates execution on GPUs 1/2/3 only

### Merge Script

- `merge_sc_results.py` - Merges new SC results with existing checkpoint_merged.json

## GPU Allocation

| GPU | Modality 1 | Modality 2 | Modality 3 |
|-----|-----------|-----------|------------|
| 0   | image     | text_latex | text_markdown (sequential) |
| 1   | mix_csv   | text_csv   | - |
| 2   | mix_html  | text_html  | - |
| 3   | mix_latex | mix_markdown (sequential) | - |

## GPU Allocation (3 GPUs: 1/2/3)

| GPU | Sequential Modalities |
|-----|------------------------|
| 1   | image → text_latex → text_html |
| 2   | mix_latex → mix_csv → text_csv |
| 3   | mix_html → mix_markdown → text_markdown |

## Input/Output Files

### Input Files

1. **QA Data**:
   - `/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data/QA_final_sc_filled.json`

2. **Existing Merged Results**:
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_image_a100_merged/checkpoint_merged.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_mix_csv_a100_merged/checkpoint_merged.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_mix_html_a100_merged/checkpoint_merged.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_mix_latex_a100_merged/checkpoint_merged.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_mix_markdown_a100_merged/checkpoint_merged.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_csv_a100_merged/checkpoint_merged.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_html_a100_merged/checkpoint_merged.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_latex_a100_merged/checkpoint_merged.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_markdown_a100_merged/checkpoint_merged.json`

### Output Files

1. **New SC Checkpoints** (created during inference):
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_image_a100/checkpoint.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_mix_csv_a100/checkpoint.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_mix_html_a100/checkpoint.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_mix_latex_a100/checkpoint.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_mix_markdown_a100/checkpoint.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_csv_a100/checkpoint.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_html_a100/checkpoint.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_latex_a100/checkpoint.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_markdown_a100/checkpoint.json`

2. **Merged Results** (created by merge script):
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_image_a100_merged/results_sc_filled.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_mix_csv_a100_merged/results_sc_filled.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_mix_html_a100_merged/results_sc_filled.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_mix_latex_a100_merged/results_sc_filled.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_mix_markdown_a100_merged/results_sc_filled.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_csv_a100_merged/results_sc_filled.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_html_a100_merged/results_sc_filled.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_latex_a100_merged/results_sc_filled.json`
   - `../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_text_markdown_a100_merged/results_sc_filled.json`

3. **Logs**:
   - `../result/qwen3vl_local_a100/sc_rerun_logs/image_gpu0.log`
   - `../result/qwen3vl_local_a100/sc_rerun_logs/mix_csv_gpu1.log`
   - `../result/qwen3vl_local_a100/sc_rerun_logs/mix_html_gpu2.log`
   - `../result/qwen3vl_local_a100/sc_rerun_logs/mix_latex_gpu3.log`
   - `../result/qwen3vl_local_a100/sc_rerun_logs/mix_markdown_gpu3.log`
   - `../result/qwen3vl_local_a100/sc_rerun_logs/text_latex_gpu0.log`
   - `../result/qwen3vl_local_a100/sc_rerun_logs/text_markdown_gpu0.log`
   - `../result/qwen3vl_local_a100/sc_rerun_logs/text_csv_gpu1.log`
   - `../result/qwen3vl_local_a100/sc_rerun_logs/text_html_gpu2.log`

## Merge Logic

The merge script:

1. **Loads** new SC checkpoint.json and existing checkpoint_merged.json
2. **Identifies** Structure Comprehending items by `QuestionType == "Structure Comprehending"`
3. **Replaces** SC items in checkpoint_merged.json with new results based on matching IDs
4. **Keeps** all non-SC items unchanged (text_* non-SC outputs are not merged)
5. **Recomputes** aggregate metrics grouped by QuestionType
6. **Saves** to results_sc_filled.json with metadata

## Manual Execution (Individual Modality)

To run a single modality manually:

```bash
# Example: Run only image modality on GPU 0
CUDA_VISIBLE_DEVICES=0 python run_sc_image.py

# Example: Run only mix_csv modality on GPU 1
CUDA_VISIBLE_DEVICES=1 python run_sc_mix_csv.py

# Example: Run only text_html modality on GPU 2
CUDA_VISIBLE_DEVICES=2 python run_text_html.py
```

## Troubleshooting

### Check GPU availability
```bash
nvidia-smi
```

### Check if inference is running
```bash
# Look for python processes
ps aux | grep inference_qwen3vl_local_a100

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### View logs in real-time
```bash
tail -f ../result/qwen3vl_local_a100/sc_rerun_logs/image_gpu0.log
```

### Verify checkpoint files were created
```bash
ls -lh ../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_*/checkpoint.json
```

### Check merge results
```bash
ls -lh ../result/qwen3vl_local_a100/Qwen3-VL-8B-Instruct_*_merged/results_sc_filled.json
```

## Modifications to inference_qwen3vl_local_a100.py

Added support for `--use_sc_filled` flag:

```python
parser.add_argument('--use_sc_filled', action='store_true',
                    help='Use QA_final_sc_filled.json with updated structure comprehending data')
```

Data loading logic now checks:
```python
if opt.use_sc_filled:
    qa_file = 'QA_final_sc_filled.json'
elif opt.use_long:
    qa_file = 'QA_long.json'
else:
    qa_file = 'QA_final.json'
```

## Notes

- **Batch size**: Default is 8, adjust if needed for memory constraints
- **No sharding**: Each modality runs on a single GPU without data sharding
- **Clean inference**: Uses `--no_resume` to ensure fresh inference with updated data
- **Sequential on GPU 0/1/2/3**: text_* runs after image/mix tasks on its assigned GPU; mix_markdown waits for mix_latex
- **Text_* scope**: text_* runs all question types, but the merge step only updates SC items
- **Metric recomputation**: Aggregate metrics are automatically recalculated after merge

## Expected Runtime

- Image and mix_* (SC-only): ~10-30 minutes each (depends on number of SC items)
- Text_* (all question types): ~30-90 minutes each
- Total parallel time: ~60-120 minutes (limited by slowest text_* + sequential dependencies)

## Contact

For issues or questions, check the logs in `sc_rerun_logs/` directory.
