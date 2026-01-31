"""
Local inference script for Qwen3-VL on RealHiTBench dataset with TEXT TRUNCATION.
Based on inference_qwen3vl_local_a100_truncate.py with added text truncation to prevent OOM.

⚠️ NEW FEATURE: Automatic text truncation for extremely large HTML/text inputs
   - MAX_INPUT_TOKENS = 100,000 (prevents OOM while allowing most tables)
   - Texts exceeding this limit will be truncated with a warning
   - Uses official default max_pixels settings (same as default script)

Supports three input modalities: image-only, text-only, and image+text (mix).
Uses DataParallel for multi-GPU data parallelism.

Usage:
    # Text-only with truncation
    python inference_qwen3vl_local_a100_truncate.py --modality text --format html --model_dir /path/to/model
    
    # Mix with truncation
    python inference_qwen3vl_local_a100_truncate.py --modality mix --format html --model_dir /path/to/model
"""

import sys
import os

# CRITICAL: Disable PIL decompression bomb check BEFORE any other imports
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Allow arbitrarily large images

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import datetime
import json
import ast
import argparse
import re
import time
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

# Import prompts based on modality
from answer_prompt_mlm import Answer_Prompt as Answer_Prompt_Image
from answer_prompt_mlm import User_Prompt as User_Prompt_Image
from answer_prompt_mix import Answer_Prompt as Answer_Prompt_Mix
from answer_prompt_mix import User_Prompt as User_Prompt_Mix
from answer_prompt_llm import Answer_Prompt as Answer_Prompt_Text
from answer_prompt_llm import User_Prompt as User_Prompt_Text

from gpt_eval_prompt import Eval_Prompt
from gpt_eval import get_eval_score
from metrics.qa_metrics import QAMetric
from utils.common_util import read_file
from timeout_decorator import timeout
from utils.chart_process import build_eval_code, surround_pycode_with_main
from utils.chart_metric_util import compute_pie_chart_metric, compute_general_chart_metric

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# TEXT TRUNCATION CONFIGURATION
# ============================================================
# Maximum input tokens allowed before truncation
# Based on analysis of missing IDs, most tables are well below this limit
# Only extreme outliers (like economy-table14_swap with 334K tokens) will be truncated
MAX_INPUT_TOKENS = 100000  # Allow up to 100K tokens before truncation

# File extension mapping
FILE_EXTENSIONS = {
    "latex": "txt",
    "markdown": "md",
    "csv": "csv",
    "html": "html"
}

# Global model and processor (loaded once)
_model = None
_processor = None
_use_data_parallel = False


def load_qwen3_vl_local(model_dir, use_flash_attn=True, use_model_parallel=False):
    """
    Load local Qwen3-VL model with multi-GPU support.
    Same as default version - uses official default processor settings.
    """
    global _model, _processor, _use_data_parallel
    
    if _model is not None and _processor is not None:
        return _model, _processor
    
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    
    num_gpus = torch.cuda.device_count()
    print(f"Loading Qwen3-VL model from {model_dir}...")
    print(f"Available GPUs: {num_gpus}")
    
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    
    if use_model_parallel:
        print("Using MODEL PARALLELISM (device_map='auto') - layers distributed across GPUs")
        try:
            _model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map="auto"
            )
            print(f"Model loaded with {attn_impl} attention")
        except Exception as e:
            print(f"Warning: Could not load with {attn_impl}: {e}")
            print("Falling back to default attention implementation...")
            _model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        _use_data_parallel = False
        if hasattr(_model, 'hf_device_map'):
            devices = set(_model.hf_device_map.values())
            print(f"Model distributed across devices: {devices}")
    else:
        print("Using DATA PARALLELISM (DataParallel) - all GPUs compute simultaneously")
        try:
            _model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            ).cuda()
            print(f"Model loaded with {attn_impl} attention on cuda:0")
        except Exception as e:
            print(f"Warning: Could not load with {attn_impl}: {e}")
            print("Falling back to default attention implementation...")
            _model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
            ).cuda()
        
        if num_gpus > 1:
            print(f"Wrapping model with DataParallel across {num_gpus} GPUs")
            _model = torch.nn.DataParallel(_model)
            _use_data_parallel = True
            print(f"DataParallel enabled on devices: {list(range(num_gpus))}")
        else:
            print("Single GPU detected, DataParallel not needed")
            _use_data_parallel = False
    
    # Use official default processor settings
    _processor = AutoProcessor.from_pretrained(model_dir)
    print(f"Processor configured with OFFICIAL DEFAULT settings")
    print(f"  Text truncation: MAX_INPUT_TOKENS = {MAX_INPUT_TOKENS:,}")
    
    return _model, _processor


def get_base_model(model):
    """Get the base model from DataParallel wrapper if needed."""
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def get_model_device(model):
    """Get the device of the model (first device for DataParallel)."""
    if isinstance(model, torch.nn.DataParallel):
        return torch.device('cuda:0')
    elif hasattr(model, 'device'):
        return model.device
    else:
        return next(model.parameters()).device


def truncate_text_if_needed(messages_text: str, processor, max_tokens: int = MAX_INPUT_TOKENS):
    """
    Check text length and truncate if it exceeds max_tokens limit.
    
    Args:
        messages_text: The input text
        processor: The processor for tokenization
        max_tokens: Maximum allowed tokens
        
    Returns:
        (truncated_text, original_token_count, was_truncated)
    """
    # Create test messages to check token count
    test_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": messages_text}
            ]
        }
    ]
    
    try:
        # Tokenize to check length (without adding to GPU)
        test_inputs = processor.apply_chat_template(
            test_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        input_tokens = test_inputs.input_ids.shape[1]
        
        if input_tokens <= max_tokens:
            return messages_text, input_tokens, False
        
        # Text is too long, need to truncate
        print(f"  [TRUNCATE] Input too large ({input_tokens:,} tokens), truncating to {max_tokens:,}", flush=True)
        
        # Calculate truncation ratio (leave 10% margin for safety)
        char_ratio = (max_tokens / input_tokens) * 0.90
        truncate_length = int(len(messages_text) * char_ratio)
        
        truncated_text = messages_text[:truncate_length]
        truncated_text += "\n\n[...TEXT TRUNCATED DUE TO LENGTH LIMIT...]"
        
        # Verify truncated length
        verify_messages = [{"role": "user", "content": [{"type": "text", "text": truncated_text}]}]
        verify_inputs = processor.apply_chat_template(
            verify_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        final_tokens = verify_inputs.input_ids.shape[1]
        print(f"  [TRUNCATE] Result: {final_tokens:,} tokens (original: {input_tokens:,})", flush=True)
        
        return truncated_text, input_tokens, True
        
    except Exception as e:
        print(f"  [TRUNCATE ERROR] Could not check token count: {e}", flush=True)
        # If tokenization fails, return original text
        return messages_text, -1, False


def get_text_response_local(messages_text: str, model, processor, opt, max_tokens=4096):
    """
    Get response for text-only input with automatic truncation for large inputs.
    
    Args:
        messages_text: The prompt text
        model: Loaded model
        processor: Loaded processor
        opt: Options
        max_tokens: Maximum tokens to generate
        
    Returns:
        Response text, or error message starting with '[ERROR]' if processing fails
    """
    try:
        # STEP 1: Truncate text if needed
        truncated_text, original_tokens, was_truncated = truncate_text_if_needed(
            messages_text, processor, MAX_INPUT_TOKENS
        )
        
        # STEP 2: Create messages with (potentially truncated) text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": truncated_text}
                ]
            }
        ]
        
        # STEP 3: Apply chat template and tokenize
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        device = get_model_device(model)
        inputs = inputs.to(device)
        
        # Log token count
        input_tokens = inputs.input_ids.shape[1]
        if input_tokens > 8000:
            print(f"  Warning: Large text input ({input_tokens:,} tokens)", flush=True)
        
        # STEP 4: Generate response
        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
            }
            if opt.temperature > 0:
                generate_kwargs.update({
                    "temperature": opt.temperature,
                    "do_sample": True,
                    "top_p": opt.top_p,
                    "top_k": opt.top_k,
                    "repetition_penalty": getattr(opt, 'repetition_penalty', 1.0),
                })
            else:
                generate_kwargs["do_sample"] = False
            
            generated_ids = model.generate(**generate_kwargs)
        
        # STEP 5: Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
        
    except torch.cuda.OutOfMemoryError as e:
        # OOM still occurred even after truncation
        print(f"  [OOM ERROR] Text input too large even after truncation ({input_tokens:,} tokens)", flush=True)
        torch.cuda.empty_cache()
        return f"[ERROR] OOM: Text too large ({input_tokens:,} tokens)"
        
    except Exception as e:
        print(f"  [ERROR] Failed to process text: {str(e)}", flush=True)
        return f"[ERROR] {str(e)}"


def get_image_response_local(messages_text: str, image_file: str, model, processor, opt, max_tokens=4096):
    """
    Get response for image input with text truncation support.
    
    Args:
        messages_text: The prompt text
        image_file: Path to image file
        model: Loaded model
        processor: Loaded processor
        opt: Options
        max_tokens: Maximum tokens to generate
        
    Returns:
        Response text, or error message starting with '[ERROR]' if processing fails
    """
    try:
        # Check image size first
        img = Image.open(image_file)
        img_size = img.size
        img_pixels = img_size[0] * img_size[1]
        
        # STEP 1: Truncate text if needed (for mix modality)
        truncated_text, original_tokens, was_truncated = truncate_text_if_needed(
            messages_text, processor, MAX_INPUT_TOKENS
        )
        
        # STEP 2: Create messages with image and (potentially truncated) text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_file}"},
                    {"type": "text", "text": truncated_text}
                ]
            }
        ]
        
        # STEP 3: Apply chat template with image processing
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        device = get_model_device(model)
        inputs = inputs.to(device)
        
        # STEP 4: Generate response
        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
            }
            if opt.temperature > 0:
                generate_kwargs.update({
                    "temperature": opt.temperature,
                    "do_sample": True,
                    "top_p": opt.top_p,
                    "top_k": opt.top_k,
                    "repetition_penalty": getattr(opt, 'repetition_penalty', 1.0),
                })
            else:
                generate_kwargs["do_sample"] = False
            
            base_model = get_base_model(model)
            generated_ids = base_model.generate(**generate_kwargs)
        
        # STEP 5: Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
        
    except torch.cuda.OutOfMemoryError as e:
        # Handle OOM for large images
        print(f"  [OOM ERROR] Image too large to process: {image_file}, size={img_size}", flush=True)
        torch.cuda.empty_cache()
        return f"[ERROR] OOM: Image too large ({img_size})"
    
    except FileNotFoundError as e:
        print(f"  [ERROR] Image file not found: {image_file}", flush=True)
        return f"[ERROR] File not found: {image_file}"
        
    except Exception as e:
        print(f"  [ERROR] Failed to process image {image_file}: {str(e)}", flush=True)
        return f"[ERROR] {str(e)}"


def get_final_answer_local(messages_text: str, answer_format: str, opt, model, processor,
                           image_file: str = None, sleep_time=1, max_retry=3):
    """
    Get final answer with retry logic.
    Uses truncation-enabled text/image response functions.
    """
    retry = 0
    current_messages = messages_text
    
    while retry < max_retry:
        try:
            if opt.modality == 'text':
                response = get_text_response_local(current_messages, model, processor, opt)
            else:  # image or mix
                response = get_image_response_local(current_messages, image_file, model, processor, opt)
            
            # Check for error responses
            if response.startswith("[ERROR]"):
                print(f"  Processing failed: {response}", flush=True)
                return response
            
            # Check for final answer format
            if "[Final Answer]:" in response:
                final_answer = response.split("[Final Answer]:")[-1].strip()
                return final_answer
            elif "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                return final_answer
            else:
                retry += 1
                print(f"No 'Final Answer' found (attempt {retry}/{max_retry}), requesting again...")
                current_messages = messages_text + f'\nNote: Please check your output format. You do not need to do much explaining, just give the final answer in the given format: "[Final Answer]: {answer_format}'
                time.sleep(sleep_time)
                
        except Exception as e:
            retry += 1
            print(f"Generation Error (attempt {retry}/{max_retry}): {e}")
            if retry >= max_retry:
                return f"[ERROR] Failed after {max_retry} retries: {str(e)}"
            time.sleep(sleep_time)
    
    return "[ERROR] Failed to get proper final answer format"
def process_batch_queries(
    batch_queries: List[Dict],
    opt,
    model,
    processor,
    qa_metric,
    file_path: str,
    image_file_path: str,
    table_file_path: str
) -> List[Dict]:
    """
    Process a batch of queries together for improved throughput.
    Only batches queries that can use the same processing path (non-Visualization, non-special eval).
    
    Args:
        batch_queries: List of query dicts
        opt: Options
        model: Loaded model
        processor: Loaded processor
        qa_metric: QA metric calculator
        file_path: Path to table files
        image_file_path: Path to image files
        table_file_path: Path to xlsx tables
        
    Returns:
        List of result dicts
    """
    results = []
    
    # Separate queries by processing type
    batchable_queries = []  # Can be processed in batch
    special_queries = []    # Need individual processing (Visualization, special eval)
    
    for query in batch_queries:
        qtype = query['QuestionType']
        subtype = query.get('SubQType', '')
        
        # Visualization and certain analysis types need special handling
        if qtype == 'Visualization' or subtype in ['Summary Analysis', 'Anomaly Analysis']:
            special_queries.append(query)
        else:
            batchable_queries.append(query)
    
    # Process batchable queries together
    if batchable_queries:
        batch_start_time = time.time()  # Start timing for batch
        
        batch_messages = []
        batch_answer_formats = []
        batch_image_files = []
        batch_references = []
        
        for query in batchable_queries:
            answer_format = get_answer_format(query)
            messages = build_messages(query, answer_format, opt)
            image_file = f'{image_file_path}/{query["FileName"]}.png'
            
            batch_messages.append(messages)
            batch_answer_formats.append(answer_format)
            batch_image_files.append(image_file)
            batch_references.append(query['FinalAnswer'])
        
        # Batch inference
        batch_responses = get_batch_final_answers_local(
            batch_messages,
            batch_answer_formats,
            opt,
            model,
            processor,
            batch_image_files if opt.modality != 'text' else None
        )
        
        # Calculate metrics for batch
        batch_scores = qa_metric.compute(references=batch_references, predictions=batch_responses)
        
        # Calculate time per query (average for batch)
        batch_total_time = time.time() - batch_start_time
        time_per_query = batch_total_time / len(batchable_queries)
        
        # Also compute individual scores for each sample
        for i, (query, response) in enumerate(zip(batchable_queries, batch_responses)):
            individual_scores = qa_metric.compute(references=[batch_references[i]], predictions=[response])
            
            result = {
                'id': query['id'],
                'FileName': query['FileName'],
                'QuestionType': query['QuestionType'],
                'SubQType': query.get('SubQType', ''),
                'Question': query['Question'],
                'Reference': batch_references[i],
                'Prediction': response,
                'Metrics': individual_scores,
                'ProcessingTime': round(time_per_query, 2)  # Average time per query in batch
            }
            results.append(result)
            
            # Print progress
            print(f"  [Batch] Query {query['id']}: {individual_scores} ({time_per_query:.2f}s avg)")
    
    # Process special queries individually
    for query in special_queries:
        query_start_time = time.time()  # Start timing for this query
        
        question_type = query['QuestionType']
        answer_format = get_answer_format(query)
        messages = build_messages(query, answer_format, opt)
        image_file = f'{image_file_path}/{query["FileName"]}.png'
        
        metric_scores = {}
        
        if question_type == 'Visualization':
            response = get_final_answer_local(messages, answer_format, opt, model, processor, image_file)
            reference = query['ProcessedAnswer']
            chart_type = query['SubQType'].split()[0]
            
            # 路径替换：同时处理单引号和双引号的 .xlsx 文件路径
            full_xlsx_path = f"{table_file_path}/{query['FileName']}.xlsx"
            # 1. 替换单引号包裹的 .xlsx 路径
            python_code = re.sub(r"'[^']*\.xlsx'", f"'{full_xlsx_path}'", response)
            # 2. 替换双引号包裹的 .xlsx 路径
            python_code = re.sub(r'"[^"]*\.xlsx"', f'"{full_xlsx_path}"', python_code)
            # 3. 兜底：替换裸字符串 table.xlsx（处理遗漏情况）
            python_code = python_code.replace("table.xlsx", full_xlsx_path)
            
            prediction, ecr_1 = exec_and_get_y_reference(python_code, chart_type)
            metric_scores['ECR'] = ecr_1
            
            if prediction != '':
                try:
                    prediction = ast.literal_eval(prediction)
                    ref_parsed = ast.literal_eval(reference)
                    if chart_type == 'PieChart':
                        metric_scores['Pass'] = compute_pie_chart_metric(ref_parsed, prediction)
                    else:
                        metric_scores['Pass'] = compute_general_chart_metric(ref_parsed, prediction)
                except Exception as e:
                    metric_scores['Pass'] = False
            else:
                metric_scores['Pass'] = False
        else:
            # Data Analysis with special eval
            response = get_final_answer_local(messages, answer_format, opt, model, processor, image_file)
            reference = query['FinalAnswer']
            
            if hasattr(opt, 'eval_api_key') and opt.eval_api_key:
                table_content = read_file(f'{file_path}/{query["FileName"]}.{FILE_EXTENSIONS[opt.format]}')
                eval_prompt = Eval_Prompt[query['SubQType']].format_map({
                    'Question': query['Question'],
                    'Table': table_content,
                    'Reference_Answer': reference,
                    'Predicted_Answer': response
                })
                try:
                    metric_scores['GPT_EVAL'] = get_eval_score(eval_prompt, opt.eval_api_key)
                except:
                    metric_scores['GPT_EVAL'] = 'N/A'
            else:
                metric_scores['GPT_EVAL'] = 'N/A (no eval_api_key)'
        
        # Calculate query processing time
        query_time = time.time() - query_start_time
        
        result = {
            'id': query['id'],
            'FileName': query['FileName'],
            'QuestionType': question_type,
            'SubQType': query.get('SubQType', ''),
            'Question': query['Question'],
            'Reference': reference if question_type != 'Visualization' else query['ProcessedAnswer'],
            'Prediction': response,
            'Metrics': metric_scores,
            'ProcessingTime': round(query_time, 2)  # Time in seconds
        }
        results.append(result)
        print(f"  [Single] Query {query['id']}: {metric_scores} ({query_time:.2f}s)")
    
    return results


def gen_solution_batch(opt):
    """Main function with batch inference support for higher throughput."""
    start_time = datetime.datetime.now()
    
    print(f"{'='*60}")
    print(f"BATCH INFERENCE MODE (batch_size={opt.batch_size})")
    print(f"{'='*60}")
    
    # Load model
    model, processor = load_qwen3_vl_local(
        opt.model_dir, 
        use_flash_attn=opt.use_flash_attn,
        use_model_parallel=opt.use_model_parallel
    )
    
    # Initialize metric
    qa_metric = QAMetric()
    
    # Load dataset
    dataset_path = opt.data_path
    qa_path = opt.qa_path if opt.qa_path else f'{dataset_path}/data'
    # Choose QA file based on options: use_sc_filled takes priority over use_long
    if opt.use_sc_filled:
        qa_file = 'QA_final_sc_filled.json'
    elif opt.use_long:
        qa_file = 'QA_long.json'
    else:
        qa_file = 'QA_final.json'
    with open(f'{qa_path}/{qa_file}', 'r') as fp:
        dataset = json.load(fp)
        querys = dataset['queries']
    print(f"Loaded QA file: {qa_file} ({len(querys)} queries)")
    
    # Filter by question type if specified
    if opt.question_type:
        querys = [q for q in querys if q['QuestionType'] == opt.question_type]
        print(f"Filtered to {len(querys)} queries of type: {opt.question_type}")
    
    # Limit number of queries if specified
    if opt.max_queries > 0:
        querys = querys[:opt.max_queries]
        print(f"Limited to {len(querys)} queries")
    
    # Load IDs to skip from external checkpoint (for multi-process sharding)
    skip_ids = set()
    if opt.skip_checkpoint and os.path.exists(opt.skip_checkpoint):
        with open(opt.skip_checkpoint, 'r') as f:
            skip_data = json.load(f)
            skip_ids = set(skip_data.get('processed_ids', []))
        print(f"Loaded {len(skip_ids)} processed IDs to skip from: {opt.skip_checkpoint}")
    
    # Apply sharding if specified (for multi-process parallelism)
    if opt.shard_id is not None and opt.num_shards is not None:
        # First filter out already processed
        querys = [q for q in querys if q['id'] not in skip_ids]
        
        # Then split into shards
        total_queries = len(querys)
        shard_size = (total_queries + opt.num_shards - 1) // opt.num_shards
        shard_start = opt.shard_id * shard_size
        shard_end = min(shard_start + shard_size, total_queries)
        querys = querys[shard_start:shard_end]
        print(f"Shard {opt.shard_id}/{opt.num_shards}: Processing queries {shard_start+1}-{shard_end} ({len(querys)} queries)")
    
    # Setup output directory (use separate folder for default config experiments)
    model_name = os.path.basename(opt.model_dir)
    output_base = os.path.abspath(f'../result/qwen3vl_local_a100_default')
    if not os.path.exists(output_base):
        os.makedirs(output_base, exist_ok=True)
    
    modality_suffix = f"{opt.modality}"
    if opt.modality != 'image':
        modality_suffix += f"_{opt.format}"
    
    # Add shard suffix to output path for multi-process mode
    output_file_path = f'{output_base}/{model_name}_{modality_suffix}_default'
    if opt.shard_id is not None:
        output_file_path = f'{output_file_path}_shard{opt.shard_id}'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path, exist_ok=True)
    
    # File paths
    file_path = f'{opt.data_path}/{opt.format}'
    image_file_path = f'{opt.data_path}/image'
    table_file_path = f'{opt.data_path}/tables'
    
    all_eval_results = []
    
    # Resume from checkpoint if exists
    checkpoint_file = f'{output_file_path}/checkpoint_batch.json'
    processed_ids = set()
    if os.path.exists(checkpoint_file) and opt.resume:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            all_eval_results = checkpoint_data.get('results', [])
            processed_ids = set(checkpoint_data.get('processed_ids', []))
        print(f"Resuming from checkpoint with {len(processed_ids)} processed queries")
    
    # Filter out already processed queries (from own checkpoint)
    pending_queries = [q for q in querys if q['id'] not in processed_ids]
    print(f"Pending queries after filtering: {len(pending_queries)}")
    
    # Process in batches
    total_batches = (len(pending_queries) + opt.batch_size - 1) // opt.batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        batch_start = batch_idx * opt.batch_size
        batch_end = min(batch_start + opt.batch_size, len(pending_queries))
        batch_queries = pending_queries[batch_start:batch_end]
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}/{total_batches}: Processing {len(batch_queries)} queries")
        print(f"{'='*60}")
        
        try:
            batch_results = process_batch_queries(
                batch_queries, opt, model, processor, qa_metric,
                file_path, image_file_path, table_file_path
            )
            
            all_eval_results.extend(batch_results)
            processed_ids.update([r['id'] for r in batch_results])
            
            # Save checkpoint after each batch
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'results': all_eval_results,
                    'processed_ids': list(processed_ids)
                }, f, indent=2, ensure_ascii=False)
            print(f"Checkpoint saved: {len(processed_ids)} queries processed")
            
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate aggregate metrics
    end_time = datetime.datetime.now()
    
    # Aggregate by question type
    metrics_by_type = {}
    for result in all_eval_results:
        qtype = result['QuestionType']
        if qtype not in metrics_by_type:
            metrics_by_type[qtype] = []
        metrics_by_type[qtype].append(result['Metrics'])
    
    # Calculate averages
    aggregate_metrics = {}
    for qtype, metrics_list in metrics_by_type.items():
        aggregate_metrics[qtype] = {}
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        for key in all_keys:
            # Handle Pass/ECR metrics specially (boolean/string types)
            if key in ['Pass', 'ECR']:
                bool_values = []
                for m in metrics_list:
                    val = m.get(key)
                    if isinstance(val, bool):
                        bool_values.append(1 if val else 0)
                    elif isinstance(val, str):
                        if val.lower() == 'true':
                            bool_values.append(1)
                        elif val.lower() == 'false':
                            bool_values.append(0)
                if bool_values:
                    aggregate_metrics[qtype][key] = sum(bool_values) / len(metrics_list)  # denominator = all samples
            else:
                # Handle numeric metrics (int, float)
                values = [m.get(key) for m in metrics_list if isinstance(m.get(key), (int, float))]
                if values:
                    aggregate_metrics[qtype][key] = sum(values) / len(values)
    
    # Save final results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    final_results = {
        'config': {
            'model_dir': opt.model_dir,
            'modality': opt.modality,
            'format': opt.format,
            'data_path': opt.data_path,
            'batch_size': opt.batch_size,
            'total_queries': len(all_eval_results),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'throughput': len(all_eval_results) / (end_time - start_time).total_seconds() if all_eval_results else 0
        },
        'aggregate_metrics': aggregate_metrics,
        'results': all_eval_results
    }
    
    result_file = f'{output_file_path}/results_batch_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("BATCH EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total queries: {len(all_eval_results)}")
    print(f"Duration: {(end_time - start_time).total_seconds():.2f}s")
    print(f"Throughput: {final_results['config']['throughput']:.2f} queries/sec")
    print(f"Results saved to: {result_file}")
    print(f"\nAggregate Metrics by Question Type:")
    for qtype, metrics in aggregate_metrics.items():
        print(f"  {qtype}:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
    
    # Keep checkpoint file as backup (contains all results and processed IDs)
    print(f"Checkpoint preserved at: {checkpoint_file}")
    
    return final_results


def gen_solution(opt):
    """Main function to run inference and evaluation."""
    start_time = datetime.datetime.now()
    
    # Load model
    model, processor = load_qwen3_vl_local(
        opt.model_dir, 
        use_flash_attn=opt.use_flash_attn,
        use_model_parallel=opt.use_model_parallel
    )
    
    # Initialize metric
    qa_metric = QAMetric()
    
    # Load dataset
    dataset_path = opt.data_path
    qa_path = opt.qa_path if opt.qa_path else f'{dataset_path}/data'
    # Choose QA file based on options: use_sc_filled takes priority over use_long
    if opt.use_sc_filled:
        qa_file = 'QA_final_sc_filled.json'
    elif opt.use_long:
        qa_file = 'QA_long.json'
    else:
        qa_file = 'QA_final.json'
    with open(f'{qa_path}/{qa_file}', 'r') as fp:
        dataset = json.load(fp)
        querys = dataset['queries']
    print(f"Loaded QA file: {qa_file} ({len(querys)} queries)")
    
    # Filter by question type if specified
    if opt.question_type:
        querys = [q for q in querys if q['QuestionType'] == opt.question_type]
        print(f"Filtered to {len(querys)} queries of type: {opt.question_type}")
    
    # Limit number of queries if specified
    if opt.max_queries > 0:
        querys = querys[:opt.max_queries]
        print(f"Limited to {len(querys)} queries")
    
    # Load IDs to skip from external checkpoint (for multi-process sharding)
    skip_ids = set()
    if opt.skip_checkpoint and os.path.exists(opt.skip_checkpoint):
        with open(opt.skip_checkpoint, 'r') as f:
            skip_data = json.load(f)
            skip_ids = set(skip_data.get('processed_ids', []))
        print(f"Loaded {len(skip_ids)} processed IDs to skip from: {opt.skip_checkpoint}")
    
    # Apply sharding if specified (for multi-process parallelism)
    if opt.shard_id is not None and opt.num_shards is not None:
        # First filter out already processed
        querys = [q for q in querys if q['id'] not in skip_ids]
        
        # Then split into shards
        total_queries = len(querys)
        shard_size = (total_queries + opt.num_shards - 1) // opt.num_shards
        shard_start = opt.shard_id * shard_size
        shard_end = min(shard_start + shard_size, total_queries)
        querys = querys[shard_start:shard_end]
        print(f"Shard {opt.shard_id}/{opt.num_shards}: Processing queries {shard_start+1}-{shard_end} ({len(querys)} queries)")
    
    # Setup output directory (use separate folder for default config experiments)
    model_name = os.path.basename(opt.model_dir)
    output_base = os.path.abspath(f'../result/qwen3vl_local_a100_default')
    if not os.path.exists(output_base):
        os.makedirs(output_base, exist_ok=True)
    
    modality_suffix = f"{opt.modality}"
    if opt.modality != 'image':
        modality_suffix += f"_{opt.format}"
    
    # Add shard suffix to output path for multi-process mode
    output_file_path = f'{output_base}/{model_name}_{modality_suffix}_default'
    if opt.shard_id is not None:
        output_file_path = f'{output_file_path}_shard{opt.shard_id}'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path, exist_ok=True)
    
    # File paths
    file_path = f'{opt.data_path}/{opt.format}'
    image_file_path = f'{opt.data_path}/image'
    table_file_path = f'{opt.data_path}/tables'
    
    all_eval_results = []
    
    # Resume from checkpoint if exists
    checkpoint_file = f'{output_file_path}/checkpoint.json'
    processed_ids = set()
    if os.path.exists(checkpoint_file) and opt.resume:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            all_eval_results = checkpoint_data.get('results', [])
            processed_ids = set(checkpoint_data.get('processed_ids', []))
        print(f"Resuming from checkpoint with {len(processed_ids)} processed queries")
    
    for query in tqdm(querys, desc=f"Processing {opt.modality} modality"):
        if query['id'] in processed_ids:
            continue
            
        try:
            query_start_time = time.time()  # Start timing for this query
            
            print(f"\n{'='*60}")
            print(f"Query ID: {query['id']} | Type: {query['QuestionType']} | SubType: {query.get('SubQType', 'N/A')}")
            print(f"{'='*60}")
            
            question_type = query['QuestionType']
            answer_format = get_answer_format(query)
            messages = build_messages(query, answer_format, opt)
            image_file = f'{image_file_path}/{query["FileName"]}.png'
            
            metric_scores = {}
            response = ""
            
            if question_type == 'Visualization':
                # Visualization task: generate code and evaluate
                response = get_final_answer_local(messages, answer_format, opt, model, processor, image_file)
                reference = query['ProcessedAnswer']
                chart_type = query['SubQType'].split()[0]
                
                # 路径替换：同时处理单引号和双引号的 .xlsx 文件路径
                full_xlsx_path = f"{table_file_path}/{query['FileName']}.xlsx"
                # 1. 替换单引号包裹的 .xlsx 路径
                python_code = re.sub(r"'[^']*\.xlsx'", f"'{full_xlsx_path}'", response)
                # 2. 替换双引号包裹的 .xlsx 路径
                python_code = re.sub(r'"[^"]*\.xlsx"', f'"{full_xlsx_path}"', python_code)
                # 3. 兜底：替换裸字符串 table.xlsx（处理遗漏情况）
                python_code = python_code.replace("table.xlsx", full_xlsx_path)
                
                prediction, ecr_1 = exec_and_get_y_reference(python_code, chart_type)
                metric_scores['ECR'] = ecr_1
                
                if prediction != '':
                    try:
                        prediction = ast.literal_eval(prediction)
                        reference = ast.literal_eval(reference)
                        if chart_type == 'PieChart':
                            metric_scores['Pass'] = compute_pie_chart_metric(reference, prediction)
                        else:
                            metric_scores['Pass'] = compute_general_chart_metric(reference, prediction)
                    except Exception as e:
                        metric_scores['Pass'] = 'False'
                else:
                    metric_scores['Pass'] = 'None'
                    
            else:
                reference = query['FinalAnswer']
                
                if question_type == 'Data Analysis':
                    response = get_final_answer_local(messages, answer_format, opt, model, processor, image_file)
                    
                    # GPT evaluation for data analysis tasks
                    if query['SubQType'] in ['Summary Analysis', 'Anomaly Analysis']:
                        table_content = read_file(f'{file_path}/{query["FileName"]}.{FILE_EXTENSIONS[opt.format]}')
                        eval_prompt = Eval_Prompt[query['SubQType']].format_map({
                            'Question': query['Question'],
                            'Table': table_content,
                            'Reference_Answer': query['FinalAnswer'],
                            'User_Answer': response
                        })
                        # Note: GPT eval requires separate API call - skip if not configured
                        if hasattr(opt, 'eval_api_key') and opt.eval_api_key:
                            try:
                                metric_scores['GPT_EVAL'] = get_eval_score(eval_prompt, opt.eval_api_key)
                            except:
                                metric_scores['GPT_EVAL'] = 'N/A'
                        else:
                            metric_scores['GPT_EVAL'] = 'N/A (no eval_api_key)'
                    else:
                        # Standard QA metrics
                        scores = qa_metric.compute(references=[reference], predictions=[response])
                        metric_scores.update(scores)
                else:
                    # Fact Checking, Numerical Reasoning, Structure Comprehending
                    response = get_final_answer_local(messages, answer_format, opt, model, processor, image_file)
                    scores = qa_metric.compute(references=[reference], predictions=[response])
                    metric_scores.update(scores)
            
            # Calculate query processing time
            query_time = time.time() - query_start_time
            
            # Store result
            result = {
                'id': query['id'],
                'FileName': query['FileName'],
                'QuestionType': question_type,
                'SubQType': query.get('SubQType', ''),
                'Question': query['Question'],
                'Reference': reference if question_type != 'Visualization' else query['ProcessedAnswer'],
                'Prediction': response,
                'Metrics': metric_scores,
                'ProcessingTime': round(query_time, 2)  # Time in seconds
            }
            all_eval_results.append(result)
            processed_ids.add(query['id'])
            
            print(f"Prediction: {response[:200]}..." if len(response) > 200 else f"Prediction: {response}")
            print(f"Reference: {reference[:200]}..." if len(str(reference)) > 200 else f"Reference: {reference}")
            print(f"Metrics: {metric_scores}")
            print(f"Processing Time: {query_time:.2f}s")
            
            # Save checkpoint every 10 queries
            if len(processed_ids) % 10 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'results': all_eval_results,
                        'processed_ids': list(processed_ids)
                    }, f, indent=2, ensure_ascii=False)
                print(f"Checkpoint saved: {len(processed_ids)} queries processed")
                
        except Exception as e:
            print(f"Error processing query {query['id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate aggregate metrics
    end_time = datetime.datetime.now()
    
    # Aggregate by question type
    metrics_by_type = {}
    for result in all_eval_results:
        qtype = result['QuestionType']
        if qtype not in metrics_by_type:
            metrics_by_type[qtype] = []
        metrics_by_type[qtype].append(result['Metrics'])
    
    # Calculate averages
    aggregate_metrics = {}
    for qtype, metrics_list in metrics_by_type.items():
        aggregate_metrics[qtype] = {}
        # Get all metric keys
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        for key in all_keys:
            # Handle Pass/ECR metrics specially (boolean/string types)
            if key in ['Pass', 'ECR']:
                bool_values = []
                for m in metrics_list:
                    val = m.get(key)
                    if isinstance(val, bool):
                        bool_values.append(1 if val else 0)
                    elif isinstance(val, str):
                        if val.lower() == 'true':
                            bool_values.append(1)
                        elif val.lower() == 'false':
                            bool_values.append(0)
                if bool_values:
                    aggregate_metrics[qtype][key] = sum(bool_values) / len(metrics_list)  # denominator = all samples
            else:
                # Handle numeric metrics (int, float)
                values = [m.get(key) for m in metrics_list if isinstance(m.get(key), (int, float))]
                if values:
                    aggregate_metrics[qtype][key] = sum(values) / len(values)
    
    # Save final results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    final_results = {
        'config': {
            'model_dir': opt.model_dir,
            'modality': opt.modality,
            'format': opt.format,
            'data_path': opt.data_path,
            'total_queries': len(all_eval_results),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds()
        },
        'aggregate_metrics': aggregate_metrics,
        'results': all_eval_results
    }
    
    result_file = f'{output_file_path}/results_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total queries: {len(all_eval_results)}")
    print(f"Duration: {(end_time - start_time).total_seconds():.2f}s")
    print(f"Results saved to: {result_file}")
    print(f"\nAggregate Metrics by Question Type:")
    for qtype, metrics in aggregate_metrics.items():
        print(f"  {qtype}:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
    
    # Keep checkpoint file as backup (contains all results and processed IDs)
    print(f"Checkpoint preserved at: {checkpoint_file}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Local Qwen3-VL inference on RealHiTBench')
    
    # Model settings
    parser.add_argument('--model_dir', type=str, 
                        default='/mnt/data1/users/4xin/qwen/Qwen3-VL-8B-Instruct',
                        help='Path to local model directory')
    parser.add_argument('--use_flash_attn', action='store_true', default=True,
                        help='Use Flash Attention 2 (default: True)')
    parser.add_argument('--no_flash_attn', action='store_false', dest='use_flash_attn',
                        help='Disable Flash Attention 2')
    parser.add_argument('--use_model_parallel', action='store_true', default=False,
                        help='Use model parallelism (device_map="auto") instead of DataParallel. '
                             'Use this if model does not fit in single GPU memory. '
                             'Default: False (uses DataParallel for true multi-GPU parallel compute)')
    
    # Generation settings (Qwen3-VL Instruct recommended: temp=0.7, top_p=0.8, top_k=20)
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Generation temperature (0.7 recommended for Instruct)')
    parser.add_argument('--top_p', type=float, default=0.8,
                        help='Top-p sampling')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Top-k sampling')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='Repetition penalty (1.0 = no penalty)')
    parser.add_argument('--presence_penalty', type=float, default=1.5,
                        help='Presence penalty (1.5 recommended for Instruct)')
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='Maximum tokens to generate (1024 recommended for RealHiTBench, covers 99%% of answers)')
    
    # Batch inference settings
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (1=sequential, >1=batch mode)')
    
    # Input modality
    parser.add_argument('--modality', type=str, default='image',
                        choices=['image', 'text', 'mix'],
                        help='Input modality: image, text, or mix')
    parser.add_argument('--format', type=str, default='html',
                        choices=['html', 'csv', 'markdown', 'latex'],
                        help='Table format for text/mix modality')
    
    # Dataset settings
    parser.add_argument('--data_path', type=str, 
                        default='/mnt/data1/users/4xin/RealHiTBench',
                        help='Path to RealHiTBench data (images, tables, formats)')
    parser.add_argument('--qa_path', type=str, 
                        default=None,
                        help='Path to QA JSON files directory (default: data_path/data/)')
    parser.add_argument('--use_long', action='store_true',
                        help='Use QA_long.json instead of QA_final.json')
    parser.add_argument('--use_sc_filled', action='store_true',
                        help='Use QA_final_sc_filled.json with updated structure comprehending data')
    parser.add_argument('--question_type', type=str, default=None,
                        choices=['Fact Checking', 'Numerical Reasoning', 
                                 'Data Analysis', 'Visualization', 'Structure Comprehending'],
                        help='Filter by question type')
    parser.add_argument('--max_queries', type=int, default=-1,
                        help='Maximum number of queries to process (-1 for all)')
    parser.add_argument('--task_ids', type=str, default=None,
                        help='Comma-separated list of task IDs to process (e.g., "1,2,3,100"). '
                             'Only these specific tasks will be processed. Useful for rerunning failed tasks.')
    
    # Multi-process sharding for true multi-GPU parallelism
    parser.add_argument('--shard_id', type=int, default=None,
                        help='Shard ID for multi-process parallel (0, 1, 2, ...). Each shard runs on separate GPU.')
    parser.add_argument('--num_shards', type=int, default=None,
                        help='Total number of shards for multi-process parallel')
    parser.add_argument('--skip_checkpoint', type=str, default=None,
                        help='Path to existing checkpoint file to load already processed IDs to skip')
    
    # Evaluation settings
    parser.add_argument('--eval_api_key', type=str, default=None,
                        help='API key for GPT evaluation (optional)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    
    opt = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(opt.model_dir):
        raise ValueError(f"Model directory not found: {opt.model_dir}")
    if not os.path.exists(opt.data_path):
        raise ValueError(f"Data path not found: {opt.data_path}")
    
    # Validate sharding parameters
    if (opt.shard_id is not None) != (opt.num_shards is not None):
        raise ValueError("--shard_id and --num_shards must be used together")
    if opt.shard_id is not None and (opt.shard_id < 0 or opt.shard_id >= opt.num_shards):
        raise ValueError(f"--shard_id must be in range [0, {opt.num_shards-1}]")
    
    print(f"Configuration:")
    print(f"  Model: {opt.model_dir}")
    print(f"  Modality: {opt.modality}")
    print(f"  Format: {opt.format}")
    print(f"  Data path: {opt.data_path}")
    print(f"  Use Flash Attention: {opt.use_flash_attn}")
    print(f"  Multi-GPU Mode: {'Model Parallel (device_map=auto)' if opt.use_model_parallel else 'Data Parallel (DataParallel)'}")
    if opt.shard_id is not None:
        print(f"  Sharding: Shard {opt.shard_id + 1}/{opt.num_shards}")
    if opt.skip_checkpoint:
        print(f"  Skip checkpoint: {opt.skip_checkpoint}")
    print(f"  Generation Settings (Qwen3-VL Instruct Recommended):")
    print(f"    Temperature: {opt.temperature}")
    print(f"    Top-p: {opt.top_p}")
    print(f"    Top-k: {opt.top_k}")
    print(f"    Repetition Penalty: {opt.repetition_penalty}")
    print(f"    Presence Penalty: {opt.presence_penalty}")
    print(f"    Max Tokens: {opt.max_tokens}")
    print(f"  Batch size: {opt.batch_size}")
    print()
    
    # Choose batch or sequential mode
    if opt.batch_size > 1:
        gen_solution_batch(opt)
    else:
        gen_solution(opt)


if __name__ == '__main__':
    main()
