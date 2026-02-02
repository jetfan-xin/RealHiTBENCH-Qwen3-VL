#!/usr/bin/env python
"""
测试脚本：检查三种模态 (text_latex, image, mix_latex) 的输入是否正确
只处理第一个 query，详细输出所有输入数据和模型输出
"""

import sys
import os
import json
import argparse
from datetime import datetime

# CRITICAL: Set cache directories to writable location BEFORE any imports
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_DATASETS_CACHE'] = os.path.expanduser('~/.cache/huggingface/datasets')
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')

# CRITICAL: Disable PIL decompression bomb check BEFORE any other imports
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Import prompts
from answer_prompt_mlm import Answer_Prompt as Answer_Prompt_Image
from answer_prompt_mlm import User_Prompt as User_Prompt_Image
from answer_prompt_mix import Answer_Prompt as Answer_Prompt_Mix
from answer_prompt_mix import User_Prompt as User_Prompt_Mix
from answer_prompt_llm import Answer_Prompt as Answer_Prompt_Text
from answer_prompt_llm import User_Prompt as User_Prompt_Text

from utils.common_util import read_file

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# File extension mapping
FILE_EXTENSIONS = {
    "latex": "txt",
    "markdown": "md",
    "csv": "csv",
    "html": "html"
}


def load_model_and_processor(model_dir):
    """Load model and processor"""
    print("=" * 80)
    print("Loading model and processor...")
    print("=" * 80)
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    
    print(f"Model loaded from: {model_dir}")
    print(f"Processor loaded with default settings")
    return model, processor


def build_messages_for_test(query, answer_format, modality, format_type, data_path):
    """构建消息并返回详细信息"""
    question_type = query['QuestionType']
    
    # Select prompt templates based on modality
    if modality == 'text':
        Answer_Prompt = Answer_Prompt_Text
        User_Prompt = User_Prompt_Text
    elif modality == 'image':
        Answer_Prompt = Answer_Prompt_Image
        User_Prompt = User_Prompt_Image
    else:  # mix
        Answer_Prompt = Answer_Prompt_Mix
        User_Prompt = User_Prompt_Mix
    
    # Get task-specific prompt
    if question_type == 'Data Analysis':
        TASK_PROMPT = Answer_Prompt[query['SubQType']]
    else:
        TASK_PROMPT = Answer_Prompt[question_type]
    
    # For text-only mode, format the prompt with table format
    if modality == 'text':
        TASK_PROMPT = TASK_PROMPT.format(format=format_type)
    
    # Build user prompt
    table_content = None
    if modality == 'image':
        # Image-only: no table text in prompt
        table_placeholder = "[See the table image]"
        QUESTION_PROMPT = User_Prompt.format_map({
            'Table': table_placeholder,
            'Question': query['Question'],
            'Answer_format': answer_format
        })
    else:
        # Text or Mix: include table content
        file_path = f'{data_path}/{format_type}'
        table_file = f'{file_path}/{query["FileName"]}.{FILE_EXTENSIONS[format_type]}'
        table_content = read_file(table_file)
        table_placeholder = table_content
        QUESTION_PROMPT = User_Prompt.format_map({
            'Table': table_content,
            'Question': query['Question'],
            'Answer_format': answer_format
        })
    
    messages_text = TASK_PROMPT + "\n" + QUESTION_PROMPT
    
    return {
        'task_prompt': TASK_PROMPT,
        'user_prompt': QUESTION_PROMPT,
        'full_messages_text': messages_text,
        'table_content': table_content,
        'table_placeholder': table_placeholder[:500] + "..." if table_placeholder and len(table_placeholder) > 500 else table_placeholder
    }


def test_single_modality(model, processor, query, modality, format_type, data_path, log_file):
    """测试单个模态"""
    
    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write(f"MODALITY: {modality}" + (f"_{format_type}" if format_type else "") + "\n")
    log_file.write("=" * 80 + "\n\n")
    
    # Build answer format
    if query['SubQType'] == 'Exploratory Analysis':
        answer_format = "CorrelationRelation, CorrelationCoefficient"
    elif query['QuestionType'] == 'Visualization':
        answer_format = "import pandas as pd \\n import matplotlib.pyplot as plt \\n ... plt.show()"
    else:
        answer_format = "AnswerName1, AnswerName2..."
    
    # Build messages
    msg_info = build_messages_for_test(query, answer_format, modality, format_type, data_path)
    
    log_file.write("-" * 40 + "\n")
    log_file.write("TASK PROMPT:\n")
    log_file.write("-" * 40 + "\n")
    log_file.write(msg_info['task_prompt'] + "\n\n")
    
    log_file.write("-" * 40 + "\n")
    log_file.write("USER PROMPT:\n")
    log_file.write("-" * 40 + "\n")
    log_file.write(msg_info['user_prompt'][:2000] + ("..." if len(msg_info['user_prompt']) > 2000 else "") + "\n\n")
    
    if msg_info['table_content']:
        log_file.write("-" * 40 + "\n")
        log_file.write("TABLE CONTENT (first 2000 chars):\n")
        log_file.write("-" * 40 + "\n")
        log_file.write(msg_info['table_content'][:2000] + ("..." if len(msg_info['table_content']) > 2000 else "") + "\n\n")
    
    # Build input for model
    image_file = f'{data_path}/image/{query["FileName"]}.png'
    
    if modality == 'text':
        # Text-only: no image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": msg_info['full_messages_text']}
                ]
            }
        ]
        log_file.write("-" * 40 + "\n")
        log_file.write("MODEL INPUT STRUCTURE (text-only):\n")
        log_file.write("-" * 40 + "\n")
        log_file.write("[\n")
        log_file.write("  {\n")
        log_file.write('    "role": "user",\n')
        log_file.write('    "content": [\n')
        log_file.write('      {"type": "text", "text": "<FULL_MESSAGES_TEXT>"}\n')
        log_file.write("    ]\n")
        log_file.write("  }\n")
        log_file.write("]\n\n")
        log_file.write("NOTE: NO IMAGE INPUT for text modality!\n\n")
    else:
        # Image or Mix: include image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_file},
                    {"type": "text", "text": msg_info['full_messages_text']}
                ]
            }
        ]
        log_file.write("-" * 40 + "\n")
        log_file.write(f"MODEL INPUT STRUCTURE ({modality}):\n")
        log_file.write("-" * 40 + "\n")
        log_file.write("[\n")
        log_file.write("  {\n")
        log_file.write('    "role": "user",\n')
        log_file.write('    "content": [\n')
        log_file.write(f'      {{"type": "image", "image": "{image_file}"}},\n')
        log_file.write('      {"type": "text", "text": "<FULL_MESSAGES_TEXT>"}\n')
        log_file.write("    ]\n")
        log_file.write("  }\n")
        log_file.write("]\n\n")
        log_file.write(f"IMAGE FILE: {image_file}\n")
        log_file.write(f"IMAGE EXISTS: {os.path.exists(image_file)}\n")
        if os.path.exists(image_file):
            with Image.open(image_file) as img:
                log_file.write(f"IMAGE SIZE: {img.size}\n")
        log_file.write("\n")
    
    # Process and tokenize
    log_file.write("-" * 40 + "\n")
    log_file.write("TOKENIZATION:\n")
    log_file.write("-" * 40 + "\n")
    
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        log_file.write(f"Input IDs shape: {inputs['input_ids'].shape}\n")
        log_file.write(f"Total tokens: {inputs['input_ids'].shape[1]}\n")
        if 'pixel_values' in inputs:
            log_file.write(f"Pixel values shape: {inputs['pixel_values'].shape}\n")
        else:
            log_file.write("Pixel values: None (text-only)\n")
        if 'image_grid_thw' in inputs:
            log_file.write(f"Image grid THW: {inputs['image_grid_thw']}\n")
        log_file.write("\n")
        
        # Generate response
        log_file.write("-" * 40 + "\n")
        log_file.write("GENERATING RESPONSE...\n")
        log_file.write("-" * 40 + "\n")
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.8,
                top_k=20,
            )
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        log_file.write("\n")
        log_file.write("-" * 40 + "\n")
        log_file.write("MODEL OUTPUT:\n")
        log_file.write("-" * 40 + "\n")
        log_file.write(output_text + "\n\n")
        
        # Extract final answer if present
        if "[Final Answer]:" in output_text:
            final_answer = output_text.split("[Final Answer]:")[-1].strip()
            log_file.write(f"EXTRACTED FINAL ANSWER: {final_answer}\n")
        elif "Final Answer:" in output_text:
            final_answer = output_text.split("Final Answer:")[-1].strip()
            log_file.write(f"EXTRACTED FINAL ANSWER: {final_answer}\n")
        else:
            log_file.write("WARNING: No 'Final Answer' found in output!\n")
        
        log_file.write("\n")
        log_file.write(f"REFERENCE ANSWER: {query.get('FinalAnswer', 'N/A')}\n")
        
        return output_text
        
    except Exception as e:
        log_file.write(f"ERROR: {str(e)}\n")
        import traceback
        log_file.write(traceback.format_exc() + "\n")
        return None


def main():
    parser = argparse.ArgumentParser(description='Test modality inputs')
    parser.add_argument('--model_dir', type=str, 
                        default='/mnt/data2/projects/pan/4xin/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--data_path', type=str, 
                        default='/mnt/data2/projects/pan/4xin/datasets/RealHiTBench')
    parser.add_argument('--qa_path', type=str, 
                        default='/ltstorage/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data')
    parser.add_argument('--query_id', type=int, default=None,
                        help='Specific query ID to test (default: first query)')
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Create log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.dirname(os.path.abspath(__file__)) + '/../result/modality_test_logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = f'{log_dir}/modality_test_{timestamp}.log'
    
    print(f"Log file: {log_path}")
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write("MODALITY INPUT TEST\n")
        log_file.write(f"Time: {datetime.now().isoformat()}\n")
        log_file.write("=" * 80 + "\n\n")
        
        # Load QA data
        qa_file = f'{args.qa_path}/QA_final_sc_filled.json'
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
        queries = qa_data['queries']
        
        # Get query to test
        if args.query_id is not None:
            query = next((q for q in queries if q['id'] == args.query_id), None)
            if query is None:
                print(f"Query ID {args.query_id} not found!")
                return
        else:
            query = queries[0]
        
        log_file.write("=" * 80 + "\n")
        log_file.write("QUERY INFORMATION\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"ID: {query['id']}\n")
        log_file.write(f"FileName: {query['FileName']}\n")
        log_file.write(f"QuestionType: {query['QuestionType']}\n")
        log_file.write(f"SubQType: {query.get('SubQType', 'N/A')}\n")
        log_file.write(f"Question: {query['Question']}\n")
        log_file.write(f"FinalAnswer: {query.get('FinalAnswer', 'N/A')}\n\n")
        
        # Load model
        model, processor = load_model_and_processor(args.model_dir)
        
        # Test three modalities
        modalities_to_test = [
            ('text', 'latex'),
            ('image', None),
            ('mix', 'latex'),
        ]
        
        for modality, format_type in modalities_to_test:
            print(f"\nTesting modality: {modality}" + (f"_{format_type}" if format_type else ""))
            test_single_modality(
                model, processor, query, modality, format_type, args.data_path, log_file
            )
            log_file.flush()
        
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("TEST COMPLETED\n")
        log_file.write("=" * 80 + "\n")
    
    print(f"\nTest completed! Log saved to: {log_path}")


if __name__ == '__main__':
    main()
