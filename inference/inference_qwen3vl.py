"""
Unified inference script for Qwen3-VL on RealHiTBench dataset.
Supports three input modalities: image-only, text-only, and image+text (mix).

Usage:
    # Image-only
    python inference_qwen3vl.py --modality image --model qwen3-vl-flash --api_key sk-xxx
    
    # Text-only (specify format)
    python inference_qwen3vl.py --modality text --model qwen3-vl-flash --format html --api_key sk-xxx
    
    # Image+Text (mix)
    python inference_qwen3vl.py --modality mix --model qwen3-vl-flash --format latex --api_key sk-xxx
"""

import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import datetime
import json
import ast
import argparse
import re
import time
from tqdm import tqdm
from openai import OpenAI
from matplotlib import pyplot as plt

# Import prompts based on modality (will be selected dynamically)
from answer_prompt_mlm import Answer_Prompt as Answer_Prompt_Image
from answer_prompt_mlm import User_Prompt as User_Prompt_Image
from answer_prompt_mix import Answer_Prompt as Answer_Prompt_Mix
from answer_prompt_mix import User_Prompt as User_Prompt_Mix
from answer_prompt_llm import Answer_Prompt as Answer_Prompt_Text
from answer_prompt_llm import User_Prompt as User_Prompt_Text

from gpt_eval_prompt import Eval_Prompt
from gpt_eval import get_eval_score
from metrics.qa_metrics import QAMetric
from utils.common_util import encode_image, read_file
from timeout_decorator import timeout
from utils.chart_process import build_eval_code, surround_pycode_with_main
from utils.chart_metric_util import compute_pie_chart_metric, compute_general_chart_metric

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# File extension mapping
FILE_EXTENSIONS = {
    "latex": "txt",
    "markdown": "md",
    "csv": "csv",
    "html": "html"
}

# Alibaba Cloud DashScope API endpoints
API_ENDPOINTS = {
    "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "global": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
}


def get_qwen3_vl_client(opt):
    """Create OpenAI-compatible client for Qwen3-VL API."""
    return OpenAI(
        api_key=opt.api_key,
        base_url=opt.base_url or API_ENDPOINTS["singapore"]
    )


def get_text_response(messages_text: str, opt, max_tokens=4096):
    """Get response for text-only input."""
    client = get_qwen3_vl_client(opt)
    messages = [{"role": "user", "content": messages_text}]
    
    chat_completion = client.chat.completions.create(
        model=opt.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
    )
    return chat_completion.choices[0].message.content


def get_image_response(messages_text: str, image_file: str, opt, max_tokens=4096):
    """Get response for image-only or image+text input."""
    client = get_qwen3_vl_client(opt)
    
    # Encode image to base64
    image_base64 = encode_image(image_file)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": messages_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ]
    
    chat_completion = client.chat.completions.create(
        model=opt.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
    )
    return chat_completion.choices[0].message.content


def get_final_answer(messages_text: str, answer_format: str, opt, 
                     image_file: str = None, sleep_time=3, max_retry=5):
    """
    Get final answer with retry logic to ensure proper format.
    
    Args:
        messages_text: The prompt text
        answer_format: Expected answer format
        opt: Options including modality
        image_file: Path to image file (for image/mix modality)
        sleep_time: Sleep time between retries
        max_retry: Maximum number of retries
    """
    retry = 0
    current_messages = messages_text
    
    while retry < max_retry:
        try:
            if opt.modality == 'text':
                response = get_text_response(current_messages, opt)
            else:  # image or mix
                response = get_image_response(current_messages, image_file, opt)
            
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
            print(f"API Error (attempt {retry}/{max_retry}): {e}")
            time.sleep(sleep_time * 2)
    
    return response if 'response' in dir() else "Error: Failed to get response"


def build_messages(query, answer_format, opt):
    """
    Build prompt messages based on modality.
    
    Args:
        query: Query dict containing question info
        answer_format: Expected answer format
        opt: Options including modality and format
    """
    question_type = query['QuestionType']
    
    # Select task prompt based on question type
    if opt.modality == 'text':
        Answer_Prompt = Answer_Prompt_Text
        User_Prompt = User_Prompt_Text
    elif opt.modality == 'image':
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
    
    # For text-only mode, need to format the prompt with table format
    if opt.modality == 'text':
        TASK_PROMPT = TASK_PROMPT.format(format=opt.format)
    
    # Build user prompt
    if opt.modality == 'image':
        # Image-only: no table text in prompt
        QUESTION_PROMPT = User_Prompt.format_map({
            'Table': "[See the table image]",
            'Question': query['Question'],
            'Answer_format': answer_format
        })
    else:
        # Text or Mix: include table content
        file_path = f'{opt.data_path}/{opt.format}'
        table_content = read_file(f'{file_path}/{query["FileName"]}.{FILE_EXTENSIONS[opt.format]}')
        QUESTION_PROMPT = User_Prompt.format_map({
            'Table': table_content,
            'Question': query['Question'],
            'Answer_format': answer_format
        })
    
    messages = TASK_PROMPT + "\n" + QUESTION_PROMPT
    return messages


def get_answer_format(query):
    """Determine expected answer format based on question type."""
    if query['SubQType'] == 'Exploratory Analysis':
        return "CorrelationRelation, CorrelationCoefficient"
    elif query['QuestionType'] == 'Visualization':
        return "import pandas as pd \n import matplotlib.pyplot as plt \n ... plt.show()"
    else:
        return "AnswerName1, AnswerName2..."


@timeout(15)
def execute(c):
    exec(c)


@timeout(20)
def exec_and_get_y_reference(answer_code, chart_type):
    """Execute visualization code and extract chart data for evaluation."""
    ecr_1 = False
    python_code, eval_code = build_eval_code(answer_code, chart_type)
    print("Code:", python_code)
    
    if python_code == "":
        return "", False
    
    try:
        python_code = surround_pycode_with_main(python_code)
        execute(python_code)
        ecr_1 = True
        plt.close('all')
    except Exception as e:
        print("Python Error:", e)
        ecr_1 = False
        return "", False
    
    if ecr_1:
        pass
    
    try:
        from io import StringIO
        output = StringIO()
        stdout = sys.stdout
        try:
            sys.stdout = output
            chart_eval_code = surround_pycode_with_main(eval_code)
            execute(chart_eval_code)
        except Exception as e:
            print("Eval Error:", e)
            return "", True
        finally:
            sys.stdout = stdout
        output_value = output.getvalue()
        print("OUTPUT VALUE:", output_value)
    except Exception as e:
        print("Eval Error:", e)
        output_value = ''
    
    if output_value != '':
        parsed_prediction = output_value.strip()
    else:
        parsed_prediction = ''
    
    plt.close('all')
    return parsed_prediction, ecr_1


def gen_solution(opt):
    """Main function to run inference and evaluation."""
    start_time = datetime.datetime.now()
    
    # Initialize metric
    qa_metric = QAMetric()
    
    # Load dataset
    dataset_path = opt.data_path
    qa_file = 'QA_final.json' if not opt.use_long else 'QA_long.json'
    with open(f'{dataset_path}/{qa_file}', 'r') as fp:
        dataset = json.load(fp)
        querys = dataset['queries']
    
    # Filter by question type if specified
    if opt.question_type:
        querys = [q for q in querys if q['QuestionType'] == opt.question_type]
        print(f"Filtered to {len(querys)} queries of type: {opt.question_type}")
    
    # Limit number of queries if specified
    if opt.max_queries > 0:
        querys = querys[:opt.max_queries]
        print(f"Limited to {len(querys)} queries")
    
    # Setup output directory
    output_base = os.path.abspath(f'../result/qwen3vl')
    if not os.path.exists(output_base):
        os.makedirs(output_base, exist_ok=True)
    
    modality_suffix = f"{opt.modality}"
    if opt.modality != 'image':
        modality_suffix += f"_{opt.format}"
    
    output_file_path = f'{output_base}/{opt.model}_{modality_suffix}'
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
                response = get_final_answer(messages, answer_format, opt, image_file)
                reference = query['ProcessedAnswer']
                chart_type = query['SubQType'].split()[0]
                
                # Replace table path in generated code
                python_code = re.sub(r"'[^']*\.xlsx'", f"'{table_file_path}/{query['FileName']}.xlsx'", response)
                python_code = python_code.replace("table.xlsx", f"{table_file_path}/{query['FileName']}.xlsx")
                
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
                    response = get_final_answer(messages, answer_format, opt, image_file)
                    
                    # GPT evaluation for data analysis tasks
                    if query['SubQType'] in ['Summary Analysis', 'Anomaly Analysis']:
                        table_content = read_file(f'{file_path}/{query["FileName"]}.{FILE_EXTENSIONS[opt.format]}')
                        eval_prompt = Eval_Prompt[query['SubQType']].format_map({
                            'Question': query['Question'],
                            'Table': table_content,
                            'Reference_Answer': query['FinalAnswer'],
                            'User_Answer': response
                        })
                    else:
                        eval_prompt = Eval_Prompt[query['SubQType']].format_map({
                            'Question': query['Question'],
                            'Reference_Answer': query['FinalAnswer'],
                            'User_Answer': response
                        })
                    
                    # Note: GPT eval requires separate API key for OpenAI
                    try:
                        eval_score = get_eval_score([eval_prompt], opt)
                        metric_scores['GPT_EVAL'] = eval_score
                    except Exception as e:
                        print(f"GPT Eval failed: {e}")
                        metric_scores['GPT_EVAL'] = 'N/A'
                    
                    prediction = response
                    metric_scores.update(qa_metric.compute([reference], [prediction]))
                    
                elif question_type == 'Structure Comprehending':
                    # Structure comprehension: compare original and swapped table
                    response = get_final_answer(messages, answer_format, opt, image_file)
                    reference = response  # First response becomes reference
                    
                    # Use swapped table
                    query_swap = query.copy()
                    query_swap["FileName"] = query["FileName"] + "_swap"
                    image_file_swap = f'{image_file_path}/{query_swap["FileName"]}.png'
                    messages_swap = build_messages(query_swap, answer_format, opt)
                    
                    response_swap = get_final_answer(messages_swap, answer_format, opt, image_file_swap)
                    prediction = response_swap
                    
                    metric_scores = qa_metric.compute([reference], [prediction])
                    
                else:
                    # Standard QA tasks (Fact Checking, Numerical Reasoning)
                    response = get_final_answer(messages, answer_format, opt, image_file)
                    prediction = response
                    metric_scores = qa_metric.compute([reference], [prediction])
            
            # Build evaluation result
            eval_result = {
                'Id': query['id'],
                'FileName': query['FileName'],
                'QuestionType': query['QuestionType'],
                'SubQType': query.get('SubQType', ''),
                'Question': query['Question'],
                'Model_Answer': response,
                'Correct_Answer': query['FinalAnswer'],
                'F1': metric_scores.get('F1', ''),
                'EM': metric_scores.get('EM', ''),
                'ROUGE-L': metric_scores.get('ROUGE-L', ''),
                'SacreBLEU': metric_scores.get('SacreBLEU', ''),
                'GPT_EVAL': metric_scores.get('GPT_EVAL', ''),
                'ECR': metric_scores.get('ECR', ''),
                'Pass': metric_scores.get('Pass', ''),
                'Modality': opt.modality,
                'Format': opt.format if opt.modality != 'image' else 'N/A'
            }
            
            print(f"Result: F1={eval_result['F1']}, EM={eval_result['EM']}")
            all_eval_results.append(eval_result)
            processed_ids.add(query['id'])
            
            # Save checkpoint periodically
            if len(all_eval_results) % opt.checkpoint_interval == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'results': all_eval_results,
                        'processed_ids': list(processed_ids)
                    }, f, indent=2, ensure_ascii=False)
                print(f"Checkpoint saved: {len(all_eval_results)} results")
            
            # Rate limiting
            time.sleep(opt.sleep_time)
            
        except Exception as e:
            print(f"Error processing query {query['id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate and print summary statistics
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Compute aggregated metrics
    metrics_summary = compute_summary_metrics(all_eval_results)
    
    # Save final results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f'{output_file_path}/results_{timestamp}.json'
    
    final_output = {
        'config': {
            'model': opt.model,
            'modality': opt.modality,
            'format': opt.format,
            'data_path': opt.data_path,
            'total_queries': len(querys),
            'processed_queries': len(all_eval_results),
            'duration_seconds': duration
        },
        'summary': metrics_summary,
        'results': all_eval_results
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {opt.model}")
    print(f"Modality: {opt.modality}")
    print(f"Format: {opt.format}")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Results saved to: {result_file}")
    print(f"\nSummary Metrics:")
    for key, value in metrics_summary.items():
        print(f"  {key}: {value}")
    
    return all_eval_results


def compute_summary_metrics(results):
    """Compute aggregated metrics across all results."""
    if not results:
        return {}
    
    summary = {}
    
    # Numeric metrics
    for metric in ['F1', 'EM', 'ROUGE-L', 'SacreBLEU']:
        values = [r[metric] for r in results if r[metric] != '' and r[metric] is not None]
        if values:
            try:
                numeric_values = [float(v) for v in values]
                summary[f'avg_{metric}'] = round(sum(numeric_values) / len(numeric_values), 4)
            except:
                pass
    
    # ECR and Pass rates for visualization
    ecr_values = [r['ECR'] for r in results if r['ECR'] != '']
    if ecr_values:
        summary['ECR_rate'] = round(sum(1 for v in ecr_values if v == True) / len(ecr_values), 4)
    
    pass_values = [r['Pass'] for r in results if r['Pass'] != '' and r['Pass'] != 'None']
    if pass_values:
        summary['Pass_rate'] = round(sum(1 for v in pass_values if v == 'True' or v == True) / len(pass_values), 4)
    
    # Breakdown by question type
    by_type = {}
    for r in results:
        qtype = r['QuestionType']
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(r)
    
    for qtype, type_results in by_type.items():
        f1_values = [float(r['F1']) for r in type_results if r['F1'] != '' and r['F1'] is not None]
        if f1_values:
            summary[f'{qtype}_avg_F1'] = round(sum(f1_values) / len(f1_values), 4)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL inference on RealHiTBench')
    
    # Model settings
    parser.add_argument('--model', type=str, default='qwen3-vl-flash',
                        help='Model name (qwen3-vl-flash, qwen3-vl-plus)')
    parser.add_argument('--api_key', type=str, required=True,
                        help='Alibaba Cloud DashScope API key')
    parser.add_argument('--base_url', type=str, default=None,
                        help='API base URL (default: Singapore endpoint)')
    
    # Input modality
    parser.add_argument('--modality', type=str, required=True,
                        choices=['image', 'text', 'mix'],
                        help='Input modality: image (image-only), text (text-only), mix (image+text)')
    parser.add_argument('--format', type=str, default='html',
                        choices=['latex', 'markdown', 'csv', 'html'],
                        help='Table text format (for text and mix modality)')
    
    # Data settings
    parser.add_argument('--data_path', type=str, 
                        default='/mnt/data1/users/4xin/RealHiTBench',
                        help='Path to RealHiTBench dataset')
    parser.add_argument('--use_long', action='store_true',
                        help='Use QA_long.json instead of QA_final.json')
    parser.add_argument('--question_type', type=str, default=None,
                        choices=['Fact Checking', 'Numerical Reasoning', 'Data Analysis', 
                                'Visualization', 'Structure Comprehending'],
                        help='Filter by question type')
    parser.add_argument('--max_queries', type=int, default=-1,
                        help='Maximum number of queries to process (-1 for all)')
    
    # Execution settings
    parser.add_argument('--sleep_time', type=float, default=0.5,
                        help='Sleep time between API calls (seconds)')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Save checkpoint every N queries')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if exists')
    
    opt = parser.parse_args()
    
    # Validate settings
    if opt.modality == 'image' and opt.format:
        print("Note: --format is ignored for image-only modality")
    
    print(f"\n{'='*60}")
    print("Configuration:")
    print(f"{'='*60}")
    for key, value in vars(opt).items():
        if key != 'api_key':
            print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # Run inference
    gen_solution(opt)


if __name__ == '__main__':
    main()
