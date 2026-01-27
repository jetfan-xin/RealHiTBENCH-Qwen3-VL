#!/usr/bin/env python3
"""
增强版：重新计算已有 results_*.json 中可视化任务的 ECR 与 Pass 指标。
添加详细的执行过程调试信息，包括 build_eval_code 生成的代码和执行错误。

cd /ltstorage/home/4xin/image_table/RealHiTBench && python utils/recompute_visualization_metrics_enhanced.py \
  result/qwen3vl_local/Qwen3-VL-8B-Instruct_text_html/results_20260127_074206.json \
  --viz-only \
  --save-debug \
  --output result/qwen3vl_local/Qwen3-VL-8B-Instruct_text_html/results_viz_enhanced.json

"""

import argparse
import ast
import json
import os
import pathlib
import re
import sys
from typing import Any, Dict, List
import io
import matplotlib.pyplot as plt

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
ORIG_CWD = pathlib.Path.cwd()

sys.path.insert(0, str(BASE_DIR / "inference"))
sys.path.insert(0, str(BASE_DIR))
from utils.chart_process import build_eval_code, surround_pycode_with_main
from utils.chart_metric_util import compute_general_chart_metric, compute_pie_chart_metric
from utils.recompute_aggregate_metrics import recompute_aggregate_metrics
from timeout_decorator import timeout

os.chdir(ORIG_CWD)


@timeout(15)
def execute(c):
    exec(c)


def replace_table_path(code: str, file_name: str, table_dir: pathlib.Path) -> str:
    """将 Prediction 中的 table.xlsx 替换为真实路径，兼容单双引号。"""
    full_path = table_dir / f"{file_name}.xlsx"
    replaced = re.sub(r"'[^']*\.xlsx'", f"'{full_path}'", code)
    replaced = re.sub(r'"[^"]*\.xlsx"', f'"{full_path}"', replaced)
    replaced = replaced.replace("table.xlsx", str(full_path))
    return replaced


def exec_and_get_y_reference_with_debug(answer_code, chart_type):
    """
    增强版执行函数，返回详细的调试信息。
    
    Returns:
        tuple: (pred_str, ecr_flag, debug_dict)
    """
    debug = {
        "stage1_success": False,
        "stage1_error": None,
        "stage2_success": False,
        "stage2_error": None,
        "python_code_extracted": None,
        "eval_code_generated": None,
        "output_value": None,
    }
    
    ecr_1 = False
    python_code, eval_code = build_eval_code(answer_code, chart_type)
    debug["python_code_extracted"] = python_code
    debug["eval_code_generated"] = eval_code
    
    print("Code:", python_code)
    
    if python_code == "":
        debug["stage1_error"] = "Code extraction failed"
        return "", False, debug
    
    # Stage 1: Execute python code
    try:
        python_code = surround_pycode_with_main(python_code)
        execute(python_code)
        ecr_1 = True
        debug["stage1_success"] = True
        plt.close('all')
    except Exception as e:
        print("Python Error:", e)
        ecr_1 = False
        debug["stage1_error"] = f"{type(e).__name__}: {str(e)}"
        return "", False, debug
    
    # Stage 2: Execute eval code to extract Y values
    try:
        from io import StringIO
        output = StringIO()
        stdout = sys.stdout
        try:
            sys.stdout = output
            chart_eval_code = surround_pycode_with_main(eval_code)
            execute(chart_eval_code)
            debug["stage2_success"] = True
        except Exception as e:
            print("Eval Error:", e)
            debug["stage2_error"] = f"{type(e).__name__}: {str(e)}"
            return "", True, debug
        finally:
            sys.stdout = stdout
        output_value = output.getvalue()
        debug["output_value"] = output_value
        print("OUTPUT VALUE:", output_value)
    except Exception as e:
        print("Eval Error:", e)
        debug["stage2_error"] = f"Outer exception: {type(e).__name__}: {str(e)}"
        output_value = ''
    
    if output_value != '':
        parsed_prediction = output_value.strip()
    else:
        parsed_prediction = ''
    
    plt.close('all')
    return parsed_prediction, ecr_1, debug


def recompute_for_file(json_path: pathlib.Path, table_dir: pathlib.Path, save_debug: bool = False) -> Dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    results: List[Dict[str, Any]] = data.get("results", [])
    for item in results:
        if item.get("QuestionType") != "Visualization":
            continue

        chart_type = item.get("SubQType", "").split()[0]
        file_name = item.get("FileName", "")
        prediction_code = item.get("Prediction", "")

        # 初始化调试信息
        debug_info = {
            "original_prediction": prediction_code,
            "chart_type": chart_type,
            "file_name": file_name,
            "table_path": str(table_dir / f"{file_name}.xlsx"),
        }

        # 路径替换
        python_code = replace_table_path(prediction_code, file_name, table_dir)
        debug_info["code_after_path_replacement"] = python_code

        # 执行代码，提取预测值与 ECR（使用增强版函数）
        pred_str, ecr_flag, exec_debug = exec_and_get_y_reference_with_debug(python_code, chart_type)
        debug_info["ecr"] = ecr_flag
        debug_info["extracted_y_string"] = pred_str
        debug_info["execution_details"] = exec_debug

        # 重新计算 Pass
        pass_flag = False
        ref_str = item.get("Reference", "").strip()
        if ref_str.endswith("\n"):
            ref_str = ref_str[:-1]
        debug_info["reference_string"] = ref_str
        
        if pred_str:
            try:
                pred_val = ast.literal_eval(pred_str)
                debug_info["prediction_parsed"] = pred_val
                
                ref_val = ast.literal_eval(ref_str)
                debug_info["reference_parsed"] = ref_val
                
                # 验证pred_val是有效的数值列表
                def is_valid_numeric_data(data):
                    """检查数据是否为有效的数值数据（嵌套列表或单个列表）"""
                    if isinstance(data, (int, float)):
                        return True
                    if isinstance(data, list):
                        if len(data) == 0:
                            return False
                        if isinstance(data[0], (int, float)):
                            return True
                        if isinstance(data[0], list):
                            return all(
                                isinstance(item, (int, float)) 
                                for sublist in data 
                                for item in (sublist if isinstance(sublist, list) else [sublist])
                            )
                    return False
                
                is_valid = is_valid_numeric_data(pred_val)
                debug_info["prediction_is_valid"] = is_valid
                
                if is_valid:
                    if chart_type == "PieChart":
                        pass_flag = compute_pie_chart_metric(ref_val, pred_val)
                        debug_info["metric_function"] = "compute_pie_chart_metric"
                    else:
                        pass_flag = compute_general_chart_metric(ref_val, pred_val)
                        debug_info["metric_function"] = "compute_general_chart_metric"
                else:
                    debug_info["pass_fail_reason"] = "prediction data is not valid numeric"
            except Exception as e:
                pass_flag = False
                debug_info["pass_computation_error"] = str(e)
        else:
            debug_info["pass_fail_reason"] = "extracted_y_string is empty"

        debug_info["pass"] = pass_flag

        metrics = item.get("Metrics", {})
        metrics["ECR"] = bool(ecr_flag)
        metrics["Pass"] = pass_flag
        item["Metrics"] = metrics
        
        # 保存调试信息
        if save_debug:
            item["debug_info"] = debug_info

    data["results"] = results
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Enhanced: Recompute Visualization ECR and Pass with detailed debug info")
    parser.add_argument("json_path", type=pathlib.Path, help="Path to results_*.json")
    parser.add_argument(
        "--table-dir",
        type=pathlib.Path,
        help="Directory containing .xlsx tables (default: <config.data_path>/tables)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Output path (default: <json_path> with _viz_enhanced suffix)",
    )
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file")
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="Output only Visualization tasks with recomputed metrics and aggregated stats",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save detailed debug information including build_eval_code outputs and execution errors",
    )
    args = parser.parse_args()

    if not args.json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {args.json_path}")

    raw = json.loads(args.json_path.read_text(encoding="utf-8"))
    config = raw.get("config", {})
    default_table = pathlib.Path(config.get("data_path", ".")) / "tables"
    table_dir = args.table_dir or default_table

    data = recompute_for_file(args.json_path, table_dir, save_debug=args.save_debug)

    # 可选：仅输出 Visualization 任务
    if args.viz_only:
        viz_results = [r for r in data.get("results", []) if r.get("QuestionType") == "Visualization"]
        if not viz_results:
            raise ValueError("No Visualization tasks found in results")
        data = {
            "config": data.get("config", {}),
            "results": viz_results,
            "aggregate_metrics": {
                "Visualization": recompute_aggregate_metrics(viz_results).get("Visualization", {})
            },
        }

    if args.inplace:
        out_path = args.json_path
    else:
        if args.viz_only:
            suffix = "_viz_enhanced.json"
        else:
            suffix = "_viz_recomputed_enhanced.json"
        out_path = args.output or args.json_path.with_name(args.json_path.stem + suffix)

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recomputed visualization metrics with enhanced debug info written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
