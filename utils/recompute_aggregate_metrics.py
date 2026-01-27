#!/usr/bin/env python3
"""
重新计算已有results JSON文件的aggregate_metrics
修复Pass/ECR布尔值统计问题
"""

import json
import pathlib
import sys
from typing import Dict, List, Any


def recompute_aggregate_metrics(results_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    重新计算aggregate_metrics，正确处理Pass/ECR布尔值
    
    Args:
        results_list: 包含Metrics字段的结果列表
    
    Returns:
        按QuestionType分组的聚合指标
    """
    # 按QuestionType分组
    metrics_by_type = {}
    for r in results_list:
        qtype = r["QuestionType"]
        if qtype not in metrics_by_type:
            metrics_by_type[qtype] = []
        metrics_by_type[qtype].append(r["Metrics"])
    
    # 计算聚合指标
    aggregate_metrics = {}
    for qtype, lst in metrics_by_type.items():
        aggregate_metrics[qtype] = {}
        
        # 获取所有指标键
        all_keys = set()
        for m in lst:
            all_keys.update(m.keys())
        
        for k in all_keys:
            # 特殊处理Pass/ECR布尔值
            if k in ["Pass", "ECR"]:
                bool_values = []
                for m in lst:
                    v = m.get(k)
                    if isinstance(v, bool):
                        bool_values.append(1 if v else 0)
                    elif isinstance(v, str):
                        if v.lower() == "true":
                            bool_values.append(1)
                        elif v.lower() == "false":
                            bool_values.append(0)
                        # 'None'字符串跳过
                if bool_values:
                    # 分母是所有样本数，包括None的
                    aggregate_metrics[qtype][k] = sum(bool_values) / len(lst)
            else:
                # 处理数值指标
                vals = [m.get(k) for m in lst if isinstance(m.get(k), (int, float))]
                if vals:
                    aggregate_metrics[qtype][k] = sum(vals) / len(vals)
    
    return aggregate_metrics


def update_json_file(json_path: pathlib.Path, dry_run: bool = False) -> bool:
    """
    更新单个JSON文件的aggregate_metrics
    
    Args:
        json_path: JSON文件路径
        dry_run: 是否仅检查，不实际修改
    
    Returns:
        是否成功更新
    """
    try:
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {json_path}")
        
        # 读取文件
        data = json.loads(json_path.read_text(encoding='utf-8'))
        
        # 检查必要字段
        if "results" not in data:
            print(f"  ⚠️  Skipped: No 'results' field")
            return False
        
        if not data["results"]:
            print(f"  ⚠️  Skipped: Empty results list")
            return False
        
        # 重新计算
        old_agg = data.get("aggregate_metrics", {})
        new_agg = recompute_aggregate_metrics(data["results"])
        
        # 对比变化
        print(f"  📊 Question types: {list(new_agg.keys())}")
        
        # 检查Pass/ECR是否新增
        for qtype, metrics in new_agg.items():
            old_metrics = old_agg.get(qtype, {})
            new_keys = set(metrics.keys()) - set(old_metrics.keys())
            if new_keys:
                print(f"  ✨ {qtype}: New metrics added: {new_keys}")
            
            # 显示Pass/ECR值
            if "Pass" in metrics:
                old_pass = old_metrics.get("Pass", "N/A")
                print(f"  📈 {qtype}: Pass@1 = {metrics['Pass']:.4f} (old: {old_pass})")
            if "ECR" in metrics:
                old_ecr = old_metrics.get("ECR", "N/A")
                print(f"  📈 {qtype}: ECR = {metrics['ECR']:.4f} (old: {old_ecr})")
        
        # 更新数据
        data["aggregate_metrics"] = new_agg
        
        # 写入文件
        if not dry_run:
            json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            print(f"  ✅ Updated successfully")
        else:
            print(f"  ℹ️  Would update (dry run mode)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="重新计算results JSON文件的aggregate_metrics"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSON文件路径或目录路径（支持通配符）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅检查，不实际修改文件"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="递归处理目录下所有results_*.json文件"
    )
    
    args = parser.parse_args()
    
    # 收集所有JSON文件
    json_files = []
    for path_str in args.paths:
        path = pathlib.Path(path_str)
        
        if path.is_file() and path.suffix == ".json":
            json_files.append(path)
        elif path.is_dir():
            if args.recursive:
                json_files.extend(path.rglob("results_*.json"))
            else:
                json_files.extend(path.glob("results_*.json"))
        else:
            # 尝试通配符
            json_files.extend(pathlib.Path(".").glob(path_str))
    
    # 去重和排序
    json_files = sorted(set(json_files))
    
    if not json_files:
        print("❌ No JSON files found")
        return 1
    
    print(f"{'=' * 70}")
    print(f"Found {len(json_files)} JSON file(s) to process")
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No files will be modified")
    print(f"{'=' * 70}")
    
    # 处理所有文件
    success_count = 0
    for json_path in json_files:
        if update_json_file(json_path, dry_run=args.dry_run):
            success_count += 1
    
    print(f"\n{'=' * 70}")
    print(f"✅ Successfully processed: {success_count}/{len(json_files)}")
    print(f"{'=' * 70}")
    
    return 0 if success_count == len(json_files) else 1


if __name__ == "__main__":
    sys.exit(main())
