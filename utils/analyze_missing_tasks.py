#!/usr/bin/env python3
"""
分析所有results.json文件中缺失的任务，并生成skip_ids.json报告。

功能:
1. 加载QA_final_sc_filled.json中的所有任务(3,070个)
2. 扫描result/complied/目录下的所有results.json文件
3. 识别每个文件中缺失的任务ID
4. 分析缺失原因(文件不存在、OOM错误、处理错误等)
5. 在每个结果目录下生成skip_ids.json报告
6. 生成总体统计报告

用法:
    python utils/analyze_missing_tasks.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from datetime import datetime


# 文件扩展名映射
FILE_EXTENSIONS = {
    "latex": "txt",
    "markdown": "md",
    "csv": "csv",
    "html": "html"
}


class MissingTaskAnalyzer:
    """分析缺失任务的主类"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.qa_file = self.workspace_root / "data" / "QA_final_sc_filled.json"
        self.data_path = Path("/data/pan/4xin/datasets/RealHiTBench")
        self.result_base = self.workspace_root / "result" / "complied"
        
        # 加载主任务列表
        self.all_queries = self._load_master_task_list()
        self.all_task_ids = set(q['id'] for q in self.all_queries)
        self.query_by_id = {q['id']: q for q in self.all_queries}
        
        print(f"✓ 加载主任务列表: {len(self.all_task_ids)} 个任务")
        print(f"  任务ID范围: {min(self.all_task_ids)} - {max(self.all_task_ids)}")
        print()
    
    def _load_master_task_list(self) -> List[Dict]:
        """加载QA_final_sc_filled.json中的所有任务"""
        with open(self.qa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['queries']
    
    def _parse_result_path(self, result_path: Path) -> Tuple[str, str, str]:
        """
        从结果文件路径解析配置信息
        
        返回: (config_name, modality, format)
        例如: qwen3vl_resize_pic/mix_html/results.json
             -> ("qwen3vl_resize_pic", "mix", "html")
        """
        parts = result_path.relative_to(self.result_base).parts
        
        config_name = parts[0]  # qwen3vl_default_pic, qwen3vl_resize_pic, qwen3vl_text
        modality_format = parts[1]  # image, mix_html, text_csv, etc.
        
        # 解析modality和format
        if modality_format == 'image':
            modality = 'image'
            format_type = None
        elif modality_format.startswith('mix_'):
            modality = 'mix'
            format_type = modality_format.split('_', 1)[1]  # html, csv, latex
        elif modality_format.startswith('text_'):
            modality = 'text'
            format_type = modality_format.split('_', 1)[1]  # html, csv, latex
        else:
            raise ValueError(f"Unknown modality/format pattern: {modality_format}")
        
        return config_name, modality, format_type
    
    def _check_file_exists(self, filename: str, modality: str, format_type: str = None, 
                          question_type: str = None) -> Dict[str, bool]:
        """
        检查任务所需的文件是否存在
        
        返回: {"image": True/False, "text": True/False, "xlsx": True/False}
        """
        result = {}
        
        # 检查image文件 (.png)
        if modality in ['image', 'mix']:
            image_path = self.data_path / "image" / f"{filename}.png"
            result['image'] = image_path.exists()
        
        # 检查text文件 (.csv/.html/.txt for latex)
        if modality in ['text', 'mix'] and format_type:
            ext = FILE_EXTENSIONS[format_type]
            text_path = self.data_path / format_type / f"{filename}.{ext}"
            result['text'] = text_path.exists()
        
        # 检查Visualization所需的xlsx文件
        if question_type == 'Visualization':
            xlsx_path = self.data_path / "tables" / f"{filename}.xlsx"
            result['xlsx'] = xlsx_path.exists()
        
        return result
    
    def _determine_skip_reason(self, task_id: int, modality: str, format_type: str = None) -> str:
        """
        确定任务被跳过的原因
        
        可能的原因:
        1. 图像文件不存在 (image/mix modality)
        2. 文本文件不存在 (text/mix modality)
        3. Visualization所需xlsx文件不存在
        4. 未完成运行 (文件存在但未处理)
        5. 处理错误 (OOM、异常等)
        """
        query = self.query_by_id[task_id]
        filename = query['FileName']
        question_type = query['QuestionType']
        
        # 检查文件依赖
        file_status = self._check_file_exists(filename, modality, format_type, question_type)
        
        missing_files = []
        
        # Image模态或Mix模态需要png文件
        if 'image' in file_status and not file_status['image']:
            missing_files.append(f"{filename}.png")
        
        # Text模态或Mix模态需要格式文件
        if 'text' in file_status and not file_status['text']:
            ext = FILE_EXTENSIONS[format_type]
            missing_files.append(f"{filename}.{ext}")
        
        # Visualization任务需要xlsx文件
        if 'xlsx' in file_status and not file_status['xlsx']:
            missing_files.append(f"{filename}.xlsx")
        
        # 构建原因描述
        if missing_files:
            return f"Missing source file(s): {', '.join(missing_files)}"
        else:
            return "Incomplete run or processing error (files exist but task not processed)"
    
    def _load_result_file(self, result_path: Path) -> Dict:
        """加载结果JSON文件"""
        with open(result_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _analyze_single_result_file(self, result_path: Path) -> Dict:
        """
        分析单个结果文件
        
        返回包含以下信息的字典:
        - completed_ids: 已完成的任务ID集合
        - success_ids: 成功的任务ID集合
        - error_ids: 有错误的任务ID集合
        - missing_ids: 缺失的任务ID集合
        - skip_reasons: {task_id: reason} 缺失原因映射
        """
        print(f"  分析: {result_path.relative_to(self.result_base)}")
        
        # 解析配置
        config_name, modality, format_type = self._parse_result_path(result_path)
        
        # 加载结果数据
        result_data = self._load_result_file(result_path)
        results = result_data.get('results', [])
        
        # 分类任务
        completed_ids = set()
        success_ids = set()
        error_ids = set()
        
        for r in results:
            task_id = r['id']
            completed_ids.add(task_id)
            
            prediction = r.get('Prediction', '')
            if isinstance(prediction, str) and prediction.startswith('[ERROR]'):
                error_ids.add(task_id)
            else:
                success_ids.add(task_id)
        
        # 找出缺失的任务
        missing_ids = self.all_task_ids - completed_ids
        
        # 分析缺失原因
        skip_reasons = {}
        file_dependency_issues = []
        incomplete_runs = []
        
        for task_id in sorted(missing_ids):
            reason = self._determine_skip_reason(task_id, modality, format_type)
            skip_reasons[task_id] = reason
            
            if "Missing source file" in reason:
                file_dependency_issues.append(task_id)
            else:
                incomplete_runs.append(task_id)
        
        print(f"    ✓ 已完成: {len(completed_ids)}")
        print(f"      - 成功: {len(success_ids)}")
        print(f"      - 错误: {len(error_ids)}")
        print(f"    ✗ 缺失: {len(missing_ids)}")
        print(f"      - 文件依赖问题: {len(file_dependency_issues)}")
        print(f"      - 未完成运行: {len(incomplete_runs)}")
        
        return {
            'result_path': str(result_path.relative_to(self.result_base)),
            'config_name': config_name,
            'modality': modality,
            'format': format_type,
            'total_tasks': len(self.all_task_ids),
            'completed_count': len(completed_ids),
            'success_count': len(success_ids),
            'error_count': len(error_ids),
            'missing_count': len(missing_ids),
            'completed_ids': sorted(list(completed_ids)),
            'success_ids': sorted(list(success_ids)),
            'error_ids': sorted(list(error_ids)),
            'missing_ids': sorted(list(missing_ids)),
            'skip_reasons': {str(k): v for k, v in skip_reasons.items()},
            'file_dependency_issues': file_dependency_issues,
            'incomplete_runs': incomplete_runs,
            'analysis_time': datetime.now().isoformat()
        }
    
    def _save_skip_ids_report(self, analysis: Dict, result_dir: Path):
        """在结果目录下保存skip_ids.json报告"""
        skip_ids_file = result_dir / "skip_ids.json"
        
        # 准备报告数据
        report = {
            'metadata': {
                'result_file': analysis['result_path'],
                'config': analysis['config_name'],
                'modality': analysis['modality'],
                'format': analysis['format'],
                'analysis_time': analysis['analysis_time'],
                'master_task_list': str(self.qa_file.relative_to(self.workspace_root))
            },
            'statistics': {
                'total_tasks': analysis['total_tasks'],
                'completed': analysis['completed_count'],
                'success': analysis['success_count'],
                'error': analysis['error_count'],
                'missing': analysis['missing_count'],
                'file_dependency_issues': len(analysis['file_dependency_issues']),
                'incomplete_runs': len(analysis['incomplete_runs'])
            },
            'skip_ids': analysis['missing_ids'],
            'error_ids': analysis['error_ids'],
            'skip_reasons': analysis['skip_reasons'],
            'categorized': {
                'file_dependency_issues': analysis['file_dependency_issues'],
                'incomplete_runs': analysis['incomplete_runs']
            }
        }
        
        # 保存
        with open(skip_ids_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"    → 保存: {skip_ids_file.relative_to(self.workspace_root)}")
    
    def find_all_result_files(self) -> List[Path]:
        """扫描result/complied目录，找到所有results.json文件"""
        result_files = []
        
        for config_dir in self.result_base.iterdir():
            if not config_dir.is_dir():
                continue
            
            for modality_dir in config_dir.iterdir():
                if not modality_dir.is_dir():
                    continue
                
                # 查找results.json或results_*.json
                for result_file in modality_dir.glob('results*.json'):
                    # 排除已经生成的skip_ids.json
                    if result_file.name != 'skip_ids.json':
                        result_files.append(result_file)
        
        return sorted(result_files)
    
    def analyze_all_results(self) -> List[Dict]:
        """分析所有结果文件"""
        result_files = self.find_all_result_files()
        
        print("=" * 80)
        print(f"扫描到 {len(result_files)} 个结果文件")
        print("=" * 80)
        print()
        
        all_analyses = []
        
        for i, result_path in enumerate(result_files, 1):
            print(f"[{i}/{len(result_files)}]")
            
            # 分析单个文件
            analysis = self._analyze_single_result_file(result_path)
            all_analyses.append(analysis)
            
            # 保存skip_ids.json
            result_dir = result_path.parent
            self._save_skip_ids_report(analysis, result_dir)
            print()
        
        return all_analyses
    
    def generate_summary_report(self, all_analyses: List[Dict]) -> Dict:
        """生成总体统计报告"""
        print("=" * 80)
        print("生成总体统计报告")
        print("=" * 80)
        print()
        
        # 按配置分组统计
        by_config = defaultdict(list)
        by_modality = defaultdict(list)
        by_format = defaultdict(list)
        
        for analysis in all_analyses:
            by_config[analysis['config_name']].append(analysis)
            by_modality[analysis['modality']].append(analysis)
            if analysis['format']:
                by_format[analysis['format']].append(analysis)
        
        # 统计常见缺失任务
        missing_task_frequency = defaultdict(int)
        for analysis in all_analyses:
            for task_id in analysis['missing_ids']:
                missing_task_frequency[task_id] += 1
        
        # 找出在所有配置中都缺失的任务
        all_missing = [tid for tid, count in missing_task_frequency.items() 
                       if count == len(all_analyses)]
        
        # 找出在多个配置中缺失的任务
        frequently_missing = [(tid, count) for tid, count in missing_task_frequency.items() 
                              if count >= 3]
        frequently_missing.sort(key=lambda x: x[1], reverse=True)
        
        # 总体统计
        total_completed = sum(a['completed_count'] for a in all_analyses)
        total_missing = sum(a['missing_count'] for a in all_analyses)
        total_errors = sum(a['error_count'] for a in all_analyses)
        
        summary = {
            'metadata': {
                'analysis_time': datetime.now().isoformat(),
                'total_result_files': len(all_analyses),
                'total_tasks_in_master_list': len(self.all_task_ids)
            },
            'overall_statistics': {
                'total_completed_across_all_files': total_completed,
                'total_missing_across_all_files': total_missing,
                'total_errors_across_all_files': total_errors,
                'average_completion_rate': f"{total_completed / (len(all_analyses) * len(self.all_task_ids)) * 100:.2f}%"
            },
            'by_configuration': {},
            'by_modality': {},
            'by_format': {},
            'missing_task_patterns': {
                'always_missing': all_missing,
                'frequently_missing': [{'task_id': tid, 'missing_in_files': count} 
                                       for tid, count in frequently_missing[:20]]
            },
            'recommendations': []
        }
        
        # 配置统计
        for config_name, analyses in by_config.items():
            summary['by_configuration'][config_name] = {
                'file_count': len(analyses),
                'avg_completion': sum(a['completed_count'] for a in analyses) / len(analyses),
                'avg_missing': sum(a['missing_count'] for a in analyses) / len(analyses)
            }
        
        # 模态统计
        for modality, analyses in by_modality.items():
            summary['by_modality'][modality] = {
                'file_count': len(analyses),
                'avg_completion': sum(a['completed_count'] for a in analyses) / len(analyses),
                'avg_missing': sum(a['missing_count'] for a in analyses) / len(analyses)
            }
        
        # 格式统计
        for format_type, analyses in by_format.items():
            summary['by_format'][format_type] = {
                'file_count': len(analyses),
                'avg_completion': sum(a['completed_count'] for a in analyses) / len(analyses),
                'avg_missing': sum(a['missing_count'] for a in analyses) / len(analyses)
            }
        
        # 生成建议
        if all_missing:
            summary['recommendations'].append({
                'priority': 'HIGH',
                'issue': f'{len(all_missing)} tasks missing in ALL result files',
                'action': 'Check if source files exist for these tasks across all formats'
            })
        
        if frequently_missing:
            summary['recommendations'].append({
                'priority': 'MEDIUM',
                'issue': f'{len(frequently_missing)} tasks missing in 3+ result files',
                'action': 'Investigate file dependencies and consider format conversion'
            })
        
        # 识别最需要关注的配置
        worst_completion = min(all_analyses, key=lambda x: x['completed_count'])
        summary['recommendations'].append({
            'priority': 'MEDIUM',
            'issue': f"Lowest completion: {worst_completion['result_path']} ({worst_completion['completed_count']} tasks)",
            'action': f"Focus on completing {worst_completion['missing_count']} missing tasks"
        })
        
        return summary
    
    def save_summary_report(self, summary: Dict):
        """保存总体统计报告"""
        summary_file = self.workspace_root / "utils" / "missing_tasks_summary.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 总体报告已保存: {summary_file.relative_to(self.workspace_root)}")
        print()
        
        # 打印关键统计
        print("关键统计:")
        print(f"  总结果文件数: {summary['metadata']['total_result_files']}")
        print(f"  主任务列表任务数: {summary['metadata']['total_tasks_in_master_list']}")
        print(f"  平均完成率: {summary['overall_statistics']['average_completion_rate']}")
        print(f"  所有文件都缺失的任务: {len(summary['missing_task_patterns']['always_missing'])}")
        print(f"  频繁缺失的任务(3+): {len(summary['missing_task_patterns']['frequently_missing'])}")
        print()
        
        # 打印建议
        if summary['recommendations']:
            print("建议:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  [{rec['priority']}] {rec['issue']}")
                print(f"        → {rec['action']}")
            print()
    
    def run(self):
        """运行完整的分析流程"""
        print("=" * 80)
        print("RealHiTBench 缺失任务分析工具")
        print("=" * 80)
        print()
        
        # 分析所有结果文件
        all_analyses = self.analyze_all_results()
        
        # 生成总体报告
        summary = self.generate_summary_report(all_analyses)
        
        # 保存总体报告
        self.save_summary_report(summary)
        
        print("=" * 80)
        print("✅ 分析完成！")
        print("=" * 80)
        print()
        print("生成的文件:")
        print(f"  • 每个结果目录下的 skip_ids.json (共 {len(all_analyses)} 个)")
        print(f"  • utils/missing_tasks_summary.json (总体报告)")
        print()


def main():
    """主函数"""
    workspace_root = "/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL"
    
    analyzer = MissingTaskAnalyzer(workspace_root)
    analyzer.run()


if __name__ == '__main__':
    main()
