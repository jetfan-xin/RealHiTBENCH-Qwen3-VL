#!/usr/bin/env python3
"""
针对skip_ids.json中缺失的任务进行推理（排除文件缺失的任务）

功能:
1. 读取各个结果目录下的skip_ids.json
2. 提取需要重新推理的任务ID（排除file_dependency_issues）
3. 为每个配置生成专门的推理脚本，仅处理缺失的任务
4. 支持text_truncation来处理OOM问题

用法:
    python utils/generate_missing_task_inference.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class MissingTaskInferenceGenerator:
    """生成缺失任务推理脚本的类"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.result_base = self.workspace_root / "result" / "complied"
        self.inference_dir = self.workspace_root / "inference"
        self.output_dir = self.workspace_root / "inference" / "rerun_missing_tasks"
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        
    def find_skip_ids_files(self) -> List[Path]:
        """查找所有skip_ids.json文件"""
        skip_files = []
        for skip_file in self.result_base.rglob("skip_ids.json"):
            skip_files.append(skip_file)
        return sorted(skip_files)
    
    def load_skip_ids(self, skip_file: Path) -> Dict:
        """加载skip_ids.json文件"""
        with open(skip_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_inference_script(self, skip_data: Dict, output_file: Path):
        """
        生成针对特定配置的推理脚本
        
        参数:
            skip_data: skip_ids.json的内容
            output_file: 输出脚本路径
        """
        metadata = skip_data['metadata']
        config = metadata['config']
        modality = metadata['modality']
        format_type = metadata.get('format', None)
        
        # 获取需要重新运行的任务ID（排除文件依赖问题）
        incomplete_runs = skip_data['categorized']['incomplete_runs']
        error_ids = skip_data.get('error_ids', [])
        
        # 合并需要重新运行的ID
        rerun_ids = sorted(set(incomplete_runs + error_ids))
        
        if not rerun_ids:
            print(f"  跳过 {config}/{modality}_{format_type or 'image'}: 无需重新运行的任务")
            return None
        
        # 选择inference脚本
        # 如果有OOM错误，使用truncate版本
        use_truncate = len(error_ids) > 0 or modality in ['text', 'mix']
        
        if use_truncate:
            inference_script = "inference_qwen3vl_local_a100_truncate_with_task_ids.py"
            comment = f"# 使用truncate版本防止OOM (error_ids: {len(error_ids)})"
        else:
            inference_script = "inference_qwen3vl_local_a100_default.py"
            comment = f"# 使用default版本 - 需要手动添加--task_ids支持"
        
        # 生成脚本内容
        script_content = f'''#!/usr/bin/env python3
"""
重新运行缺失的任务: {config}/{modality}_{format_type or 'image'}

生成时间: {datetime.now().isoformat()}
任务来源: {metadata['result_file']}
需要重新运行的任务数: {len(rerun_ids)}
  - Incomplete runs: {len(incomplete_runs)}
  - Error tasks: {len(error_ids)}

{comment}
"""

import subprocess
import sys
import os
import json

def main():
    # 配置
    modality = "{modality}"
    format_type = {f'"{format_type}"' if format_type else 'None'}
    batch_size = 1  # 使用batch_size=1避免OOM
    model_dir = "/data/pan/4xin/models/Qwen3-VL-8B-Instruct"
    data_path = "/data/pan/4xin/datasets/RealHiTBench"
    qa_path = "/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data"
    
    # 需要重新运行的任务ID
    task_ids = {rerun_ids}
    
    print("=" * 80)
    print(f"重新运行缺失任务: {{modality}}" + (f"_{{format_type}}" if format_type else ""))
    print("=" * 80)
    print(f"配置: {config}")
    print(f"任务数量: {{len(task_ids)}}")
    print(f"Inference脚本: {inference_script}")
    print(f"使用文本截断: {use_truncate}")
    print()
    
    # 构建命令
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inference_path = os.path.join(script_dir, "{inference_script}")
    
    cmd = [
        sys.executable,
        inference_path,
        "--modality", modality,
        "--model_dir", model_dir,
        "--data_path", data_path,
        "--qa_path", qa_path,
        "--use_sc_filled",
        "--batch_size", str(batch_size),
        "--task_ids", ",".join(map(str, task_ids)),  # 指定任务ID
        "--resume"  # 使用resume模式，会加载已有checkpoint并合并
    ]
    
    if format_type:
        cmd.extend(["--format", format_type])
    
    print(f"命令: {{' '.join(cmd)}}")
    print("-" * 80)
    print()
    
    # 运行推理
    result = subprocess.run(cmd, cwd=script_dir)
    
    if result.returncode == 0:
        print()
        print("=" * 80)
        print(f"✓ 成功完成 {{modality}}" + (f"_{{format_type}}" if format_type else "") + f" 的 {{len(task_ids)}} 个缺失任务")
        print("=" * 80)
    else:
        print()
        print("=" * 80)
        print(f"✗ 运行失败，退出码: {{result.returncode}}")
        print("=" * 80)
        sys.exit(result.returncode)

if __name__ == '__main__':
    main()
'''
        
        # 保存脚本
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 添加执行权限
        os.chmod(output_file, 0o755)
        
        return {
            'config': config,
            'modality': modality,
            'format': format_type,
            'script_path': str(output_file.relative_to(self.workspace_root)),
            'rerun_count': len(rerun_ids),
            'incomplete_count': len(incomplete_runs),
            'error_count': len(error_ids),
            'use_truncate': use_truncate
        }
    
    def generate_master_script(self, script_info_list: List[Dict]):
        """生成主运行脚本，按顺序执行所有重新运行任务"""
        master_script = self.output_dir / "run_all_missing_tasks.sh"
        
        script_content = f'''#!/bin/bash
# 主脚本：运行所有缺失任务的推理
# 生成时间: {datetime.now().isoformat()}
# 总共需要运行: {len(script_info_list)} 个配置

set -e  # 遇到错误时退出

echo "========================================================================"
echo "运行所有缺失任务的推理"
echo "========================================================================"
echo "总配置数: {len(script_info_list)}"
echo "总任务数: {sum(info['rerun_count'] for info in script_info_list)}"
echo ""

'''
        
        for i, info in enumerate(script_info_list, 1):
            format_str = f"_{info['format']}" if info['format'] else ""
            script_content += f'''
echo "------------------------------------------------------------------------"
echo "[{i}/{len(script_info_list)}] 运行: {info['config']}/{info['modality']}{format_str}"
echo "  任务数: {info['rerun_count']} (incomplete: {info['incomplete_count']}, error: {info['error_count']})"
echo "  使用truncate: {info['use_truncate']}"
echo "------------------------------------------------------------------------"

python {info['script_path']}

if [ $? -ne 0 ]; then
    echo "✗ 失败: {info['config']}/{info['modality']}{format_str}"
    exit 1
fi

echo "✓ 完成: {info['config']}/{info['modality']}{format_str}"
echo ""

'''
        
        script_content += '''
echo "========================================================================"
echo "✅ 所有缺失任务推理完成！"
echo "========================================================================"
'''
        
        with open(master_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        os.chmod(master_script, 0o755)
        
        return master_script
    
    def generate_summary(self, script_info_list: List[Dict]):
        """生成推理任务摘要"""
        summary_file = self.output_dir / "missing_tasks_summary.txt"
        
        total_tasks = sum(info['rerun_count'] for info in script_info_list)
        total_incomplete = sum(info['incomplete_count'] for info in script_info_list)
        total_errors = sum(info['error_count'] for info in script_info_list)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"缺失任务推理摘要\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n\n")
            
            f.write(f"总体统计:\n")
            f.write(f"  配置数量: {len(script_info_list)}\n")
            f.write(f"  总任务数: {total_tasks}\n")
            f.write(f"    - Incomplete runs: {total_incomplete}\n")
            f.write(f"    - Error tasks: {total_errors}\n\n")
            
            f.write(f"按配置详细信息:\n")
            f.write(f"{'-' * 80}\n")
            
            for info in script_info_list:
                format_str = f"_{info['format']}" if info['format'] else ""
                f.write(f"\n配置: {info['config']}/{info['modality']}{format_str}\n")
                f.write(f"  任务数: {info['rerun_count']}\n")
                f.write(f"    - Incomplete: {info['incomplete_count']}\n")
                f.write(f"    - Error: {info['error_count']}\n")
                f.write(f"  使用truncate: {'是' if info['use_truncate'] else '否'}\n")
                f.write(f"  脚本: {info['script_path']}\n")
        
        return summary_file
    
    def run(self):
        """运行完整的生成流程"""
        print("=" * 80)
        print("生成缺失任务推理脚本")
        print("=" * 80)
        print()
        
        # 查找所有skip_ids.json
        skip_files = self.find_skip_ids_files()
        print(f"找到 {len(skip_files)} 个skip_ids.json文件\n")
        
        # 为每个配置生成推理脚本
        script_info_list = []
        
        for skip_file in skip_files:
            skip_data = self.load_skip_ids(skip_file)
            metadata = skip_data['metadata']
            config = metadata['config']
            modality = metadata['modality']
            format_type = metadata.get('format', 'image')
            
            # 生成脚本文件名
            script_name = f"rerun_{config}_{modality}_{format_type}.py"
            output_file = self.output_dir / script_name
            
            print(f"处理: {config}/{modality}_{format_type}")
            
            script_info = self.generate_inference_script(skip_data, output_file)
            
            if script_info:
                script_info_list.append(script_info)
                print(f"  ✓ 生成脚本: {output_file.relative_to(self.workspace_root)}")
                print(f"    任务数: {script_info['rerun_count']}")
            
            print()
        
        # 生成主运行脚本
        if script_info_list:
            master_script = self.generate_master_script(script_info_list)
            print(f"✓ 生成主脚本: {master_script.relative_to(self.workspace_root)}")
            print()
            
            # 生成摘要
            summary_file = self.generate_summary(script_info_list)
            print(f"✓ 生成摘要: {summary_file.relative_to(self.workspace_root)}")
            print()
        
        # 打印统计
        print("=" * 80)
        print("生成完成")
        print("=" * 80)
        print()
        print(f"生成的脚本数量: {len(script_info_list)}")
        print(f"总任务数: {sum(info['rerun_count'] for info in script_info_list)}")
        print()
        print("运行方式:")
        print(f"  1. 运行所有: bash {master_script.relative_to(self.workspace_root)}")
        print(f"  2. 单独运行: python inference/rerun_missing_tasks/rerun_*.py")
        print()


def main():
    workspace_root = "/ltstorage/home/pan/4xin/RealHiTBENCH-Qwen3-VL"
    # workspace_root = "/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL"
    generator = MissingTaskInferenceGenerator(workspace_root)
    generator.run()


if __name__ == '__main__':
    main()
