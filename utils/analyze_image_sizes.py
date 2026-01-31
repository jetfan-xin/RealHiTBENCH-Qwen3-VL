#!/usr/bin/env python3
"""
分析 RealHiTBench 数据集中所有图片的大小，
找出小于 min_pixels 和大于 max_pixels 的图片。
"""

import os
import sys
from PIL import Image
from pathlib import Path
from collections import defaultdict

# 禁用 PIL 的大图限制
Image.MAX_IMAGE_PIXELS = None

# Qwen3-VL 的像素配置（当前代码中的设置）
# 注意：Qwen3-VL 正确的 factor 应该是 32，但当前代码用的是 28
CURRENT_MIN_PIXELS = 256 * 28 * 28   # ~200,704 pixels
CURRENT_MAX_PIXELS = 2048 * 28 * 28  # ~1,605,632 pixels

# Qwen3-VL 正确的 factor=32 设置
CORRECT_MIN_PIXELS = 256 * 32 * 32   # ~262,144 pixels  
CORRECT_MAX_PIXELS = 2048 * 32 * 32  # ~2,097,152 pixels

# 官方默认值
OFFICIAL_MIN_PIXELS = 4 * 32 * 32        # ~4,096 pixels
OFFICIAL_MAX_PIXELS = 16384 * 32 * 32    # ~16,777,216 pixels


def get_image_info(image_path):
    """获取图片的尺寸信息"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            pixels = width * height
            return {
                'path': image_path,
                'filename': os.path.basename(image_path),
                'width': width,
                'height': height,
                'pixels': pixels,
                'megapixels': pixels / 1_000_000,
                'size_kb': os.path.getsize(image_path) / 1024
            }
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None


def format_pixels(pixels):
    """格式化像素数显示"""
    if pixels >= 1_000_000:
        return f"{pixels/1_000_000:.2f}M"
    elif pixels >= 1_000:
        return f"{pixels/1_000:.1f}K"
    else:
        return str(pixels)


def analyze_images(image_dir, min_pixels, max_pixels, config_name=""):
    """分析图片目录"""
    print(f"\n{'='*80}")
    print(f"配置: {config_name}")
    print(f"min_pixels = {min_pixels:,} ({format_pixels(min_pixels)})")
    print(f"max_pixels = {max_pixels:,} ({format_pixels(max_pixels)})")
    print(f"{'='*80}")
    
    # 收集所有图片信息
    all_images = []
    
    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    
    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if Path(filename).suffix.lower() in image_extensions:
                filepath = os.path.join(root, filename)
                info = get_image_info(filepath)
                if info:
                    all_images.append(info)
    
    if not all_images:
        print(f"未找到图片在: {image_dir}")
        return
    
    # 按像素数排序（从大到小）
    all_images.sort(key=lambda x: x['pixels'], reverse=True)
    
    # 分类
    too_small = [img for img in all_images if img['pixels'] < min_pixels]
    too_large = [img for img in all_images if img['pixels'] > max_pixels]
    in_range = [img for img in all_images if min_pixels <= img['pixels'] <= max_pixels]
    
    # 统计信息
    print(f"\n📊 统计摘要:")
    print(f"  总图片数: {len(all_images)}")
    print(f"  < min_pixels ({format_pixels(min_pixels)}): {len(too_small)} 张 ({100*len(too_small)/len(all_images):.1f}%)")
    print(f"  > max_pixels ({format_pixels(max_pixels)}): {len(too_large)} 张 ({100*len(too_large)/len(all_images):.1f}%)")
    print(f"  在范围内: {len(in_range)} 张 ({100*len(in_range)/len(all_images):.1f}%)")
    
    # 像素分布
    pixels_list = [img['pixels'] for img in all_images]
    print(f"\n📈 像素分布:")
    print(f"  最小: {format_pixels(min(pixels_list))} ({all_images[-1]['filename']})")
    print(f"  最大: {format_pixels(max(pixels_list))} ({all_images[0]['filename']})")
    print(f"  中位数: {format_pixels(sorted(pixels_list)[len(pixels_list)//2])}")
    print(f"  平均: {format_pixels(int(sum(pixels_list)/len(pixels_list)))}")
    
    # 显示超大图片（> max_pixels）
    if too_large:
        print(f"\n🔴 超过 max_pixels 的图片 ({len(too_large)} 张，按大小降序):")
        print(f"{'排名':<6} {'文件名':<50} {'尺寸':<20} {'像素数':<15} {'超出倍数':<10}")
        print("-" * 110)
        for i, img in enumerate(too_large[:50], 1):  # 只显示前50张
            ratio = img['pixels'] / max_pixels
            print(f"{i:<6} {img['filename']:<50} {img['width']}x{img['height']:<12} {format_pixels(img['pixels']):<15} {ratio:.2f}x")
        if len(too_large) > 50:
            print(f"  ... 还有 {len(too_large) - 50} 张")
    
    # 显示过小图片（< min_pixels）
    if too_small:
        print(f"\n🟡 小于 min_pixels 的图片 ({len(too_small)} 张，按大小降序):")
        print(f"{'排名':<6} {'文件名':<50} {'尺寸':<20} {'像素数':<15}")
        print("-" * 100)
        for i, img in enumerate(too_small[:30], 1):  # 只显示前30张
            print(f"{i:<6} {img['filename']:<50} {img['width']}x{img['height']:<12} {format_pixels(img['pixels']):<15}")
        if len(too_small) > 30:
            print(f"  ... 还有 {len(too_small) - 30} 张")
    
    # 显示 _swap 图片分析
    swap_images = [img for img in all_images if '_swap' in img['filename']]
    non_swap_images = [img for img in all_images if '_swap' not in img['filename']]
    
    if swap_images:
        print(f"\n🔄 _swap 图片分析 (SC-filled 使用):")
        swap_too_large = [img for img in swap_images if img['pixels'] > max_pixels]
        print(f"  _swap 图片总数: {len(swap_images)}")
        print(f"  _swap 超过 max_pixels: {len(swap_too_large)} 张 ({100*len(swap_too_large)/len(swap_images):.1f}%)")
        if swap_images:
            swap_pixels = [img['pixels'] for img in swap_images]
            print(f"  _swap 最大: {format_pixels(max(swap_pixels))}")
            print(f"  _swap 平均: {format_pixels(int(sum(swap_pixels)/len(swap_pixels)))}")
    
    if non_swap_images:
        print(f"\n📷 原始图片分析:")
        non_swap_too_large = [img for img in non_swap_images if img['pixels'] > max_pixels]
        print(f"  原始图片总数: {len(non_swap_images)}")
        print(f"  原始图片超过 max_pixels: {len(non_swap_too_large)} 张 ({100*len(non_swap_too_large)/len(non_swap_images):.1f}%)")
        if non_swap_images:
            non_swap_pixels = [img['pixels'] for img in non_swap_images]
            print(f"  原始图片最大: {format_pixels(max(non_swap_pixels))}")
            print(f"  原始图片平均: {format_pixels(int(sum(non_swap_pixels)/len(non_swap_pixels)))}")
    
    return {
        'total': len(all_images),
        'too_small': too_small,
        'too_large': too_large,
        'in_range': in_range,
        'all_images': all_images
    }


def main():
    # 数据集路径
    image_dir = "/data/pan/4xin/datasets/RealHiTBench/image"
    
    if not os.path.exists(image_dir):
        print(f"错误: 图片目录不存在: {image_dir}")
        sys.exit(1)
    
    print("="*80)
    print("RealHiTBench 图片大小分析")
    print("="*80)
    print(f"图片目录: {image_dir}")
    
    # 分析 1: 当前代码配置 (factor=28，实际上是错误的)
    result1 = analyze_images(
        image_dir, 
        CURRENT_MIN_PIXELS, 
        CURRENT_MAX_PIXELS,
        "当前代码配置 (factor=28, 错误)"
    )
    
    # 分析 2: 正确的 Qwen3-VL 配置 (factor=32)
    result2 = analyze_images(
        image_dir,
        CORRECT_MIN_PIXELS,
        CORRECT_MAX_PIXELS,
        "正确 Qwen3-VL 配置 (factor=32)"
    )
    
    # 分析 3: 官方默认配置
    result3 = analyze_images(
        image_dir,
        OFFICIAL_MIN_PIXELS,
        OFFICIAL_MAX_PIXELS,
        "官方默认配置"
    )
    
    # 总结建议
    print("\n" + "="*80)
    print("💡 建议")
    print("="*80)
    
    if result1:
        large_ratio = len(result1['too_large']) / result1['total'] * 100
        print(f"""
当前配置 (max={format_pixels(CURRENT_MAX_PIXELS)}):
  - {len(result1['too_large'])}/{result1['total']} 张图片超限 ({large_ratio:.1f}%)
  - 这些图片会被 resize 到 ~{format_pixels(CURRENT_MAX_PIXELS)}

官方默认 (max={format_pixels(OFFICIAL_MAX_PIXELS)}):  
  - {len(result3['too_large'])}/{result3['total']} 张图片超限
  - 大图片保持更高分辨率，但显存消耗大幅增加
  - ⚠️ 可能导致 OOM，特别是 mix 模态（图片+长文本）

建议:
  1. 如果继续使用自定义配置，修正 factor 为 32:
     min_pixels = 256 * 32 * 32  # = {256*32*32:,}
     max_pixels = 2048 * 32 * 32 # = {2048*32*32:,}
  
  2. 当前保守的 max_pixels 设置对于防止 OOM 是合理的
""")


if __name__ == "__main__":
    main()
