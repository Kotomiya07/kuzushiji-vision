#!/usr/bin/env python3
"""
バウンディングボックスサイズ分析スクリプト

このスクリプトはYOLOデータセットのアノテーションから文字サイズを分析し、
タイル分割時の推奨オーバーラップ率を計算します。

使用例:
    python scripts/analyze_bbox_sizes.py
    python scripts/analyze_bbox_sizes.py --input_dir datasets/my_dataset
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def analyze_bbox_sizes(input_dir: str, split: str = "train") -> dict:
    """
    バウンディングボックスのサイズを分析する

    Args:
        input_dir: データセットのルートディレクトリ
        split: 分析対象の分割 (train, val, test)

    Returns:
        dict: 分析結果
    """
    labels_dir = Path(input_dir) / split / "labels"
    
    if not labels_dir.exists():
        print(f"エラー: ラベルディレクトリが存在しません: {labels_dir}")
        return None

    widths = []
    heights = []

    label_files = list(labels_dir.glob("*.txt"))
    
    for label_file in label_files:
        with open(label_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    w, h = float(parts[3]), float(parts[4])
                    widths.append(w)
                    heights.append(h)

    if len(widths) == 0:
        print("エラー: バウンディングボックスが見つかりませんでした")
        return None

    widths = np.array(widths)
    heights = np.array(heights)

    # 統計計算
    stats = {
        "total_bboxes": len(widths),
        "width": {
            "mean": float(widths.mean()),
            "std": float(widths.std()),
            "min": float(widths.min()),
            "max": float(widths.max()),
            "p50": float(np.percentile(widths, 50)),
            "p75": float(np.percentile(widths, 75)),
            "p90": float(np.percentile(widths, 90)),
            "p95": float(np.percentile(widths, 95)),
            "p99": float(np.percentile(widths, 99)),
        },
        "height": {
            "mean": float(heights.mean()),
            "std": float(heights.std()),
            "min": float(heights.min()),
            "max": float(heights.max()),
            "p50": float(np.percentile(heights, 50)),
            "p75": float(np.percentile(heights, 75)),
            "p90": float(np.percentile(heights, 90)),
            "p95": float(np.percentile(heights, 95)),
            "p99": float(np.percentile(heights, 99)),
        },
    }

    # 推奨オーバーラップ率の計算
    max_dim = max(stats["width"]["max"], stats["height"]["max"])
    p99_dim = max(stats["width"]["p99"], stats["height"]["p99"])
    p95_dim = max(stats["width"]["p95"], stats["height"]["p95"])

    stats["recommended_overlap"] = {
        "conservative": float(p99_dim),  # 99%の文字をカバー
        "safe": float(max_dim),  # 全ての文字をカバー
        "balanced": float(p95_dim),  # 95%の文字をカバー
    }

    return stats


def print_analysis(stats: dict):
    """分析結果を表示"""
    print("\n" + "=" * 60)
    print("         バウンディングボックスサイズ分析結果")
    print("=" * 60)
    
    print(f"\n総バウンディングボックス数: {stats['total_bboxes']:,}")
    
    print("\n【幅（Width）】")
    print(f"  平均:    {stats['width']['mean']:.4f} ({stats['width']['mean']*100:.2f}%)")
    print(f"  標準偏差: {stats['width']['std']:.4f}")
    print(f"  最小:    {stats['width']['min']:.4f}")
    print(f"  最大:    {stats['width']['max']:.4f} ({stats['width']['max']*100:.2f}%)")
    print(f"  50%タイル: {stats['width']['p50']:.4f}")
    print(f"  95%タイル: {stats['width']['p95']:.4f}")
    print(f"  99%タイル: {stats['width']['p99']:.4f}")
    
    print("\n【高さ（Height）】")
    print(f"  平均:    {stats['height']['mean']:.4f} ({stats['height']['mean']*100:.2f}%)")
    print(f"  標準偏差: {stats['height']['std']:.4f}")
    print(f"  最小:    {stats['height']['min']:.4f}")
    print(f"  最大:    {stats['height']['max']:.4f} ({stats['height']['max']*100:.2f}%)")
    print(f"  50%タイル: {stats['height']['p50']:.4f}")
    print(f"  95%タイル: {stats['height']['p95']:.4f}")
    print(f"  99%タイル: {stats['height']['p99']:.4f}")
    
    print("\n" + "-" * 60)
    print("推奨オーバーラップ率:")
    print("-" * 60)
    rec = stats['recommended_overlap']
    print(f"  バランス型 (95%カバー): {rec['balanced']:.4f} ({rec['balanced']*100:.2f}%)")
    print(f"  保守的 (99%カバー):     {rec['conservative']:.4f} ({rec['conservative']*100:.2f}%)")
    print(f"  安全 (100%カバー):      {rec['safe']:.4f} ({rec['safe']*100:.2f}%)")
    
    # 推奨設定の提案
    if rec['conservative'] <= 0.15:
        print(f"\n✓ デフォルトの15%オーバーラップで99%以上の文字がカバーされます")
    elif rec['balanced'] <= 0.15:
        print(f"\n⚠ デフォルトの15%オーバーラップで95%以上の文字がカバーされます")
        print(f"  99%カバーするには --tile_overlap {rec['conservative']:.2f} を使用してください")
    else:
        print(f"\n⚠ 大きな文字があります。--tile_overlap {rec['conservative']:.2f} を推奨します")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="YOLOデータセットのバウンディングボックスサイズを分析し、推奨オーバーラップ率を計算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/yolo_dataset_character_detection",
        help="データセットのルートディレクトリ",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="分析対象の分割 (デフォルト: train)",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"エラー: ディレクトリが存在しません: {args.input_dir}")
        sys.exit(1)
    
    print(f"データセットを分析中: {args.input_dir} ({args.split})")
    
    stats = analyze_bbox_sizes(args.input_dir, args.split)
    
    if stats:
        print_analysis(stats)


if __name__ == "__main__":
    main()
