#!/usr/bin/env python3
"""
YOLOデータセットの画像枚数と文字インスタンス数をカウントするスクリプト

このスクリプトは、/data/yolo_dataset_character_detection/ 以下の
train, val, test各データセットに含まれる画像枚数および文字インスタンス数を
カウントして結果を表示します。
"""

import glob
import os


def count_images_and_annotations(dataset_dir: str) -> dict[str, tuple[int, int]]:
    """
    各データセット（train, val, test）の画像枚数と文字インスタンス数をカウント

    Args:
        dataset_dir: YOLOデータセットのベースディレクトリパス

    Returns:
        各データセットの結果を含む辞書
        キー: データセット名 (train, val, test)
        値: (画像枚数, 文字インスタンス数) のタプル
    """
    results = {}

    # 各データセットをチェック
    for subset in ["train", "val", "test"]:
        subset_dir = os.path.join(dataset_dir, subset)

        if not os.path.exists(subset_dir):
            print(f"警告: {subset_dir} が存在しません")
            results[subset] = (0, 0)
            continue

        # 画像ディレクトリとラベルディレクトリのパス
        images_dir = os.path.join(subset_dir, "images")
        labels_dir = os.path.join(subset_dir, "labels")

        # 画像枚数をカウント
        image_count = 0
        if os.path.exists(images_dir):
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            for ext in image_extensions:
                image_count += len(glob.glob(os.path.join(images_dir, ext)))

        # 文字インスタンス数をカウント（ラベルファイルの行数の合計）
        annotation_count = 0
        if os.path.exists(labels_dir):
            label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
            for label_file in label_files:
                try:
                    with open(label_file, encoding="utf-8") as f:
                        # 空行を除いて行数をカウント
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        annotation_count += len(lines)
                except OSError as e:
                    print(f"警告: {label_file} の読み込みに失敗しました: {e}")

        results[subset] = (image_count, annotation_count)
        print(f"{subset}: 画像枚数={image_count}, 文字インスタンス数={annotation_count}")

    return results


import argparse


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="YOLOデータセットの統計情報を表示")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/yolo_dataset_character_detection",
        help="YOLOデータセットのベースディレクトリパス"
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir

    print("YOLOデータセットの統計情報")
    print("=" * 50)
    print(f"データセットディレクトリ: {dataset_dir}")
    print()

    # 各データセットの統計を取得
    results = count_images_and_annotations(dataset_dir)

    # 結果をまとめて表示
    print("\n" + "=" * 50)
    print("統計結果サマリー")
    print("=" * 50)

    total_images = 0
    total_annotations = 0

    for subset in ["train", "val", "test"]:
        if subset in results:
            image_count, annotation_count = results[subset]
            total_images += image_count
            total_annotations += annotation_count
            print(f"{subset:>5}: 画像 {image_count:>6} 枚, 文字インスタンス {annotation_count:>8} 個")

    print("-" * 50)
    print(f"{'合計':>5}: 画像 {total_images:>6} 枚, 文字インスタンス {total_annotations:>8} 個")

    # 各データセットの割合を計算
    print("\n" + "=" * 50)
    print("データセット割合")
    print("=" * 50)

    if total_images > 0:
        for subset in ["train", "val", "test"]:
            if subset in results:
                image_count, annotation_count = results[subset]
                image_ratio = (image_count / total_images) * 100
                annotation_ratio = (annotation_count / total_annotations) * 100 if total_annotations > 0 else 0
                print(f"{subset:>5}: 画像 {image_ratio:>5.1f}%, 文字インスタンス {annotation_ratio:>5.1f}%")


if __name__ == "__main__":
    main()
