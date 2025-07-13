#!/usr/bin/env python3
"""
元画像+4分割+9分割データセット作成スクリプト

このスクリプトは既存のYOLOデータセットから元画像、4分割画像、9分割画像の計14枚を生成してデータ拡張を行います。
- 元画像をそのまま保持
- 元画像を2x2に分割して4枚生成
- 元画像を3x3に分割して9枚生成
- 対応するアノテーションも適切に変換
- YOLO形式を維持
- 1つの元画像から14枚の画像を生成（14倍のデータ拡張）

使用例:
    python scripts/create_quadrant_dataset.py
    python scripts/create_quadrant_dataset.py --visualize --sample_count 5
    python scripts/create_quadrant_dataset.py --output_dir data/custom_output
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.quadrant_processor import MultiGridProcessor


def create_dataset_yaml(output_dir: str):
    """
    出力データセット用のYAML設定ファイルを作成する

    Args:
        output_dir: 出力データセットのディレクトリ
    """
    yaml_content = f"""path: {output_dir}  # データセットのルートディレクトリ
train: train/images  # 訓練画像のディレクトリ
val: val/images      # 検証画像のディレクトリ
test: test/images    # テスト画像のディレクトリ
nc: 1  # 文字位置検出のみ（文字種分類なし）

# クラス名
names:
  0: character  # 文字クラス（種類は問わず、文字の存在のみ）

# データセットの形式
task: detect  # 物体検出タスク

# 元画像+4分割+9分割データセット設定
multi_grid_dataset:
  created_from: data/yolo_dataset_character_detection
  creation_date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
  split_method: original_plus_2x2_plus_3x3_grid
  images_per_original: 14  # 元画像1枚 + 4分割4枚 + 9分割9枚 = 計14枚
  positions:
    - original
    # 4分割 (2x2)
    - quad_top_left
    - quad_top_right
    - quad_bottom_left
    - quad_bottom_right
    # 9分割 (3x3)
    - nine_top_left
    - nine_top_center
    - nine_top_right
    - nine_middle_left
    - nine_middle_center
    - nine_middle_right
    - nine_bottom_left
    - nine_bottom_center
    - nine_bottom_right
"""

    output_path = os.path.join(output_dir, "dataset.yaml")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"データセット設定ファイルを作成しました: {output_path}")


def visualize_sample_splits(input_dir: str, output_dir: str, sample_count: int = 3):
    """
    分割結果のサンプル可視化（元画像+4分割+9分割の計14枚）

    Args:
        input_dir: 入力データセットのディレクトリ
        output_dir: 出力データセットのディレクトリ
        sample_count: 可視化するサンプル数
    """
    print(f"\n分割結果の可視化（{sample_count}サンプル）...")

    # 訓練データから適当なサンプルを選択
    input_images_dir = Path(input_dir) / "train" / "images"
    image_files = list(input_images_dir.glob("*.jpg"))[:sample_count]

    visualization_dir = Path(output_dir) / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(image_files):
        try:
            # 元画像とアノテーション
            original_image = cv2.imread(str(image_path))
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            annotation_path = Path(input_dir) / "train" / "labels" / f"{image_path.stem}.txt"

            # 元画像のアノテーション数を取得
            original_annotation_count = 0
            original_output_annotation_path = Path(output_dir) / "train" / "labels" / f"{image_path.stem}_original.txt"
            if original_output_annotation_path.exists():
                with open(original_output_annotation_path) as f:
                    for line in f:
                        if line.strip():
                            original_annotation_count += 1

            # 4分割画像の読み込み
            quad_images = {}
            quad_positions = ["top_left", "top_right", "bottom_left", "bottom_right"]

            for pos in quad_positions:
                split_image_path = Path(output_dir) / "train" / "images" / f"{image_path.stem}_quad_{pos}.jpg"
                split_annotation_path = Path(output_dir) / "train" / "labels" / f"{image_path.stem}_quad_{pos}.txt"

                if split_image_path.exists():
                    split_image = cv2.imread(str(split_image_path))
                    split_image_rgb = cv2.cvtColor(split_image, cv2.COLOR_BGR2RGB)

                    # アノテーション読み込み
                    annotations = []
                    if split_annotation_path.exists():
                        with open(split_annotation_path) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    annotations.append([float(x) for x in parts[1:]])

                    quad_images[pos] = (split_image_rgb, annotations)

            # 9分割画像の読み込み
            nine_images = {}
            nine_positions = ["top_left", "top_center", "top_right", 
                            "middle_left", "middle_center", "middle_right",
                            "bottom_left", "bottom_center", "bottom_right"]

            for pos in nine_positions:
                split_image_path = Path(output_dir) / "train" / "images" / f"{image_path.stem}_nine_{pos}.jpg"
                split_annotation_path = Path(output_dir) / "train" / "labels" / f"{image_path.stem}_nine_{pos}.txt"

                if split_image_path.exists():
                    split_image = cv2.imread(str(split_image_path))
                    split_image_rgb = cv2.cvtColor(split_image, cv2.COLOR_BGR2RGB)

                    # アノテーション読み込み
                    annotations = []
                    if split_annotation_path.exists():
                        with open(split_annotation_path) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    annotations.append([float(x) for x in parts[1:]])

                    nine_images[pos] = (split_image_rgb, annotations)

            # 可視化プロット作成（5行3列のレイアウト）
            fig, axes = plt.subplots(5, 3, figsize=(15, 25))
            fig.suptitle(f"元画像+4分割+9分割結果例 {i + 1}: {image_path.name}", fontsize=16)

            # 元画像の表示（1行目中央に配置）
            axes[0, 1].imshow(original_image_rgb)
            axes[0, 1].set_title(f"元画像 ({original_annotation_count}個)", fontsize=12)
            axes[0, 1].axis("off")

            # 元画像のアノテーション表示
            if annotation_path.exists():
                h, w = original_image_rgb.shape[:2]
                with open(annotation_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, cx, cy, bw, bh = [float(x) for x in parts]
                            x = (cx - bw / 2) * w
                            y = (cy - bh / 2) * h
                            rect = patches.Rectangle((x, y), bw * w, bh * h, linewidth=1, edgecolor="red", facecolor="none")
                            axes[0, 1].add_patch(rect)

            # 1行目の左右は空白
            axes[0, 0].axis("off")
            axes[0, 2].axis("off")

            # 4分割画像の表示（2行目）
            quad_plot_positions = [
                (1, 0, "top_left", "4分割左上"),
                (1, 1, "top_right", "4分割右上"),
                (1, 2, "bottom_left", "4分割左下"),
                # bottom_rightは3行目の最初に表示
            ]

            for row, col, pos, title in quad_plot_positions:
                if pos in quad_images:
                    image, annotations = quad_images[pos]
                    axes[row, col].imshow(image)
                    axes[row, col].set_title(f"{title} ({len(annotations)}個)", fontsize=10)
                    axes[row, col].axis("off")

                    # アノテーション表示
                    h, w = image.shape[:2]
                    for cx, cy, bw, bh in annotations:
                        x = (cx - bw / 2) * w
                        y = (cy - bh / 2) * h
                        rect = patches.Rectangle((x, y), bw * w, bh * h, linewidth=1, edgecolor="blue", facecolor="none")
                        axes[row, col].add_patch(rect)
                else:
                    axes[row, col].axis("off")
                    axes[row, col].text(0.5, 0.5, "画像なし", ha="center", va="center")

            # bottom_rightの表示（3行目最初）
            if "bottom_right" in quad_images:
                image, annotations = quad_images["bottom_right"]
                axes[2, 0].imshow(image)
                axes[2, 0].set_title(f"4分割右下 ({len(annotations)}個)", fontsize=10)
                axes[2, 0].axis("off")

                # アノテーション表示
                h, w = image.shape[:2]
                for cx, cy, bw, bh in annotations:
                    x = (cx - bw / 2) * w
                    y = (cy - bh / 2) * h
                    rect = patches.Rectangle((x, y), bw * w, bh * h, linewidth=1, edgecolor="blue", facecolor="none")
                    axes[2, 0].add_patch(rect)

            # 9分割画像の表示（3行目〜5行目）
            nine_plot_positions = [
                (2, 1, "top_left", "9分割左上"),
                (2, 2, "top_center", "9分割上中"),
                (3, 0, "top_right", "9分割右上"),
                (3, 1, "middle_left", "9分割左中"),
                (3, 2, "middle_center", "9分割中央"),
                (4, 0, "middle_right", "9分割右中"),
                (4, 1, "bottom_left", "9分割左下"),
                (4, 2, "bottom_center", "9分割下中"),
            ]

            for row, col, pos, title in nine_plot_positions:
                if pos in nine_images:
                    image, annotations = nine_images[pos]
                    axes[row, col].imshow(image)
                    axes[row, col].set_title(f"{title} ({len(annotations)}個)", fontsize=8)
                    axes[row, col].axis("off")

                    # アノテーション表示
                    h, w = image.shape[:2]
                    for cx, cy, bw, bh in annotations:
                        x = (cx - bw / 2) * w
                        y = (cy - bh / 2) * h
                        rect = patches.Rectangle((x, y), bw * w, bh * h, linewidth=1, edgecolor="green", facecolor="none")
                        axes[row, col].add_patch(rect)
                else:
                    axes[row, col].axis("off")

            # bottom_rightの9分割を別途処理
            if "bottom_right" in nine_images:
                # 別のサブプロットに表示（代替として残りのスペースを使用）
                pass

            # 保存
            output_path = visualization_dir / f"sample_{i + 1:02d}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"可視化完了: {output_path}")

        except Exception as e:
            print(f"可視化エラー {image_path.name}: {e}")


def save_processing_stats(stats: dict, output_dir: str):
    """
    処理統計を保存する

    Args:
        stats: 処理統計辞書
        output_dir: 出力ディレクトリ
    """
    stats_file = os.path.join(output_dir, "processing_stats.json")

    # 処理日時を追加
    stats["processing_info"] = {
        "timestamp": datetime.now().isoformat(),
        "total_processed_images": sum(split_stats.get("processed_images", 0) for split_stats in stats.values()),
        "total_generated_images": sum(split_stats.get("processed_images", 0) for split_stats in stats.values())
        * 14,  # 元画像+4分割+9分割=14倍
        "total_annotations": sum(split_stats.get("total_annotations", 0) for split_stats in stats.values()),
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"処理統計を保存しました: {stats_file}")


def print_summary(stats: dict):
    """
    処理結果のサマリーを表示する

    Args:
        stats: 処理統計辞書
    """
    print("\n" + "=" * 60)
    print("         元画像+4分割+9分割データセット作成完了")
    print("=" * 60)

    total_processed = 0
    total_generated = 0
    total_annotations = 0

    for split_name, split_stats in stats.items():
        if split_name == "processing_info":
            continue

        processed = split_stats.get("processed_images", 0)
        generated = processed * 14  # 元画像+4分割+9分割なので14倍
        annotations = split_stats.get("total_annotations", 0)

        total_processed += processed
        total_generated += generated
        total_annotations += annotations

        print(f"{split_name.upper():>8}: {processed:>6} 画像 → {generated:>6} 画像 ({annotations:>7} アノテーション)")

    print("-" * 60)
    print(f"{'合計':>8}: {total_processed:>6} 画像 → {total_generated:>6} 画像 ({total_annotations:>7} アノテーション)")
    print(f"拡張率: {total_generated / total_processed:.1f}倍" if total_processed > 0 else "拡張率: N/A")
    print("=" * 60)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="YOLOデータセットの元画像+4分割+9分割によるデータ拡張（14倍化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python scripts/create_quadrant_dataset.py
  python scripts/create_quadrant_dataset.py --visualize --sample_count 5
  python scripts/create_quadrant_dataset.py --output_dir data/custom_multi_grid_dataset
        """,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/yolo_dataset_character_detection",
        help="入力データセットのディレクトリ (デフォルト: data/yolo_dataset_character_detection)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/yolo_dataset_character_detection_quadrant",
        help="出力データセットのディレクトリ (デフォルト: data/yolo_dataset_character_detection_quadrant)",
    )

    parser.add_argument(
        "--overlap_threshold", type=float, default=0.5, help="バウンディングボックスの重複判定閾値 (デフォルト: 0.5)"
    )

    parser.add_argument("--visualize", action="store_true", help="分割結果の可視化を行う")

    parser.add_argument("--sample_count", type=int, default=3, help="可視化するサンプル数 (デフォルト: 3)")

    args = parser.parse_args()

    # 入力ディレクトリの存在確認
    if not os.path.exists(args.input_dir):
        print(f"エラー: 入力ディレクトリが存在しません: {args.input_dir}")
        sys.exit(1)

    print("元画像+4分割+9分割データセット作成を開始します...")
    print(f"入力ディレクトリ: {args.input_dir}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print(f"重複判定閾値: {args.overlap_threshold}")
    print("※ 1つの元画像から14枚の画像（元画像1枚+4分割4枚+9分割9枚）を生成します")

    # プロセッサー初期化
    processor = MultiGridProcessor(
        input_dir=args.input_dir, output_dir=args.output_dir, overlap_threshold=args.overlap_threshold
    )

    try:
        # データセット処理の実行
        stats = processor.process_full_dataset()

        # 設定ファイルの作成
        create_dataset_yaml(args.output_dir)

        # 処理統計の保存
        save_processing_stats(stats, args.output_dir)

        # 結果サマリーの表示
        print_summary(stats)

        # 可視化の実行（オプション）
        if args.visualize:
            visualize_sample_splits(args.input_dir, args.output_dir, args.sample_count)

        print("\n元画像+4分割+9分割データセット作成が正常に完了しました！")
        print(f"出力先: {args.output_dir}")
        print("各元画像から14枚の画像（元画像1枚+4分割4枚+9分割9枚）が生成されました。")

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
