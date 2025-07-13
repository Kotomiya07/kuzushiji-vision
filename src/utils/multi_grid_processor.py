"""
画像分割とアノテーション変換処理のユーティリティ

このモジュールは以下の機能を提供します：
- 画像の4分割処理（2x2グリッド）
- 画像の9分割処理（3x3グリッド）
- YOLOアノテーションの座標変換
- 境界にかかる文字の適切な処理
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np


class MultiGridProcessor:
    """画像の4分割・9分割とアノテーション変換を行うクラス"""

    def __init__(self, input_dir: str, output_dir: str, overlap_threshold: float = 0.5):
        """
        Args:
            input_dir: 入力データセットのルートディレクトリ
            output_dir: 出力データセットのルートディレクトリ
            overlap_threshold: バウンディングボックスの重複判定閾値（0.5推奨）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.overlap_threshold = overlap_threshold

        # 4分割位置の定義（2x2グリッド）
        self.quadrants = {
            "top_left": (0, 0, 0.5, 0.5),  # (x1, y1, x2, y2) in normalized coordinates
            "top_right": (0.5, 0, 1.0, 0.5),
            "bottom_left": (0, 0.5, 0.5, 1.0),
            "bottom_right": (0.5, 0.5, 1.0, 1.0),
        }

        # 9分割位置の定義（3x3グリッド）
        self.ninegrids = {
            "top_left": (0, 0, 1/3, 1/3),
            "top_center": (1/3, 0, 2/3, 1/3),
            "top_right": (2/3, 0, 1.0, 1/3),
            "middle_left": (0, 1/3, 1/3, 2/3),
            "middle_center": (1/3, 1/3, 2/3, 2/3),
            "middle_right": (2/3, 1/3, 1.0, 2/3),
            "bottom_left": (0, 2/3, 1/3, 1.0),
            "bottom_center": (1/3, 2/3, 2/3, 1.0),
            "bottom_right": (2/3, 2/3, 1.0, 1.0),
        }

        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def split_image(self, image_path: str, grid_type: str = "quad") -> dict[str, np.ndarray]:
        """
        画像を4分割または9分割する

        Args:
            image_path: 分割する画像のパス
            grid_type: 分割タイプ ("quad" for 4分割, "nine" for 9分割)

        Returns:
            Dict[str, np.ndarray]: 分割された画像の辞書 {position: image_array}
        """
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めませんでした: {image_path}")

        h, w = image.shape[:2]
        split_images = {}

        # 分割タイプに応じて使用する分割定義を選択
        if grid_type == "nine":
            grids = self.ninegrids
        else:
            grids = self.quadrants

        for position, (x1, y1, x2, y2) in grids.items():
            # 正規化座標をピクセル座標に変換
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)

            # 画像分割
            split_images[position] = image[py1:py2, px1:px2]

        return split_images

    def convert_annotations(self, annotation_path: str, grid_position: str, grid_type: str = "quad") -> list[str]:
        """
        アノテーションを分割後の画像に対応する座標に変換する

        Args:
            annotation_path: 元のアノテーションファイルのパス
            grid_position: 分割位置 (quad: top_left, top_right, bottom_left, bottom_right)
                                    (nine: top_left, top_center, top_right, middle_left, middle_center, 
                                           middle_right, bottom_left, bottom_center, bottom_right)
            grid_type: 分割タイプ ("quad" for 4分割, "nine" for 9分割)

        Returns:
            List[str]: 変換後のアノテーション行のリスト
        """
        if not os.path.exists(annotation_path):
            return []

        # 分割タイプに応じて使用する分割定義を選択
        if grid_type == "nine":
            grids = self.ninegrids
        else:
            grids = self.quadrants

        # 分割エリアの正規化座標
        area_x1, area_y1, area_x2, area_y2 = grids[grid_position]
        area_w = area_x2 - area_x1
        area_h = area_y2 - area_y1

        converted_annotations = []

        with open(annotation_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    continue

                class_id = parts[0]
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # バウンディングボックスの範囲計算
                bbox_x1 = center_x - width / 2
                bbox_y1 = center_y - height / 2
                bbox_x2 = center_x + width / 2
                bbox_y2 = center_y + height / 2

                # 分割エリアとの重複判定
                overlap_area = self._calculate_overlap(bbox_x1, bbox_y1, bbox_x2, bbox_y2, area_x1, area_y1, area_x2, area_y2)

                # バウンディングボックスの面積
                bbox_area = width * height

                # 重複率が閾値以上の場合のみ含める
                if overlap_area / bbox_area >= self.overlap_threshold:
                    # 座標を分割エリア内の相対座標に変換
                    new_center_x = (center_x - area_x1) / area_w
                    new_center_y = (center_y - area_y1) / area_h
                    new_width = width / area_w
                    new_height = height / area_h

                    # 座標が範囲内に収まるように調整
                    new_center_x = max(0, min(1, new_center_x))
                    new_center_y = max(0, min(1, new_center_y))
                    new_width = min(new_width, 2 * new_center_x, 2 * (1 - new_center_x))
                    new_height = min(new_height, 2 * new_center_y, 2 * (1 - new_center_y))

                    # 有効なサイズのバウンディングボックスのみ保持
                    if new_width > 0 and new_height > 0:
                        converted_line = f"{class_id} {new_center_x:.6f} {new_center_y:.6f} {new_width:.6f} {new_height:.6f}"
                        converted_annotations.append(converted_line)

        return converted_annotations

    def _calculate_overlap(
        self,
        bbox_x1: float,
        bbox_y1: float,
        bbox_x2: float,
        bbox_y2: float,
        area_x1: float,
        area_y1: float,
        area_x2: float,
        area_y2: float,
    ) -> float:
        """
        2つの矩形の重複面積を計算する

        Args:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2: バウンディングボックスの座標
            area_x1, area_y1, area_x2, area_y2: 分割エリアの座標

        Returns:
            float: 重複面積
        """
        # 重複する矩形の座標計算
        overlap_x1 = max(bbox_x1, area_x1)
        overlap_y1 = max(bbox_y1, area_y1)
        overlap_x2 = min(bbox_x2, area_x2)
        overlap_y2 = min(bbox_y2, area_y2)

        # 重複がない場合
        if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
            return 0.0

        # 重複面積を計算
        return (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)

    def process_single_image(
        self, image_path: str, annotation_path: str, output_images_dir: str, output_labels_dir: str
    ) -> dict[str, int]:
        """
        単一の画像とアノテーションを元画像+4分割+9分割（計14枚）で処理する

        Args:
            image_path: 画像ファイルのパス
            annotation_path: アノテーションファイルのパス
            output_images_dir: 出力画像ディレクトリ
            output_labels_dir: 出力アノテーションディレクトリ

        Returns:
            Dict[str, int]: 各位置（original + 4分割 + 9分割）でのアノテーション数
        """
        # 出力ディレクトリの作成
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        # 画像ファイル名（拡張子なし）
        image_stem = Path(image_path).stem

        annotation_counts = {}

        # 1. 元画像をそのままコピー
        original_output_path = os.path.join(output_images_dir, f"{image_stem}_original.jpg")
        original_image = cv2.imread(image_path)
        cv2.imwrite(original_output_path, original_image)

        # 元画像のアノテーションをそのままコピー
        original_annotation_count = 0
        original_annotation_output = os.path.join(output_labels_dir, f"{image_stem}_original.txt")
        if os.path.exists(annotation_path):
            with open(annotation_path, encoding="utf-8") as src_f:
                with open(original_annotation_output, "w", encoding="utf-8") as dst_f:
                    for line in src_f:
                        if line.strip():
                            dst_f.write(line)
                            original_annotation_count += 1
        else:
            # アノテーションファイルが存在しない場合は空ファイルを作成
            with open(original_annotation_output, "w", encoding="utf-8") as f:
                pass

        annotation_counts["original"] = original_annotation_count

        # 2. 画像を4分割
        split_images_quad = self.split_image(image_path, "quad")

        for position, image_array in split_images_quad.items():
            # 分割画像の保存
            output_image_path = os.path.join(output_images_dir, f"{image_stem}_quad_{position}.jpg")
            cv2.imwrite(output_image_path, image_array)

            # アノテーション変換
            converted_annotations = self.convert_annotations(annotation_path, position, "quad")
            annotation_counts[f"quad_{position}"] = len(converted_annotations)

            # アノテーションファイルの保存
            output_annotation_path = os.path.join(output_labels_dir, f"{image_stem}_quad_{position}.txt")
            with open(output_annotation_path, "w", encoding="utf-8") as f:
                for annotation in converted_annotations:
                    f.write(annotation + "\n")

        # 3. 画像を9分割
        split_images_nine = self.split_image(image_path, "nine")

        for position, image_array in split_images_nine.items():
            # 分割画像の保存
            output_image_path = os.path.join(output_images_dir, f"{image_stem}_nine_{position}.jpg")
            cv2.imwrite(output_image_path, image_array)

            # アノテーション変換
            converted_annotations = self.convert_annotations(annotation_path, position, "nine")
            annotation_counts[f"nine_{position}"] = len(converted_annotations)

            # アノテーションファイルの保存
            output_annotation_path = os.path.join(output_labels_dir, f"{image_stem}_nine_{position}.txt")
            with open(output_annotation_path, "w", encoding="utf-8") as f:
                for annotation in converted_annotations:
                    f.write(annotation + "\n")

        return annotation_counts

    def process_dataset_split(self, split_name: str) -> dict[str, int]:
        """
        データセット分割（train/val/test）を処理する

        Args:
            split_name: 分割名 (train, val, test)

        Returns:
            Dict[str, int]: 処理統計
        """
        input_split_dir = self.input_dir / split_name
        output_split_dir = self.output_dir / split_name

        input_images_dir = input_split_dir / "images"
        input_labels_dir = input_split_dir / "labels"
        output_images_dir = output_split_dir / "images"
        output_labels_dir = output_split_dir / "labels"

        if not input_images_dir.exists():
            self.logger.warning(f"画像ディレクトリが存在しません: {input_images_dir}")
            return {"processed_images": 0, "total_annotations": 0}

        # 画像ファイルのリスト取得
        image_files = list(input_images_dir.glob("*.jpg")) + list(input_images_dir.glob("*.png"))

        processed_images = 0
        total_annotations = 0

        for image_path in image_files:
            try:
                # 対応するアノテーションファイルのパス
                annotation_path = input_labels_dir / f"{image_path.stem}.txt"

                # 単一画像の処理
                annotation_counts = self.process_single_image(
                    str(image_path), str(annotation_path), str(output_images_dir), str(output_labels_dir)
                )

                processed_images += 1
                total_annotations += sum(annotation_counts.values())

                if processed_images % 100 == 0:
                    self.logger.info(f"{split_name}: {processed_images}/{len(image_files)} 画像処理完了")

            except Exception as e:
                self.logger.error(f"画像処理エラー {image_path}: {e}")
                continue

        stats = {"processed_images": processed_images, "total_images": len(image_files), "total_annotations": total_annotations}

        self.logger.info(f"{split_name} 処理完了: {stats}")
        return stats

    def process_full_dataset(self) -> dict[str, dict[str, int]]:
        """
        完全なデータセットを処理する

        Returns:
            Dict[str, Dict[str, int]]: 分割ごとの処理統計
        """
        splits = ["train", "val", "test"]
        all_stats = {}

        for split in splits:
            self.logger.info(f"{split} データセットの処理開始...")
            all_stats[split] = self.process_dataset_split(split)

        return all_stats
