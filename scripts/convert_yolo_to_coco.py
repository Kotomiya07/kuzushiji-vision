"""YOLO形式データセットをCOCO形式に変換するスクリプト.

YOLO形式:
- ラベルファイル: class x_center y_center width height (正規化座標)
- 画像とラベルが別ディレクトリ

COCO形式:
- JSON形式でアノテーション、画像、カテゴリ情報を統合
- bbox: [x, y, width, height] (ピクセル座標、左上基準)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


def load_yolo_labels(label_path: Path) -> list[dict[str, float]]:
    """YOLOラベルファイルを読み込む.

    Args:
        label_path: ラベルファイルのパス

    Returns:
        ラベル情報のリスト [{"class": int, "x_center": float, "y_center": float, "width": float, "height": float}]
    """
    labels = []
    if not label_path.exists():
        return labels

    with label_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            labels.append({
                "class": int(parts[0]),
                "x_center": float(parts[1]),
                "y_center": float(parts[2]),
                "width": float(parts[3]),
                "height": float(parts[4]),
            })
    return labels


def yolo_to_coco_bbox(yolo_bbox: dict[str, float], img_width: int, img_height: int) -> list[float]:
    """YOLO形式のbboxをCOCO形式に変換.

    Args:
        yolo_bbox: YOLO形式のbbox (正規化座標、中心基準)
        img_width: 画像の幅
        img_height: 画像の高さ

    Returns:
        COCO形式のbbox [x, y, width, height] (ピクセル座標、左上基準)
    """
    # 正規化座標をピクセル座標に変換
    x_center = yolo_bbox["x_center"] * img_width
    y_center = yolo_bbox["y_center"] * img_height
    width = yolo_bbox["width"] * img_width
    height = yolo_bbox["height"] * img_height

    # 中心座標を左上座標に変換
    x = x_center - width / 2
    y = y_center - height / 2

    return [x, y, width, height]


def convert_yolo_to_coco(
    dataset_dir: Path,
    split: str = "train",
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """YOLO形式データセットをCOCO形式に変換.

    Args:
        dataset_dir: データセットのルートディレクトリ
        split: データセット分割 ("train", "val", "test")
        class_names: クラス名のリスト（Noneの場合はdataset.yamlから読み込み）

    Returns:
        COCO形式のアノテーション辞書
    """
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"画像ディレクトリが見つかりません: {images_dir}")

    # クラス名の読み込み
    if class_names is None:
        import yaml

        dataset_yaml = dataset_dir / "dataset.yaml"
        if dataset_yaml.exists():
            with dataset_yaml.open(encoding="utf-8") as f:
                config = yaml.safe_load(f)
                class_names = [config["names"][i] for i in range(config["nc"])]
        else:
            class_names = ["character"]  # デフォルト

    # COCO形式の基本構造
    coco_data: dict[str, Any] = {
        "info": {
            "description": f"Kuzushiji Character Detection Dataset - {split}",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Kuzushiji Vision Lightning",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name, "supercategory": "character"} for i, name in enumerate(class_names)],
    }

    # 画像ファイルを取得
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    annotation_id = 1
    print(f"Converting {split} dataset...")

    for image_id, image_path in enumerate(tqdm(image_files, desc=f"Processing {split}")):
        # 画像情報を取得
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"警告: 画像を開けませんでした: {image_path} - {e}")
            continue

        # 画像情報を追加
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "width": img_width,
            "height": img_height,
        })

        # ラベルファイルを読み込み
        label_path = labels_dir / f"{image_path.stem}.txt"
        yolo_labels = load_yolo_labels(label_path)

        # アノテーションを追加
        for yolo_label in yolo_labels:
            coco_bbox = yolo_to_coco_bbox(yolo_label, img_width, img_height)
            area = coco_bbox[2] * coco_bbox[3]  # width * height

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": yolo_label["class"],
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0,
            })
            annotation_id += 1

    print(f"変換完了: {len(coco_data['images'])} 画像, {len(coco_data['annotations'])} アノテーション")
    return coco_data


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO形式データセットをCOCO形式に変換")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/yolo_dataset_character_detection_multi_grid",
        help="YOLOデータセットのルートディレクトリ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="COCO形式JSONの出力ディレクトリ（Noneの場合はdataset_dir/cocoに保存）",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="変換するデータ分割（train, val, test）",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=None,
        help="クラス名のリスト（指定しない場合はdataset.yamlから読み込み）",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"データセットディレクトリが見つかりません: {dataset_dir}")

    # 出力ディレクトリの設定
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = dataset_dir / "coco"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"データセット: {dataset_dir}")
    print(f"出力先: {output_dir}")
    print(f"変換対象: {', '.join(args.splits)}")
    print()

    # 各分割を変換
    for split in args.splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"警告: {split} ディレクトリが見つかりません。スキップします。")
            continue

        try:
            coco_data = convert_yolo_to_coco(
                dataset_dir=dataset_dir,
                split=split,
                class_names=args.class_names,
            )

            # JSONファイルに保存
            output_path = output_dir / f"instances_{split}.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=False)

            print(f"✓ {split} データを保存しました: {output_path}")
            print()

        except Exception as e:
            print(f"✗ {split} データの変換に失敗しました: {e}")
            import traceback

            traceback.print_exc()
            print()

    print("すべての変換が完了しました！")


if __name__ == "__main__":
    main()
