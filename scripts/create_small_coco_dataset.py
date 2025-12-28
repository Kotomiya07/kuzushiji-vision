"""テスト用に小規模なCOCOデータセットを作成するスクリプト."""

from __future__ import annotations

import json
from pathlib import Path


def create_small_dataset(
    input_json: Path,
    output_json: Path,
    max_images: int = 100,
) -> None:
    """小規模なテスト用データセットを作成.

    Args:
        input_json: 元のCOCO JSONファイル
        output_json: 出力先のCOCO JSONファイル
        max_images: 最大画像数
    """
    print(f"Loading {input_json}...")
    with input_json.open(encoding="utf-8") as f:
        data = json.load(f)

    # 最初のN枚の画像とそのアノテーションのみ取得
    small_images = data["images"][:max_images]
    image_ids = {img["id"] for img in small_images}

    small_annotations = [ann for ann in data["annotations"] if ann["image_id"] in image_ids]

    small_data = {
        "info": data["info"],
        "licenses": data.get("licenses", []),
        "images": small_images,
        "annotations": small_annotations,
        "categories": data["categories"],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(small_data, f, indent=2, ensure_ascii=False)

    print(f"✓ 作成完了: {output_json}")
    print(f"  画像数: {len(small_images)}")
    print(f"  アノテーション数: {len(small_annotations)}")


def main() -> None:
    base_dir = Path("data/yolo_dataset_character_detection_multi_grid")
    coco_dir = base_dir / "coco"
    small_dir = base_dir / "coco_small"

    # 各分割で小規模データセットを作成
    for split in ["train", "val", "test"]:
        input_json = coco_dir / f"instances_{split}.json"
        output_json = small_dir / f"instances_{split}.json"

        if not input_json.exists():
            print(f"警告: {input_json} が見つかりません。スキップします。")
            continue

        max_images = 100 if split == "train" else 50
        create_small_dataset(input_json, output_json, max_images)
        print()

    print("すべての小規模データセットを作成しました！")


if __name__ == "__main__":
    main()
