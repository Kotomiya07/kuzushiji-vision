"""テスト用の小規模データセットを作成するスクリプト."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tqdm import tqdm


def create_small_test_dataset(
    source_dir: Path,
    target_dir: Path,
    num_images: int = 100,
    splits: list[str] | None = None,
) -> None:
    """小規模なテスト用データセットを作成.

    Args:
        source_dir: 元のデータセットディレクトリ
        target_dir: 出力先ディレクトリ
        num_images: 各分割から抽出する画像数
        splits: データ分割のリスト
    """
    if splits is None:
        splits = ["train", "val", "test"]

    target_dir.mkdir(parents=True, exist_ok=True)

    # dataset.yamlをコピー
    source_yaml = source_dir / "dataset.yaml"
    if source_yaml.exists():
        shutil.copy(source_yaml, target_dir / "dataset.yaml")

    for split in splits:
        print(f"\n=== {split} データセットを処理中 ===")

        source_images_dir = source_dir / split / "images"
        source_labels_dir = source_dir / split / "labels"
        source_coco_json = source_dir / split / "_annotations.coco.json"

        if not source_images_dir.exists():
            print(f"警告: {source_images_dir} が見つかりません。スキップします。")
            continue

        # 出力ディレクトリを作成
        target_images_dir = target_dir / split / "images"
        target_labels_dir = target_dir / split / "labels"
        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_labels_dir.mkdir(parents=True, exist_ok=True)

        # 元のCOCO JSONを読み込み
        with source_coco_json.open(encoding="utf-8") as f:
            coco_data = json.load(f)

        # 最初のN枚の画像を選択
        selected_images = coco_data["images"][:num_images]
        selected_image_ids = {img["id"] for img in selected_images}

        print(f"選択された画像数: {len(selected_images)}")

        # 画像ファイルをコピー
        for img_info in tqdm(selected_images, desc="画像をコピー中"):
            img_filename = img_info["file_name"]
            source_img = source_images_dir / img_filename
            target_img = target_images_dir / img_filename

            if source_img.exists():
                shutil.copy(source_img, target_img)

            # ラベルファイルもコピー
            label_filename = Path(img_filename).stem + ".txt"
            source_label = source_labels_dir / label_filename
            target_label = target_labels_dir / label_filename

            if source_label.exists():
                shutil.copy(source_label, target_label)

        # 選択された画像のアノテーションのみを抽出
        selected_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in selected_image_ids]

        print(f"選択されたアノテーション数: {len(selected_annotations)}")

        # 新しいCOCO JSONを作成
        new_coco_data = {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "images": selected_images,
            "annotations": selected_annotations,
            "categories": coco_data.get("categories", []),
        }

        # COCO JSONを保存
        target_coco_json = target_dir / split / "_annotations.coco.json"
        with target_coco_json.open("w", encoding="utf-8") as f:
            json.dump(new_coco_data, f, indent=2, ensure_ascii=False)

        print(f"✓ {split} データセットを作成しました")
        print(f"  画像: {len(selected_images)}")
        print(f"  アノテーション: {len(selected_annotations)}")


def main() -> None:
    source_dir = Path("data/yolo_dataset_character_detection_multi_grid")
    target_dir = Path("data/yolo_dataset_character_detection_small_test")

    print("=== 小規模テストデータセット作成 ===")
    print(f"元データ: {source_dir}")
    print(f"出力先: {target_dir}")

    create_small_test_dataset(
        source_dir=source_dir,
        target_dir=target_dir,
        num_images=100,  # 各分割100枚
        splits=["train", "val", "test"],  # trainとvalのみ作成（testは学習に不要）
    )

    print("\n✓ 完了！")
    print(f"出力先: {target_dir}")


if __name__ == "__main__":
    main()
