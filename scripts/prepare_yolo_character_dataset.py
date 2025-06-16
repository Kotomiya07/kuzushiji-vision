"""
YOLOフォーマットのデータセットを準備するスクリプト（1文字位置検出用）
1ページ画像と文字アノテーションを用いてYOLOフォーマットのデータセットを準備する。
1文字ごとの位置検出を行うためのデータセットを作成する（文字種分類は行わない）。

データ構造:
- 1ページ画像: data/raw/dataset/{book_id}/images/{image_name}.jpg
- アノテーション: data/raw/dataset/{book_id}/{book_id}_coordinate.csv
- 文字位置: 各座標CSVファイルから文字のバウンディングボックス情報を取得
"""

import shutil
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.utils.util import EasyDict


def convert_to_yolo_format(box, image_width, image_height):
    """バウンディングボックスをYOLOフォーマットに変換"""
    try:
        x1, y1, x2, y2 = map(float, box)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid box format: {box}. Expected [x1, y1, x2, y2]. Error: {e}")

    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Image dimensions must be positive. Got width={image_width}, height={image_height}")

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    # 座標値を[0,1]の範囲内にクリップ
    x_center = np.clip(x_center, 0.0, 1.0)
    y_center = np.clip(y_center, 0.0, 1.0)
    width = np.clip(width, 0.0, 1.0)
    height = np.clip(height, 0.0, 1.0)

    return [x_center, y_center, width, height]


def get_image_size(image_path: Path) -> tuple[int, int]:
    """画像サイズを取得"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions (<=0) in {image_path}: ({width}, {height})")
            return width, height
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Failed to load or process image: {image_path}. Error: {e}")


def safe_literal_eval(val):
    """安全にリテラルを評価する"""
    if isinstance(val, list | tuple):
        return val
    if isinstance(val, str):
        try:
            stripped_val = val.strip()
            if stripped_val.startswith("[") and stripped_val.endswith("]"):
                evaluated = literal_eval(stripped_val)
                if isinstance(evaluated, list):
                    return evaluated
                else:
                    raise ValueError("Evaluated result is not a list.")
            else:
                raise ValueError(f"String '{val}' is not a valid list format.")
        except (ValueError, SyntaxError, TypeError, MemoryError) as e:
            raise ValueError(f"Failed to safely evaluate string '{val}' as literal list. Error: {e}")
    raise TypeError(f"Input must be a string representation of a list, or a list/tuple, but got {type(val)}")


def prepare_yolo_character_dataset(
    dataset_dir: str = "data/raw/dataset",
    output_dir: str = "data/yolo_dataset_character_detection",
    train_docs_count: int = 31,
    val_docs_count: int = 7,
    test_docs_count: int = 6,
    random_state: int = 42,
    class_id: int = 0,
):
    """
    1ページ画像と文字アノテーションを用いてYOLOフォーマットのデータセットを準備（1文字位置検出用）
    画像とアノテーションは data/raw/dataset の各書籍ディレクトリから取得する
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)

    if not dataset_path.exists():
        print(f"Error: Dataset directory not found at {dataset_path}")
        return

    # 出力ディレクトリの準備 (train, val, test, no_use)
    for split in ["train", "val", "test", "no_use"]:
        for subdir in ["images", "labels"]:
            (output_path / split / subdir).mkdir(parents=True, exist_ok=True)

    # 書籍ディレクトリを走査してアノテーションを収集
    print("Scanning dataset directory for books and annotations...")
    all_annotations = []
    available_book_ids = []

    for book_dir in dataset_path.iterdir():
        if not book_dir.is_dir():
            continue

        book_id = book_dir.name
        coordinate_file = book_dir / f"{book_id}_coordinate.csv"
        images_dir = book_dir / "images"

        # 必要なファイル・ディレクトリの存在確認
        if not coordinate_file.exists():
            print(f"Warning: Coordinate file not found for book {book_id}: {coordinate_file}")
            continue
        if not images_dir.exists():
            print(f"Warning: Images directory not found for book {book_id}: {images_dir}")
            continue

        try:
            # 座標ファイルを読み込み
            coord_df = pd.read_csv(coordinate_file)

            # 必要な列の確認（座標ファイルの形式に応じて調整が必要）
            if coord_df.empty:
                print(f"Warning: Empty coordinate file for book {book_id}")
                continue

            # 書籍IDと画像パス情報を追加
            coord_df["book_id"] = book_id
            coord_df["images_dir"] = str(images_dir)

            all_annotations.append(coord_df)
            available_book_ids.append(book_id)
            print(f"Loaded {len(coord_df)} annotations from book {book_id}")

        except Exception as e:
            print(f"Error reading coordinate file for book {book_id}: {e}")
            continue

    if not all_annotations:
        print("Error: No valid annotations found in any book directory")
        return

    # 全アノテーションを統合
    df = pd.concat(all_annotations, ignore_index=True)
    print(f"Total annotations loaded: {len(df)} from {len(available_book_ids)} books")

    # 利用可能な書籍IDリストを使用
    unique_doc_ids = available_book_ids

    # --- Split logic: use existing split info if present ---
    split_info_path = output_path / "dataset_split_info.csv"
    doc_id_splits = {"train": set(), "val": set(), "test": set(), "no_use": set()}
    use_existing_split = False
    if split_info_path.exists():
        print(f"Using existing split info from {split_info_path}.")
        try:
            split_df_info = pd.read_csv(split_info_path)
            if not all(col in split_df_info.columns for col in ["doc_id", "split"]):
                print(f"Error: {split_info_path} does not contain required columns ['doc_id', 'split']. Ignoring file.")
            else:
                for split in ["train", "val", "test", "no_use"]:
                    doc_id_splits[split] = set(split_df_info[split_df_info["split"] == split]["doc_id"].astype(str))
                use_existing_split = True
                print(f"Using existing split info from {split_info_path}. Ignoring train/val/test_docs_count arguments.")
        except Exception as e:
            print(f"Warning: Failed to load split info from {split_info_path}. Error: {e}. Proceeding with random split.")
            use_existing_split = False

    if not use_existing_split:
        total_docs = len(unique_doc_ids)
        print(f"Found {total_docs} unique document IDs with existing images.")

        # 指定された分割数と実際のドキュメント数を確認
        requested_total = train_docs_count + val_docs_count + test_docs_count
        if total_docs < requested_total:
            print(
                f"Error: Not enough unique documents ({total_docs}) to fulfill the requested split "
                f"({train_docs_count} train, {val_docs_count} val, {test_docs_count} test)."
            )
            return
        elif total_docs > requested_total:
            print(
                f"Warning: Found {total_docs} documents, but only {requested_total} are requested for the split. "
                f"{total_docs - requested_total} documents will be unused."
            )

        # ドキュメントIDを train, val, test に分割
        train_doc_ids, temp_doc_ids = train_test_split(
            unique_doc_ids,
            test_size=(val_docs_count + test_docs_count),
            train_size=train_docs_count,
            random_state=random_state,
        )

        if val_docs_count + test_docs_count > 0 and len(temp_doc_ids) > 0:
            test_proportion_in_temp = test_docs_count / (val_docs_count + test_docs_count)
            if test_proportion_in_temp == 1.0:
                val_doc_ids = np.array([])
                test_doc_ids = temp_doc_ids
            elif test_proportion_in_temp == 0.0:
                val_doc_ids = temp_doc_ids
                test_doc_ids = np.array([])
            else:
                val_doc_ids, test_doc_ids = train_test_split(
                    temp_doc_ids, test_size=test_proportion_in_temp, random_state=random_state
                )
        else:
            val_doc_ids = temp_doc_ids if val_docs_count > 0 else np.array([])
            test_doc_ids = temp_doc_ids if test_docs_count > 0 and val_docs_count == 0 else np.array([])

        print(f"Actual split: Train={len(train_doc_ids)}, Val={len(val_doc_ids)}, Test={len(test_doc_ids)}")

        doc_id_splits = {"train": set(train_doc_ids), "val": set(val_doc_ids), "test": set(test_doc_ids)}

        # --- 書籍IDの分割情報をCSVに保存 ---
        split_records = []
        for split_name, ids in doc_id_splits.items():
            for doc_id in ids:
                split_records.append(EasyDict({"doc_id": doc_id, "split": split_name}))

        split_df_info = pd.DataFrame(split_records)
        try:
            split_df_info.to_csv(split_info_path, index=False, encoding="utf-8")
            print(f"Saved dataset split information to: {split_info_path}")
        except Exception as e:
            print(f"Warning: Failed to save dataset split information to {split_info_path}. Error: {e}")

    # データセット準備ループ
    processed_count = EasyDict({"train": 0, "val": 0, "test": 0, "no_use": 0})
    skipped_count = EasyDict({"train": 0, "val": 0, "test": 0, "no_use": 0})

    for split, target_doc_ids in doc_id_splits.items():
        if not target_doc_ids:
            print(f"\nSkipping {split} dataset as no documents were assigned.")
            continue

        print(f"\nPreparing {split} dataset...")
        split_output_images = output_path / split / "images"
        split_output_labels = output_path / split / "labels"

        # このsplitに含まれる書籍をフィルタリング
        split_df = df[df["book_id"].isin(target_doc_ids)]

        print(f"Processing images from {len(target_doc_ids)} books for {split} split.")

        # --- 書籍ごとに処理 ---
        for book_id in target_doc_ids:
            book_annotations = split_df[split_df["book_id"] == book_id]
            if book_annotations.empty:
                continue

            # 書籍の画像ディレクトリを取得
            images_dir = Path(book_annotations.iloc[0]["images_dir"])

            # 画像ファイルをリストアップ
            if not images_dir.exists():
                print(f"\nWarning: Images directory not found for book {book_id}: {images_dir}")
                continue

            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            print(f"\nProcessing {len(image_files)} images from book {book_id}")

            for image_path in image_files:
                image_name = image_path.name
                dst_image_path = split_output_images / f"{book_id}_{image_name}"

                # 画像コピー
                try:
                    shutil.copy2(image_path, dst_image_path)
                except Exception as e:
                    print(f"\nWarning: Failed to copy {image_path} to {dst_image_path}. Error: {e}")
                    skipped_count[split] += 1
                    continue

                # 画像サイズ取得
                try:
                    image_width, image_height = get_image_size(image_path)
                except (FileNotFoundError, ValueError) as e:
                    print(f"\nWarning: Skipping {image_path} due to error getting size: {e}")
                    skipped_count[split] += 1
                    dst_image_path.unlink(missing_ok=True)
                    continue

                # この画像に対応する文字アノテーションを取得
                image_stem = image_path.stem

                # 座標ファイルの形式を推測して適切なフィルタリングを実行
                # Image列には拡張子なしの画像名が記録されている
                image_annotations = book_annotations[book_annotations["Image"] == image_stem]

                # 古い形式のフォールバック
                if image_annotations.empty and "image" in book_annotations.columns:
                    image_annotations = book_annotations[book_annotations["image"] == image_stem]
                if image_annotations.empty and "filename" in book_annotations.columns:
                    image_annotations = book_annotations[book_annotations["filename"] == image_name]

                label_name = f"{book_id}_{image_stem}.txt"
                label_path = split_output_labels / label_name
                annotations_written = 0

                try:
                    with open(label_path, "w", encoding="utf-8") as f:
                        for _, row in image_annotations.iterrows():
                            try:
                                # 座標情報を取得（実際のCSVの形式: X,Y,Width,Height）
                                if "X" in row and "Y" in row and "Width" in row and "Height" in row:
                                    # X, Y, Width, Height形式（実際のデータ形式）
                                    x, y, w, h = row["X"], row["Y"], row["Width"], row["Height"]
                                    box = [x, y, x + w, y + h]  # [x1, y1, x2, y2]形式に変換
                                elif "x1" in row and "y1" in row and "x2" in row and "y2" in row:
                                    # x1, y1, x2, y2形式
                                    box = [row["x1"], row["y1"], row["x2"], row["y2"]]
                                else:
                                    print(f"\nWarning: Unknown coordinate format in row for {image_name}")
                                    continue

                                # 座標値の検証
                                if not all(isinstance(coord, (int, float)) for coord in box):
                                    print(f"\nWarning: Non-numeric coordinates for {image_name}: {box}")
                                    continue

                                # 座標値が負の値や画像サイズを超えている場合の検証
                                x1, y1, x2, y2 = box
                                if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
                                    print(
                                        f"\nWarning: Coordinates out of bounds for {image_name}: {box}, image size: {image_width}x{image_height}"
                                    )
                                    # 座標をクリップして続行
                                    x1 = max(0, x1)
                                    y1 = max(0, y1)
                                    x2 = min(image_width, x2)
                                    y2 = min(image_height, y2)
                                    box = [x1, y1, x2, y2]

                                # ボックスの幅と高さが有効かチェック
                                if x2 <= x1 or y2 <= y1:
                                    print(f"\nWarning: Invalid box dimensions for {image_name}: {box}")
                                    continue

                                # YOLOフォーマットに変換
                                yolo_box = convert_to_yolo_format(box, image_width=image_width, image_height=image_height)
                                f.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")
                                annotations_written += 1

                            except (ValueError, TypeError, KeyError) as e:
                                print(f"\nWarning: Skipping invalid annotation for {image_name}: {e}")
                                continue

                    if annotations_written > 0:
                        processed_count[split] += 1
                    else:
                        # アノテーションが一つも書かれなかった場合、画像とラベルファイルを削除
                        print(f"\nWarning: No valid annotations found for {image_name}. Removing files.")
                        dst_image_path.unlink(missing_ok=True)
                        label_path.unlink(missing_ok=True)
                        skipped_count[split] += 1

                except OSError as e:
                    print(f"\nWarning: Failed to write label file {label_path}. Error: {e}")
                    skipped_count[split] += 1
                    dst_image_path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"\nWarning: An unexpected error occurred while processing annotations for {image_name}. Error: {e}")
                    skipped_count[split] += 1
                    label_path.unlink(missing_ok=True)
                    dst_image_path.unlink(missing_ok=True)

    print("\nDataset preparation summary:")
    print(f"Train images processed: {processed_count['train']}, skipped: {skipped_count['train']}")
    print(f"Validation images processed: {processed_count['val']}, skipped: {skipped_count['val']}")
    print(f"Test images processed: {processed_count['test']}, skipped: {skipped_count['test']}")
    print(f"No_use images processed: {processed_count['no_use']}, skipped: {skipped_count['no_use']}")
    print(f"YOLO character detection dataset (using page images) saved to: {output_path.resolve()}")


def create_dataset_yaml(output_dir: str):
    """データセット用のYAMLファイルを作成"""
    yaml_content = f"""# YOLO character detection dataset configuration
path: {output_dir}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 1

# Class names
names:
  0: character
"""

    yaml_path = Path(output_dir) / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Dataset YAML file created: {yaml_path}")


if __name__ == "__main__":
    # 設定
    DATASET_DIR = "data/raw/dataset"
    OUTPUT_DIR = "data/yolo_dataset_character_detection"
    TRAIN_DOCS_COUNT = 31
    VAL_DOCS_COUNT = 7
    TEST_DOCS_COUNT = 6
    RANDOM_SEED = 42
    CLASS_ID = 0  # 文字クラス（位置検出のみ）

    print("Starting YOLO character detection dataset preparation (using page images, split by book)...")
    prepare_yolo_character_dataset(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        train_docs_count=TRAIN_DOCS_COUNT,
        val_docs_count=VAL_DOCS_COUNT,
        test_docs_count=TEST_DOCS_COUNT,
        random_state=RANDOM_SEED,
        class_id=CLASS_ID,
    )

    # データセット用YAMLファイルの作成
    create_dataset_yaml(OUTPUT_DIR)

    print("YOLO character detection dataset preparation finished.")
