"""
YOLOフォーマットのデータセットを準備するスクリプト（1文字検出用）
短冊状の画像データと文字アノテーションを用いてYOLOフォーマットのデータセットを準備する。
1文字ごとのバウンディングボックス検出とクラス分類を行うためのデータセットを作成する。

データ構造:
- 短冊画像: data/column_dataset/{train,val,test}/images/{book_id}/{page_id}/{image_name}.jpg
- バウンディングボックス: data/column_dataset/{train,val,test}/bounding_boxes/{book_id}/{image_name}.json
- 文字列データ: data/column_dataset/{train,val,test}/labels/{book_id}/{image_name}.txt
- クラス情報: src/configs/data/yolo_oneline.yaml (6434クラス)
"""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.util import EasyDict  # Use absolute import

NO_MATCH_CHARS = [""]


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
    width = abs(x2 - x1)  # 幅は常に正
    height = abs(y2 - y1)  # 高さは常に正

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    # 稀に座標変換で範囲外になることがあるためクリップ
    x_center = np.clip(x_center, 0.0, 1.0)
    y_center = np.clip(y_center, 0.0, 1.0)
    width = np.clip(width, 0.0, 1.0)
    height = np.clip(height, 0.0, 1.0)

    return [x_center, y_center, width, height]


def get_image_size(image_path: Path) -> tuple[int, int]:
    """画像サイズを取得 (Pillowを使用)"""
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


def load_class_mapping(yaml_file: str) -> dict[str, int]:
    """
    yolo_oneline.yamlからクラス名->IDのマッピングを読み込む

    Args:
        yaml_file: yamlファイルのパス

    Returns:
        クラス名をキー、クラスIDを値とする辞書
    """
    try:
        class_mapping = {}

        with open(yaml_file, encoding="utf-8") as f:
            lines = f.readlines()

        # names:セクションを見つける
        in_names_section = False
        for line in lines:
            stripped_line = line.strip()

            if stripped_line == "names:":
                in_names_section = True
                continue

            if in_names_section:
                # 他のセクションが始まったら終了（インデントなしで:を含む行）
                if stripped_line and not line.startswith(" ") and ":" in stripped_line:
                    break

                # "  数値: 文字" の形式をパース（インデントありの行）
                if line.startswith("  ") and ":" in stripped_line:
                    try:
                        # "  123: 文字" を分割
                        parts = stripped_line.split(":", 1)
                        if len(parts) == 2:
                            class_id = int(parts[0].strip())
                            char = parts[1].strip()

                            # クォートを除去
                            if char.startswith('"') and char.endswith('"'):
                                char = char[1:-1]
                            elif char.startswith("'") and char.endswith("'"):
                                char = char[1:-1]

                            class_mapping[char] = class_id
                    except (ValueError, IndexError):
                        # パースに失敗した行はスキップ
                        continue

        if not class_mapping:
            raise ValueError(f"No class mapping found in {yaml_file}")

        print(f"Loaded {len(class_mapping)} classes from {yaml_file}")
        return class_mapping

    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found: {yaml_file}")
    except Exception as e:
        raise ValueError(f"Error loading class mapping from {yaml_file}: {e}")


def load_bounding_boxes(bbox_file: Path) -> list[list[float]]:
    """
    JSONファイルからバウンディングボックスを読み込む

    Args:
        bbox_file: バウンディングボックスJSONファイルのパス

    Returns:
        バウンディングボックスのリスト [[x1, y1, x2, y2], ...]
    """
    try:
        with open(bbox_file, encoding="utf-8") as f:
            bboxes = json.load(f)

        if not isinstance(bboxes, list):
            raise ValueError(f"Expected list of bounding boxes, got {type(bboxes)}")

        validated_bboxes = []
        for i, bbox in enumerate(bboxes):
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Bounding box {i} should be [x1, y1, x2, y2], got {bbox}")
            if not all(isinstance(coord, int | float) for coord in bbox):
                raise ValueError(f"All coordinates should be numeric, got {bbox}")
            validated_bboxes.append([float(coord) for coord in bbox])

        return validated_bboxes

    except FileNotFoundError:
        raise FileNotFoundError(f"Bounding box file not found: {bbox_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON file {bbox_file}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading bounding boxes from {bbox_file}: {e}")


def load_text_label(label_file: Path) -> str:
    """
    テキストファイルから文字列を読み込む

    Args:
        label_file: ラベルテキストファイルのパス

    Returns:
        文字列
    """
    try:
        with open(label_file, encoding="utf-8") as f:
            text = f.read().strip()
        return text

    except FileNotFoundError:
        raise FileNotFoundError(f"Label file not found: {label_file}")
    except Exception as e:
        raise ValueError(f"Error loading text label from {label_file}: {e}")


def match_chars_to_bboxes(text: str, bboxes: list[list[float]]) -> list[tuple[str, list[float]]]:
    """
    文字列とバウンディングボックスを対応付ける

    Args:
        text: 文字列
        bboxes: バウンディングボックスのリスト

    Returns:
        (文字, バウンディングボックス)のペアのリスト
    """
    if len(text) != len(bboxes):
        raise ValueError(f"Number of characters ({len(text)}) doesn't match number of bounding boxes ({len(bboxes)})")

    return list(zip(text, bboxes, strict=False))


def char_to_class_id(char: str, class_mapping: dict[str, int]) -> int:
    """
    文字をクラスIDに変換する

    Args:
        char: 文字
        class_mapping: クラス名->IDのマッピング

    Returns:
        クラスID
    """
    if char not in class_mapping:
        NO_MATCH_CHARS.append(char)
        raise ValueError(f"Character '{char}' not found in class mapping")

    return class_mapping[char]


def prepare_yolo_dataset_from_columns(
    input_base_dir: str,
    output_dir: str,
    class_mapping_file: str,
) -> None:
    """
    短冊状の画像データと文字アノテーションを用いてYOLOフォーマットのデータセットを準備する

    Args:
        input_base_dir: 入力データのベースディレクトリ (data/column_dataset)
        output_dir: 出力ディレクトリ
        class_mapping_file: クラスマッピングYAMLファイルのパス
    """
    input_path = Path(input_base_dir)
    output_path = Path(output_dir)

    # クラスマッピングを読み込み
    try:
        class_mapping = load_class_mapping(class_mapping_file)
    except Exception as e:
        print(f"Error: Failed to load class mapping: {e}")
        return

    # 出力ディレクトリの準備
    for split in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            (output_path / split / subdir).mkdir(parents=True, exist_ok=True)

    # 統計情報
    stats = EasyDict(
        {
            "train": {"processed": 0, "skipped": 0, "characters": 0},
            "val": {"processed": 0, "skipped": 0, "characters": 0},
            "test": {"processed": 0, "skipped": 0, "characters": 0},
        }
    )

    # 各分割に対して処理
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split} dataset...")

        split_input_path = input_path / split
        split_output_images = output_path / split / "images"
        split_output_labels = output_path / split / "labels"

        # 入力ディレクトリが存在するかチェック
        if not split_input_path.exists():
            print(f"Warning: {split_input_path} does not exist. Skipping {split} split.")
            continue

        images_dir = split_input_path / "images"
        bboxes_dir = split_input_path / "bounding_boxes"
        labels_dir = split_input_path / "labels"

        # 必要なディレクトリが存在するかチェック
        missing_dirs = []
        for dir_path, dir_name in [(images_dir, "images"), (bboxes_dir, "bounding_boxes"), (labels_dir, "labels")]:
            if not dir_path.exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            print(f"Error: Missing directories for {split}: {missing_dirs}. Skipping {split} split.")
            continue

        # 書籍ディレクトリを取得
        book_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
        print(f"Found {len(book_dirs)} book directories in {split}")

        # 各書籍を処理
        for book_dir in tqdm(book_dirs, desc=f"Processing {split} books"):
            book_id = book_dir.name

            # 対応するバウンディングボックスとラベルディレクトリ
            book_bbox_dir = bboxes_dir / book_id
            book_label_dir = labels_dir / book_id

            if not book_bbox_dir.exists():
                print(f"Warning: Bounding box directory missing for book {book_id}. Skipping.")
                continue

            if not book_label_dir.exists():
                print(f"Warning: Label directory missing for book {book_id}. Skipping.")
                continue

            # column_dataset_paddedでは、ページレベルのディレクトリがなく、
            # 直接書籍ディレクトリに画像ファイルが配置されている
            image_files = list(book_dir.glob("*.jpg"))

            for image_file in image_files:
                image_name = image_file.stem  # 拡張子なしのファイル名

                # 対応するバウンディングボックスとラベルファイル
                bbox_file = book_bbox_dir / f"{image_name}.json"
                label_file = book_label_dir / f"{image_name}.txt"

                try:
                    # ファイルの存在チェック
                    if not bbox_file.exists():
                        print(f"Warning: Bounding box file missing: {bbox_file}")
                        stats[split]["skipped"] += 1
                        continue

                    if not label_file.exists():
                        print(f"Warning: Label file missing: {label_file}")
                        stats[split]["skipped"] += 1
                        continue

                    # データを読み込み
                    bboxes = load_bounding_boxes(bbox_file)
                    text = load_text_label(label_file)

                    if not text:  # 空の文字列の場合はスキップ
                        stats[split]["skipped"] += 1
                        continue

                    # 文字とバウンディングボックスを対応付け
                    char_bbox_pairs = match_chars_to_bboxes(text, bboxes)

                    # 画像サイズを取得
                    image_width, image_height = get_image_size(image_file)

                    # 画像をコピー
                    dst_image_path = split_output_images / f"{image_name}.jpg"
                    shutil.copy2(image_file, dst_image_path)

                    # YOLOフォーマットのラベルファイルを作成
                    yolo_label_path = split_output_labels / f"{image_name}.txt"

                    with open(yolo_label_path, "w", encoding="utf-8") as f:
                        char_count = 0
                        for char, bbox in char_bbox_pairs:
                            try:
                                # クラスIDを取得
                                class_id = char_to_class_id(char, class_mapping)

                                # YOLOフォーマットに変換
                                yolo_bbox = convert_to_yolo_format(bbox, image_width, image_height)

                                # ラベルファイルに書き込み
                                f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
                                char_count += 1

                            except ValueError as e:
                                print(f"Warning: Skipping character '{char}' in {image_name}: {e}")
                                continue

                        stats[split]["characters"] += char_count

                    if char_count > 0:
                        stats[split]["processed"] += 1
                    else:
                        # 有効な文字がなかった場合はファイルを削除
                        dst_image_path.unlink(missing_ok=True)
                        yolo_label_path.unlink(missing_ok=True)
                        stats[split]["skipped"] += 1

                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    stats[split]["skipped"] += 1
                    continue

    # 統計情報を表示
    print("\n" + "=" * 50)
    print("Dataset preparation summary:")
    print("=" * 50)
    for split in ["train", "val", "test"]:
        print(f"{split.upper()}:")
        print(f"  Processed images: {stats[split]['processed']}")
        print(f"  Skipped images: {stats[split]['skipped']}")
        print(f"  Total characters: {stats[split]['characters']}")

    print(f"\nYOLO dataset saved to: {output_path.resolve()}")
    print(f"No match chars: {set(NO_MATCH_CHARS)}")


if __name__ == "__main__":
    # --- 設定 ---
    INPUT_BASE_DIR = "data/column_dataset_padded"
    OUTPUT_YOLO_DIR = "data/yolo_dataset_oneline"
    CLASS_MAPPING_FILE = "src/configs/data/yolo_oneline.yaml"
    # -------------

    print("Starting YOLO dataset preparation (using column images, character detection)...")
    prepare_yolo_dataset_from_columns(
        input_base_dir=INPUT_BASE_DIR,
        output_dir=OUTPUT_YOLO_DIR,
        class_mapping_file=CLASS_MAPPING_FILE,
    )
    print("YOLO dataset preparation finished.")
