import ast
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.util import EasyDict  # Use absolute import


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


def safe_literal_eval(val):
    """安全にリテラルを評価する"""
    if isinstance(val, list | tuple):
        return val
    if isinstance(val, str):
        try:
            stripped_val = val.strip()
            if stripped_val.startswith("[") and stripped_val.endswith("]"):
                evaluated = ast.literal_eval(stripped_val)
                if isinstance(evaluated, list):
                    return evaluated
                else:
                    raise ValueError("Evaluated result is not a list.")
            else:
                raise ValueError(f"String '{val}' is not a valid list format.")
        except (ValueError, SyntaxError, TypeError, MemoryError) as e:
            raise ValueError(f"Failed to safely evaluate string '{val}' as literal list. Error: {e}")
    raise TypeError(f"Input must be a string representation of a list, or a list/tuple, but got {type(val)}")


def get_doc_id_from_original_path(original_image_path: str) -> str:
    """
    original_image のパス文字列からドキュメントID (書籍ID) を抽出する。
    期待されるパス形式: 'data/raw/dataset/<doc_id>/images/<image_name>.jpg'
    """
    try:
        parts = Path(original_image_path).parts
        if "dataset" in parts:
            try:
                idx = parts.index("dataset")
                if idx + 1 < len(parts):
                    doc_id = parts[idx + 1]
                    if doc_id:
                        return doc_id
                    else:
                        raise ValueError("Extracted doc ID after 'dataset' is empty.")
                else:
                    raise ValueError("Path structure after 'dataset' is unexpected (no element after 'dataset').")
            except ValueError:
                raise ValueError("'dataset' segment not found in the path.")
        else:
            raise ValueError("Path does not contain the expected 'dataset' segment.")

    except (ValueError, IndexError) as e:
        raise ValueError(f"Could not extract document ID from original_image path: '{original_image_path}'. Error: {e}")


def prepare_yolo_dataset(
    column_info_file: str,
    output_dir: str,
    train_docs_count: int = 31,
    val_docs_count: int = 7,
    test_docs_count: int = 6,
    random_state: int = 42,
    class_id: int = 0,
):
    """
    1ページ画像と短冊アノテーションを用いてYOLOフォーマットのデータセットを準備 (書籍単位で分割)
    画像は data/raw/dataset からコピーし、アノテーションは column_info_file から取得する
    """
    column_info_path = Path(column_info_file)
    output_path = Path(output_dir)

    if not column_info_path.exists():
        print(f"Error: Column info file not found at {column_info_path}")
        return

    # 出力ディレクトリの準備 (train, val, test)
    for split in ["train", "val", "test", "no_use"]:
        for subdir in ["images", "labels"]:
            (output_path / split / subdir).mkdir(parents=True, exist_ok=True)

    # 列情報の読み込みと必須カラム確認
    try:
        df = pd.read_csv(column_info_path)
        # 必須カラム: original_image (画像パス), box_in_original (アノテーション)
        required_columns = ["original_image", "box_in_original"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Error: Missing required columns in {column_info_path}: {missing}")
            return
        # NaN値を含む行を削除
        df.dropna(subset=required_columns, inplace=True)
        if df.empty:
            print(f"Error: No valid data after dropping NaN in required columns from {column_info_path}")
            return

    except Exception as e:
        print(f"Error reading or processing CSV file {column_info_path}: {e}")
        return

    # ドキュメントIDを original_image パスから抽出
    doc_ids = []
    invalid_paths = []
    for path_str in df["original_image"]:
        try:
            doc_id = get_doc_id_from_original_path(str(path_str))
            doc_ids.append(doc_id)
        except ValueError as e:
            print(f"Warning: Could not extract document ID from original_image path: {path_str}. Error: {e}")
            invalid_paths.append(str(path_str))
            doc_ids.append(None)

    df["doc_id"] = doc_ids
    df.dropna(subset=["doc_id"], inplace=True)  # ID抽出に失敗した行を削除


    if invalid_paths:
        print(f"Warning: Could not extract document ID from {len(invalid_paths)} paths. Examples:")
        for i, p in enumerate(invalid_paths):
            if i >= 5:
                break
            print(f"  - {p}")
        if df.empty:
            print("Error: No valid entries remaining after removing rows with ID extraction errors.")
            return

    # --- original_image パスの存在確認 ---
    # 文字列のパスをPathオブジェクトに変換し、存在しないものをフィルタリング
    original_image_paths = df["original_image"].apply(lambda p: Path(str(p)))
    original_count = len(df)
    exists_mask = original_image_paths.apply(lambda p: p.exists())
    df = df[exists_mask]
    removed_count = original_count - len(df)
    if removed_count > 0:
        print(f"Warning: Removed {removed_count} entries due to missing original_image files.")
        # 存在しないパスの例を表示（オプション）
        missing_paths = original_image_paths[~exists_mask].unique()
        print("Examples of missing original_image paths:")
        for i, p in enumerate(missing_paths):
            if i >= 5:
                break
            print(f"  - {p}")

    if df.empty:
        print("Error: No valid entries found after checking original_image file existence.")
        return
    # ------------------------------------

    # --- Split logic: use existing split info if present ---
    split_info_path = output_path / "dataset_split_info.csv"
    doc_id_splits = {"train": set(), "val": set(), "test": set(), "no_use": set()}
    use_existing_split = False
    if split_info_path.exists():
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
        # ユニークなドキュメントIDを取得
        unique_doc_ids = df["doc_id"].unique()
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
            unique_doc_ids, test_size=(val_docs_count + test_docs_count), train_size=train_docs_count, random_state=random_state
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
        if len(train_doc_ids) != train_docs_count or len(val_doc_ids) != val_docs_count or len(test_doc_ids) != test_docs_count:
            print("Warning: The final split counts do not exactly match the requested counts.")

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
    # ------------------------------------

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

        # このsplitに含まれるデータをフィルタリング
        split_df = df[df["doc_id"].isin(target_doc_ids)]
        # このsplitで処理すべきユニークな *オリジナル画像* パスを取得
        unique_original_image_paths_str = split_df["original_image"].unique()

        print(
            f"Processing {len(unique_original_image_paths_str)} unique original images for {split} split from {len(target_doc_ids)} documents."
        )

        # --- オリジナル画像ごとに処理 ---
        for original_image_path_str in tqdm(unique_original_image_paths_str, desc=f"Processing {split} images"):
            original_image_path = Path(original_image_path_str)
            image_name = original_image_path.name
            dst_image_path = split_output_images / image_name

            # 画像コピー (既に存在する場合はスキップしない、上書きする)
            try:
                shutil.copy2(original_image_path, dst_image_path)
            except Exception as e:
                print(f"\nWarning: Failed to copy {original_image_path} to {dst_image_path}. Skipping this image. Error: {e}")
                skipped_count[split] += 1  # 画像単位でスキップカウント
                continue

            # 画像サイズ取得
            try:
                image_width, image_height = get_image_size(original_image_path)
            except (FileNotFoundError, ValueError) as e:
                print(f"\nWarning: Skipping {original_image_path} due to error getting size: {e}")
                skipped_count[split] += 1
                # コピーした画像を削除
                dst_image_path.unlink(missing_ok=True)
                continue

            # このオリジナル画像に対応するアノテーションを取得
            annotations = split_df[split_df["original_image"] == original_image_path_str]
            label_name = f"{original_image_path.stem}.txt"
            label_path = split_output_labels / label_name
            annotations_written = 0
            try:
                with open(label_path, "w", encoding="utf-8") as f:
                    for _, row in annotations.iterrows():
                        try:
                            # box_in_original を評価
                            box = safe_literal_eval(row["box_in_original"])
                            if not isinstance(box, list) or len(box) != 4:
                                raise ValueError(f"Expected list of 4 elements, got {box}")
                            if not all(isinstance(coord, int | float) for coord in box):
                                raise ValueError(f"Box coordinates must be numeric: {box}")

                        except (ValueError, TypeError) as e:
                            print(
                                f"\nWarning: Skipping invalid box data for {image_name}: '{row['box_in_original']}'. Error: {e}"
                            )
                            continue  # このアノテーションをスキップ

                        try:
                            # YOLOフォーマットに変換
                            yolo_box = convert_to_yolo_format(box, image_width=image_width, image_height=image_height)
                        except ValueError as e:
                            print(f"\nWarning: Skipping box conversion for {image_name}, box {box}. Error: {e}")
                            continue  # このアノテーションをスキップ

                        f.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")
                        annotations_written += 1

                if annotations_written > 0:
                    processed_count[split] += 1  # アノテーションが1つ以上書かれた画像をカウント
                else:
                    # アノテーションが一つも書かれなかった場合、画像と空のラベルファイルを削除
                    print(f"\nWarning: No valid annotations found for {image_name}. Removing image and label file.")
                    dst_image_path.unlink(missing_ok=True)
                    label_path.unlink(missing_ok=True)
                    skipped_count[split] += 1  # 画像単位でスキップカウント

            except OSError as e:
                print(f"\nWarning: Failed to write label file {label_path}. Skipping image {image_name}. Error: {e}")
                skipped_count[split] += 1
                # コピーした画像を削除
                dst_image_path.unlink(missing_ok=True)
            except Exception as e:
                print(
                    f"\nWarning: An unexpected error occurred while processing annotations for {image_name}. Skipping. Error: {e}"
                )
                skipped_count[split] += 1
                # 不完全なファイルを削除
                label_path.unlink(missing_ok=True)
                dst_image_path.unlink(missing_ok=True)

    print("\nDataset preparation summary:")
    print(f"Train images processed: {processed_count['train']}, skipped: {skipped_count['train']}")
    print(f"Validation images processed: {processed_count['val']}, skipped: {skipped_count['val']}")
    print(f"Test images processed: {processed_count['test']}, skipped: {skipped_count['test']}")
    print(f"No_use images processed: {processed_count['no_use']}, skipped: {skipped_count['no_use']}")
    print(f"YOLO dataset (using page images) saved to: {output_path.resolve()}")


if __name__ == "__main__":
    # --- 設定 ---
    COLUMN_INFO_FILE = "data/processed/column_info.csv"
    # 出力ディレクトリ名を変更して、以前のスクリプトの出力と区別する
    OUTPUT_YOLO_DIR = "data/yolo_dataset_page_images_by_book"
    # 指定された書籍数
    TRAIN_DOCS_COUNT = 27
    VAL_DOCS_COUNT = 4
    TEST_DOCS_COUNT = 3
    RANDOM_SEED = 42
    CLASS_ID = 0  # 列検出タスクなのでクラスは1つ (0)
    # -------------

    print("Starting YOLO dataset preparation (using page images, split by book)...")
    prepare_yolo_dataset(
        column_info_file=COLUMN_INFO_FILE,
        output_dir=OUTPUT_YOLO_DIR,
        train_docs_count=TRAIN_DOCS_COUNT,
        val_docs_count=VAL_DOCS_COUNT,
        test_docs_count=TEST_DOCS_COUNT,
        random_state=RANDOM_SEED,
        class_id=CLASS_ID,
    )
    print("YOLO dataset preparation finished.")
