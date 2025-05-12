"""
列画像データセットを書籍単位で分割し、アノテーション情報を整理するスクリプト。
- カラム画像を train/val/test ディレクトリにコピー
- アノテーション情報 (column_info.csv) を train/val/test ごとに分割して保存
- Unicode -> ID のマッピングファイルを作成
"""
import ast
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Utility Functions (from prepare_yolo_dataset.py or modified) ---


def safe_literal_eval(val):
    """安全にリテラルを評価する (主にリスト形式の文字列用)"""
    if isinstance(val, list | tuple):
        return val
    if isinstance(val, str):
        try:
            stripped_val = val.strip()
            # Check if it looks like a list representation
            if stripped_val.startswith("[") and stripped_val.endswith("]"):
                evaluated = ast.literal_eval(stripped_val)
                # Ensure the evaluated result is actually a list
                if isinstance(evaluated, list):
                    return evaluated
                else:
                    # Handle cases like "[1]" being evaluated to 1 by some libraries
                    # Although ast.literal_eval should return a list for "[1]"
                    logger.warning(f"Evaluated result for '{val}' is not a list: {type(evaluated)}. Returning empty list.")
                    return []  # Return empty list or handle as error based on requirements
            else:
                # If it doesn't look like a list, return empty list or raise error
                logger.warning(f"String '{val}' is not a valid list format. Returning empty list.")
                return []  # Assuming empty list is acceptable for non-list strings
        except (ValueError, SyntaxError, TypeError, MemoryError) as e:
            logger.error(f"Failed to safely evaluate string '{val}' as literal list. Error: {e}. Returning empty list.")
            return []  # Return empty list on error
    # Handle other types if necessary, or return empty list/raise error
    logger.warning(f"Input type {type(val)} is not a string or list/tuple. Returning empty list.")
    return []


def get_doc_id_from_original_path(original_image_path: str) -> str | None:
    """
    original_image のパス文字列からドキュメントID (書籍ID) を抽出する。
    期待されるパス形式: 'data/raw/dataset/<doc_id>/images/<image_name>.jpg'
    抽出できない場合は None を返す。
    """
    try:
        parts = Path(str(original_image_path)).parts
        # パス構造が異なる可能性を考慮し、'dataset' の後の要素をIDとする
        if "dataset" in parts:
            try:
                idx = parts.index("dataset")
                if idx + 1 < len(parts):
                    doc_id = parts[idx + 1]
                    if doc_id:  # 空文字列でないことを確認
                        return doc_id
                    else:
                        logger.warning(f"Extracted doc ID is empty for path: {original_image_path}")
                        return None
                else:
                    logger.warning(f"Path structure unexpected (no element after 'dataset'): {original_image_path}")
                    return None
            except ValueError:  # 'dataset' が見つからない場合 (parts.index がエラー)
                logger.warning(f"'dataset' segment not found in path: {original_image_path}")
                return None
        else:
            # 'dataset' がパスに含まれない場合も抽出失敗とする
            logger.warning(f"Path does not contain 'dataset' segment: {original_image_path}")
            return None
    except Exception as e:  # 予期せぬエラー
        logger.error(f"Unexpected error extracting document ID from path: '{original_image_path}'. Error: {e}")
        return None


def create_unicode_map(all_unicode_lists):
    """
    Unicode文字列のリストのリストから、ユニークなUnicode文字を抽出し、
    ソートしてIDを割り当てる辞書を作成する。
    """
    unique_unicodes = set()
    for unicode_list in all_unicode_lists:
        # safe_literal_eval は既にリストを返すと仮定
        # unicode_list が None や空リストの場合も考慮
        if unicode_list:
            unique_unicodes.update(unicode_list)

    # 'U+XXXX' 形式でないものや空文字列を除外 (念のため)
    valid_unicodes = {u for u in unique_unicodes if isinstance(u, str) and u.startswith("U+")}

    if not valid_unicodes:
        logger.warning("No valid Unicode characters found to create a map.")
        return {}

    # Unicode文字列でソート
    sorted_unicodes = sorted(valid_unicodes)

    # IDを0から割り当て
    unicode_to_id = {unicode_char: i for i, unicode_char in enumerate(sorted_unicodes)}
    logger.info(f"Created Unicode map with {len(unicode_to_id)} unique characters.")
    return unicode_to_id


# --- Main Function ---


def prepare_column_dataset(
    column_info_file: str,
    column_image_dir: str,  # カラム画像が格納されているディレクトリ
    output_dir: str,
    train_docs_count: int = 31,
    val_docs_count: int = 7,
    test_docs_count: int = 6,
    random_state: int = 42,
):
    """
    カラム画像とアノテーション情報を用いて、書籍単位で分割されたデータセットを準備する。
    - カラム画像を train/val/test ディレクトリにコピー
    - アノテーション情報 (column_info.csv) を train/val/test ごとに分割して保存
    - Unicode -> ID のマッピングファイルを作成
    """
    column_info_path = Path(column_info_file)
    # column_image_dir はCSV内のパスがルートからの相対パスであることを前提とするため、直接は使用しない
    # column_image_base_path = Path(column_image_dir) # 不要になる
    output_path = Path(output_dir)

    # --- Input Validation ---
    if not column_info_path.exists():
        logger.error(f"Error: Column info file not found at {column_info_path}")
        return
    # CSV内のパスがファイルへの直接パスなので、ディレクトリ存在確認は不要
    # if not column_image_base_path.is_dir():
    #     logger.error(f"Error: Column image directory not found at {column_image_base_path}")
    #     return

    # --- Output Directory Setup ---
    for split in ["train", "val", "test"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories created under: {output_path}")

    # --- Load and Prepare Column Info Data ---
    try:
        df = pd.read_csv(column_info_path)
        logger.info(f"Loaded column info from: {column_info_path} ({len(df)} rows)")

        # 必須カラム確認
        required_columns = ["column_image", "original_image", "unicode_ids"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"Error: Missing required columns in {column_info_path}: {missing}")
            return

        # NaN値を含む行を必須カラム基準で削除
        initial_rows = len(df)
        df.dropna(subset=required_columns, inplace=True)
        if len(df) < initial_rows:
            logger.warning(f"Removed {initial_rows - len(df)} rows with NaN in required columns.")

        if df.empty:
            logger.error(f"Error: No valid data after dropping NaN in required columns from {column_info_path}")
            return

        # unicode_ids を評価してリスト形式に変換 (エラー時は空リスト)
        df["unicode_list"] = df["unicode_ids"].apply(safe_literal_eval)

        # --- Filter by Column Width (from box_in_original) ---
        logger.info("Filtering columns by width (min_width=64)...")
        initial_rows_before_width_filter = len(df)

        def calculate_column_width(box_str):
            # box_str は safe_literal_eval で評価済みのリスト、または評価前の文字列
            # まず box_in_original を評価
            box_list = safe_literal_eval(box_str)
            if isinstance(box_list, list) and len(box_list) == 4:
                try:
                    # x_min, y_min, x_max, y_max
                    x_min, _, x_max, _ = map(float, box_list) # 数値に変換
                    width = x_max - x_min
                    return width
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse box coordinates from {box_list}: {e}. Marking as invalid width.")
                    return -1 # 計算失敗を示す値
            else:
                logger.debug(f"Invalid format for box_in_original: {box_str}. Expected list of 4 elements. Marking as invalid width.")
                return -1 # 不正な形式または計算失敗
        
        def calculate_column_height(box_str):
            # box_str は safe_literal_eval で評価済みのリスト、または評価前の文字列
            # まず box_in_original を評価
            box_list = safe_literal_eval(box_str)
            if isinstance(box_list, list) and len(box_list) == 4:
                try:
                    # x_min, y_min, x_max, y_max
                    _, y_min, _, y_max = map(float, box_list) # 数値に変換
                    height = y_max - y_min
                    return height
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse box coordinates from {box_list}: {e}. Marking as invalid height.")
                    return -1
            else:
                logger.debug(f"Invalid format for box_in_original: {box_str}. Expected list of 4 elements. Marking as invalid height.")
                return -1

        # 'box_in_original' カラムが存在するか確認
        if "box_in_original" in df.columns:
            #df["column_width"] = df["box_in_original"].apply(calculate_column_width)
            
            # # 幅が100未満、または計算に失敗した行を除外
            # valid_width_mask = (df["column_width"] >= 100)
            # df = df[valid_width_mask]
            
            # rows_after_width_filter = len(df)
            # removed_by_width = initial_rows_before_width_filter - rows_after_width_filter
            # if removed_by_width > 0:
            #     logger.info(f"Removed {removed_by_width} columns with width < 64 pixels or invalid 'box_in_original' format.")
            
            # # 高さが4000以上の行を除外
            # df["column_height"] = df["box_in_original"].apply(calculate_column_height)
            # valid_height_mask = (df["column_height"] < 4000)
            # df = df[valid_height_mask]
            # rows_after_height_filter = len(df)
            # removed_by_height = rows_after_width_filter - rows_after_height_filter
            # if removed_by_height > 0:
            #     logger.info(f"Removed {removed_by_height} columns with height >= 4000 pixels.")

            # 高さが幅の16倍より大きい行を除外
            df["column_height"] = df["box_in_original"].apply(calculate_column_height)
            df["column_width"] = df["box_in_original"].apply(calculate_column_width)
            valid_height_mask = (df["column_height"] <= 16 * df["column_width"])
            df = df[valid_height_mask]
            rows_after_height_filter = len(df)
            removed_by_height = initial_rows_before_width_filter - rows_after_height_filter
            if removed_by_height > 0:
                logger.info(f"Removed {removed_by_height} columns with height >= 16 times the width.")
        else:
            logger.warning("'box_in_original' column not found. Skipping width-based filtering.")

        if df.empty:
            logger.error(f"Error: No valid data after width filtering from {column_info_path}")
            return
        # --- End Filter by Column Width ---

    except Exception as e:
        logger.error(f"Error reading or processing CSV file {column_info_path}: {e}")
        return

    # --- Extract Document IDs ---
    df["doc_id"] = df["original_image"].apply(get_doc_id_from_original_path)

    # doc_id が None (抽出失敗) の行を削除
    initial_rows = len(df)
    df.dropna(subset=["doc_id"], inplace=True)
    removed_count = initial_rows - len(df)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} rows where document ID could not be extracted.")

    if df.empty:
        logger.error("Error: No valid entries remaining after attempting to extract document IDs.")
        return

    # --- Check Column Image Existence ---
    # column_image 列のパス文字列の先頭に 'data/' を追加して Path オブジェクトに変換し、存在確認
    # get_full_column_path 関数は不要なので削除
    df["column_image_path_obj"] = df["column_image"].apply(lambda p: Path(f"data/{str(p)}"))  # 'data/' を追加
    exists_mask = df["column_image_path_obj"].apply(lambda p: p.exists())

    initial_rows = len(df)
    df = df[exists_mask]
    removed_count = initial_rows - len(df)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} entries due to missing column_image files.")
        # 存在しないパスの例を表示（オプション） - 修正: カラム名を変更
        missing_paths = df.loc[~exists_mask, "column_image_path_obj"].unique()  # .loc を使用して boolean indexing
        logger.warning("Examples of missing column_image paths:")
        for i, p in enumerate(missing_paths):
            if i >= 5:
                break
            logger.warning(f"  - {p}")

    if df.empty:
        logger.error("Error: No valid entries found after checking column_image file existence.")
        return

    # --- Create Unicode Map ---
    all_unicode_lists = df["unicode_list"].tolist()
    unicode_to_id_map = create_unicode_map(all_unicode_lists)

    # Save Unicode map
    # unicode_map_path = output_path / "unicode_to_id.json"
    # try:
    #     with open(unicode_map_path, "w", encoding="utf-8") as f:
    #         json.dump(unicode_to_id_map, f, ensure_ascii=False, indent=4)
    #     logger.info(f"Saved Unicode map to: {unicode_map_path}")

    #     # --- Create and Save id_to_unicode map ---
    #     id_to_unicode_map = {v: k for k, v in unicode_to_id_map.items()}
    #     id_map_path = output_path / "id_to_unicode.json"
    #     with open(id_map_path, "w", encoding="utf-8") as f:
    #         json.dump(id_to_unicode_map, f, ensure_ascii=False, indent=4)
    #     logger.info(f"Saved ID to Unicode map to: {id_map_path}")

    #     # --- Create and Save unicode_to_character map ---
    #     unicode_to_char_map = {}
    #     for unicode_str in unicode_to_id_map.keys():
    #         try:
    #             # 'U+XXXX' から文字へ変換
    #             char_code = int(unicode_str[2:], 16)
    #             character = chr(char_code)
    #             unicode_to_char_map[unicode_str] = character
    #         except ValueError:
    #             logger.warning(f"Could not convert Unicode string '{unicode_str}' to character. Skipping.")
    #     char_map_path = output_path / "unicode_to_character.json"
    #     with open(char_map_path, "w", encoding="utf-8") as f:
    #         json.dump(unicode_to_char_map, f, ensure_ascii=False, indent=4)
    #     logger.info(f"Saved Unicode to Character map to: {char_map_path}")

    # except Exception as e:
    #     logger.error(f"Error saving Unicode maps: {e}")
    #     # マップ作成失敗は致命的ではないかもしれないが、警告は出す

    # --- Split Documents ---
    unique_doc_ids = df["doc_id"].unique()
    total_docs = len(unique_doc_ids)
    logger.info(f"Found {total_docs} unique document IDs with existing images and annotations.")

    requested_total = train_docs_count + val_docs_count + test_docs_count
    if total_docs < requested_total:
        logger.error(
            f"Error: Not enough unique documents ({total_docs}) to fulfill the requested split "
            f"({train_docs_count} train, {val_docs_count} val, {test_docs_count} test)."
        )
        return
    elif total_docs > requested_total:
        logger.warning(
            f"Warning: Found {total_docs} documents, but only {requested_total} are requested for the split. "
            f"{total_docs - requested_total} documents will be unused."
        )

    # Split document IDs
    try:
        train_doc_ids, temp_doc_ids = train_test_split(
            unique_doc_ids,
            test_size=(val_docs_count + test_docs_count),
            train_size=train_docs_count,  # train_size を指定
            random_state=random_state,
            shuffle=True,  # シャッフルを有効にする
        )

        if val_docs_count + test_docs_count > 0 and len(temp_doc_ids) > 0:
            # val と test の比率を計算 (0除算を回避)
            val_test_total = val_docs_count + test_docs_count
            test_proportion_in_temp = test_docs_count / val_test_total if val_test_total > 0 else 0

            if test_proportion_in_temp == 1.0:  # test のみ
                val_doc_ids = np.array([])
                test_doc_ids = temp_doc_ids
            elif test_proportion_in_temp == 0.0:  # val のみ
                val_doc_ids = temp_doc_ids
                test_doc_ids = np.array([])
            else:  # val と test 両方
                val_doc_ids, test_doc_ids = train_test_split(
                    temp_doc_ids,
                    test_size=test_proportion_in_temp,
                    random_state=random_state,
                    shuffle=True,  # シャッフルを有効にする
                )
        else:  # val も test も 0 の場合 or temp_doc_ids が空の場合
            val_doc_ids = np.array([])
            test_doc_ids = np.array([])

    except ValueError as e:
        logger.error(f"Error during train_test_split: {e}. Check document counts and total documents.")
        return

    logger.info(f"Actual split: Train={len(train_doc_ids)}, Val={len(val_doc_ids)}, Test={len(test_doc_ids)}")
    if len(train_doc_ids) != train_docs_count or len(val_doc_ids) != val_docs_count or len(test_doc_ids) != test_docs_count:
        logger.warning(
            "Warning: The final split counts may not exactly match the requested counts due to rounding or insufficient data in intermediate splits."
        )

    doc_id_splits = {"train": set(train_doc_ids), "val": set(val_doc_ids), "test": set(test_doc_ids)}

    # --- Save Split Information ---
    split_records = []
    for split_name, ids in doc_id_splits.items():
        for doc_id in ids:
            split_records.append({"doc_id": doc_id, "split": split_name})

    split_df_info = pd.DataFrame(split_records)
    split_info_path = output_path / "dataset_split_info.csv"
    try:
        split_df_info.sort_values(by=["split", "doc_id"]).to_csv(split_info_path, index=False, encoding="utf-8")
        logger.info(f"Saved dataset split information to: {split_info_path}")
    except Exception as e:
        logger.warning(f"Warning: Failed to save dataset split information to {split_info_path}. Error: {e}")

    # --- Process Each Split ---
    processed_counts = defaultdict(int)
    skipped_counts = defaultdict(int)

    for split, target_doc_ids in doc_id_splits.items():
        if not target_doc_ids:
            logger.info(f"\nSkipping {split} dataset as no documents were assigned.")
            continue

        logger.info(f"\nPreparing {split} dataset...")
        split_output_image_dir = output_path / split / "images"
        split_output_csv_path = output_path / split / f"{split}_column_info.csv"

        # Filter dataframe for the current split
        split_df = df[df["doc_id"].isin(target_doc_ids)].copy()  # .copy() を追加して SettingWithCopyWarning を回避

        if split_df.empty:
            logger.warning(f"No data found for split '{split}' after filtering by doc_id. Skipping.")
            continue

        # --- Update column_image path for the output CSV ---
        # The path should be relative to the output_dir (e.g., data/column_dataset)
        # Capture the current value of 'split' using a default argument
        def get_new_column_image_path(original_column_image_path_str, current_split=split):
            try:
                # Calculate the relative path within the 'processed/column_images' directory
                # Example: book_id/page_id/column_id.png
                relative_path_in_processed = Path(original_column_image_path_str).relative_to("processed/column_images")
                # Construct the destination path relative to the output directory root
                # Example: train/images/book_id/page_id/column_id.png
                # Use the captured 'current_split' value
                new_path = Path(current_split) / "images" / relative_path_in_processed
                return str(new_path)  # Return as string
            except ValueError:
                logger.warning(
                    f"Could not determine relative path for {original_column_image_path_str} when updating path for CSV. Keeping original."
                )
                return original_column_image_path_str  # Keep original if error

        split_df["column_image"] = split_df["column_image"].apply(get_new_column_image_path)
        logger.info(f"Updated 'column_image' paths in DataFrame for {split} split.")

        # Save the filtered annotation CSV
        try:
            # 不要になった一時列や絶対パス列を削除してから保存
            columns_to_save = [
                col
                for col in [
                    "column_image",
                    "original_image",
                    "box_in_original",
                    "char_boxes_in_column",
                    "unicode_ids",
                    "doc_id",
                ]
                if col in split_df.columns
            ]
            split_df_to_save = split_df[columns_to_save]
            split_df_to_save.to_csv(split_output_csv_path, index=False, encoding="utf-8")
            logger.info(f"Saved {split} annotation data to: {split_output_csv_path} ({len(split_df_to_save)} rows)")
        except Exception as e:
            logger.error(f"Error saving {split} annotation CSV to {split_output_csv_path}: {e}")
            # CSV保存失敗は問題なので、このsplitの処理を中断する方が良いかもしれない
            continue  # 次のsplitへ

        # Copy column images
        logger.info(f"Copying {len(split_df)} column images for {split} split...")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copying {split} images"):
            src_path = row["column_image_path_obj"]  # Pathオブジェクト (data/processed/column_images/...)

            # コピー元パス(src_path)から 'data/processed/column_images/' を除いた相対パスを取得
            try:
                # src_path は Path オブジェクトなのでそのまま relative_to を使う
                # 基準となるパスも Path オブジェクトにする
                base_path_to_remove = Path("data/processed/column_images")
                relative_path_for_dest = src_path.relative_to(base_path_to_remove)
            except ValueError:
                # src_path が予期せず 'data/processed/column_images' で始まらない場合
                logger.warning(
                    f"Could not determine relative path for destination using source path {src_path}. Skipping copy."
                )
                skipped_counts[split] += 1
                continue  # 次の画像へ

            # 出力ディレクトリと相対パスを結合して最終的なコピー先パスを作成
            dst_path = split_output_image_dir / relative_path_for_dest

            try:
                # 保存先ディレクトリが存在しない場合は作成
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)  # copy2 preserves metadata
                processed_counts[split] += 1
            except FileNotFoundError:  # 基本的にここには来ないはず (事前にチェックしているため)
                logger.warning(f"Source image not found unexpectedly: {src_path}. Skipping copy.")
                skipped_counts[split] += 1
            except Exception as e:
                logger.error(f"Failed to copy {src_path} to {dst_path}. Error: {e}")
                skipped_counts[split] += 1

    # --- Final Summary ---
    logger.info("\nDataset preparation summary:")
    for split in ["train", "val", "test"]:
        logger.info(f"{split.capitalize()} images copied: {processed_counts[split]}, skipped: {skipped_counts[split]}")
    logger.info(f"Column dataset saved to: {output_path.resolve()}")
    if unicode_map_path.exists():
        logger.info(f"Unicode map saved to: {unicode_map_path.resolve()}")
    else:
        logger.warning("Unicode map file was not created.")


if __name__ == "__main__":
    # --- Configuration ---
    COLUMN_INFO_FILE = "data/processed/column_info.csv"
    #COLUMN_INFO_FILE = "data/processed/unmatched_results.csv"
    # column_images ディレクトリのパスを指定
    COLUMN_IMAGE_DIR = "data/processed/column_images"
    # 出力ディレクトリ名を指定
    OUTPUT_DATASET_DIR = "data/column_dataset"
    # 書籍数 (prepare_yolo_dataset.py と同じ値をデフォルトにする)
    TRAIN_DOCS_COUNT = 31
    VAL_DOCS_COUNT = 7
    TEST_DOCS_COUNT = 6
    RANDOM_SEED = 42
    # -------------

    logger.info("Starting column dataset preparation (split by book)...")
    prepare_column_dataset(
        column_info_file=COLUMN_INFO_FILE,
        column_image_dir=COLUMN_IMAGE_DIR,
        output_dir=OUTPUT_DATASET_DIR,
        train_docs_count=TRAIN_DOCS_COUNT,
        val_docs_count=VAL_DOCS_COUNT,
        test_docs_count=TEST_DOCS_COUNT,
        random_state=RANDOM_SEED,
    )
    logger.info("Column dataset preparation finished.")
