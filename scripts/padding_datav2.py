"""
画像を指定された幅にリサイズし、背景色でパディングを追加するスクリプト
- 画像の平均色を計算し、その色でパディングを追加
- 画像のアスペクト比を維持
- 指定された幅にリサイズ
- メモリ上で処理 (一時ファイル不使用)
- concurrent.futures を用いた並列処理を導入
- バウンディングボックス情報も同時に処理し、スケーリングしてJSONとして保存
"""

import concurrent.futures
import json # JSONを扱うために追加
import os
import shutil
import sys
from functools import partial

import cv2
import numpy as np
import pandas as pd # pandas を使うために追加
import yaml
from PIL import Image

# --- 設定 ---
BASE_INPUT_DIR = "data/column_dataset"  # 画像が格納されているベースディレクトリ
BASE_OUTPUT_DIR = "data/column_dataset_padded"  # 出力先ベースディレクトリ
TARGET_WIDTH = 64
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")  # 処理対象の拡張子
# NOT_USE_YAML_PATH = "data/raw/not_use_data.yaml"

# --- 関数定義 ---

def ensure_dir(directory):
    """ディレクトリが存在しない場合は作成する"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        # print(f"ディレクトリを作成しました: {directory}") # 大量に呼ばれるためコメントアウト


def calculate_average_color(image: Image.Image) -> tuple[int, int, int]:
    """画像の平均色を計算する関数"""
    img_rgb = image.convert("RGB")
    np_img = np.array(img_rgb)
    avg_color_float = np.mean(np_img, axis=(0, 1))
    avg_color_int = tuple(avg_color_float.astype(int))
    return avg_color_int


def calculate_average_background_color_otsu(pil_image: Image.Image, original_image_path: str) -> tuple[int, int, int] | None:
    """
    Pillowイメージオブジェクトを受け取り、大津の二値化で背景・前景を分離し、
    背景ピクセルのみの平均色（RGB）を返す関数。
    """
    try:
        img_rgb_np = np.array(pil_image.convert("RGB"))
        img_bgr_np = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2BGR)
        if img_bgr_np is None:
            print(f"Error: Could not convert PIL image to OpenCV format for {original_image_path}", file=sys.stderr)
            return (255, 255, 255)
        gray = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        background_mask = mask == 255
        if np.sum(background_mask) == 0:
            return calculate_average_color(pil_image)
        background_pixels = img_bgr_np[background_mask]
        if background_pixels.size == 0:
            return calculate_average_color(pil_image)
        average_color_bgr = np.mean(background_pixels, axis=0)
        average_color_rgb = tuple(map(int, average_color_bgr[::-1]))
        return average_color_rgb
    except Exception as e:
        print(
            f"Error calculating average background color (Otsu) for {original_image_path}: {e}. Falling back to overall average.",
            file=sys.stderr,
        )
        try:
            return calculate_average_color(pil_image)
        except Exception as e_fallback:
            print(f"Error calculating fallback average color for {original_image_path}: {e_fallback}", file=sys.stderr)
            return (255, 255, 255)


def load_not_use_dirs(yaml_path: str) -> set:
    if not os.path.isfile(yaml_path):
        print(f"Warning: not_use_data.yaml が見つかりません: {yaml_path}")
        return set()
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
        dirs = data.get("dirs", [])
        return set(dirs)


def _process_single_image(
    image_path: str,
    target_width: int,
    max_height: int,
    input_dir_for_relpath: str, 
    output_dir_for_image: str, 
    bbox_data_map_for_split: dict
) -> tuple[str, bool, str | None]:
    """
    単一の画像を処理するワーカー関数。バウンディングボックス処理も含む。
    Returns: (画像パス, 成功フラグ, エラーメッセージ or None)
    """
    image_processed_successfully = False
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size

            avg_color = calculate_average_background_color_otsu(img, original_image_path=image_path)

            aspect_ratio = original_height / original_width
            new_height_float = target_width * aspect_ratio
            new_height = int(new_height_float)
            
            resized_img_pil = img.resize((target_width, new_height), Image.Resampling.LANCZOS)

            final_size = (target_width, max_height)
            background = Image.new("RGB", final_size, avg_color)

            paste_position = (0, 0) 
            if resized_img_pil.mode == "RGBA":
                background.paste(resized_img_pil, paste_position, resized_img_pil)
            else:
                background.paste(resized_img_pil, paste_position)

            # --- 出力パス変更ロジック (1段浅くする) START ---
            relative_path_from_split_input = os.path.relpath(image_path, input_dir_for_relpath)
            filename_only = os.path.basename(relative_path_from_split_input)
            parent_dir_of_image_in_split = os.path.dirname(relative_path_from_split_input)

            if parent_dir_of_image_in_split and parent_dir_of_image_in_split != ".":
                grandparent_dir_of_image_in_split = os.path.dirname(parent_dir_of_image_in_split)
                if grandparent_dir_of_image_in_split: 
                    modified_relative_path_for_output = os.path.join(grandparent_dir_of_image_in_split, filename_only)
                else: 
                    modified_relative_path_for_output = filename_only
            else: 
                modified_relative_path_for_output = filename_only
            
            output_image_path = os.path.join(output_dir_for_image, modified_relative_path_for_output)
            # --- 出力パス変更ロジック END ---
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)


            output_format_str = os.path.splitext(image_path)[1].lower()
            pil_output_format = Image.registered_extensions().get(output_format_str)

            if pil_output_format:
                background.save(output_image_path, format=pil_output_format)
            else:
                output_path_jpeg = os.path.splitext(output_image_path)[0] + ".jpg"
                background.save(output_path_jpeg, format="JPEG", quality=95)

            resized_img_pil.close()
            background.close()
            image_processed_successfully = True

        # --- バウンディングボックス処理 ---
        image_name_without_ext = os.path.splitext(os.path.basename(image_path))[0]
        char_boxes_str = bbox_data_map_for_split.get(image_name_without_ext)

        if char_boxes_str is not None and not pd.isna(char_boxes_str):
            try:
                bounding_boxes_original = eval(char_boxes_str)
                if not isinstance(bounding_boxes_original, list):
                    raise ValueError("Bounding box data is not a list.")

                adjusted_bounding_boxes = []
                if original_width > 0 and original_height > 0 : 
                    scale_x = target_width / original_width
                    scale_y = new_height_float / original_height 

                    for box in bounding_boxes_original:
                        if len(box) == 4:
                            x_min, y_min, x_max, y_max = box
                            adj_x_min = int(x_min * scale_x + paste_position[0])
                            adj_y_min = int(y_min * scale_y + paste_position[1])
                            adj_x_max = int(x_max * scale_x + paste_position[0])
                            adj_y_max = int(y_max * scale_y + paste_position[1])
                            adjusted_bounding_boxes.append([adj_x_min, adj_y_min, adj_x_max, adj_y_max])
                        else:
                            print(f"Warning: Invalid box format for {image_name_without_ext}: {box}. Skipping this box.", file=sys.stderr)
                else:
                    print(f"Warning: Original width or height is zero for {image_name_without_ext}. Cannot scale bounding boxes.", file=sys.stderr)
                    adjusted_bounding_boxes = bounding_boxes_original 

                book_id = image_name_without_ext.split("_")[0]
                json_output_base_dir = os.path.join(output_dir_for_image, "bounding_boxes")
                book_output_dir_for_json = os.path.join(json_output_base_dir, book_id)
                ensure_dir(book_output_dir_for_json)

                output_json_path = os.path.join(book_output_dir_for_json, f"{image_name_without_ext}.json")
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(adjusted_bounding_boxes, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"Error processing bounding boxes for {image_name_without_ext}: {e}. Skipping bbox for this image.", file=sys.stderr)
        
        return image_path, image_processed_successfully, None
    except Exception as e:
        return image_path, False, str(e)


def process_images_parallel(input_dir_split: str, output_dir_split: str, target_width: int, bbox_data_map_for_split: dict):
    if not os.path.isdir(input_dir_split):
        print(f"エラー: 入力ディレクトリ '{input_dir_split}' が見つかりません。", file=sys.stderr)
        return 

    os.makedirs(output_dir_split, exist_ok=True) 
    print(f"出力先ディレクトリ (スプリット): '{output_dir_split}'")

    image_files = []
    for root, _, files in os.walk(input_dir_split):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(os.path.join(root, filename))

    if not image_files:
        print(f"情報: ディレクトリ '{input_dir_split}' 内に処理対象の画像が見つかりませんでした。")
        return 
    print(f"{len(image_files)} 個の画像ファイルが見つかりました。")

    print("\n--- 最大高さの計算を開始 ---")
    calculated_heights = []
    for img_path_for_height in image_files:
        try:
            with Image.open(img_path_for_height) as img_for_height:
                original_width, original_height = img_for_height.size
                if original_width == 0:
                    print(f"警告: ファイル '{img_path_for_height}' の幅が0です。高さ計算をスキップします。", file=sys.stderr)
                    continue
                aspect_ratio = original_height / original_width
                new_height = int(target_width * aspect_ratio)
                calculated_heights.append(new_height)
        except Exception as e:
            print(f"警告: ファイル '{img_path_for_height}' の高さ計算中にエラー: {e}。スキップします。", file=sys.stderr)
            continue

    if not calculated_heights:
        print("エラー: 有効な画像の高さを計算できませんでした。このスプリットの処理を中止します。", file=sys.stderr)
        return 

    max_height = max(calculated_heights) if calculated_heights else 16
    if max_height % 16 != 0:
        max_height += 16 - (max_height % 16)
    print(f"最大高さを16の倍数に調整: {max_height} ピクセル")

    print("\n--- 並列画像処理を開始 ---")
    max_workers = None

    processed_count = 0
    error_count = 0

    worker_func_with_args = partial(
        _process_single_image,
        target_width=target_width,
        max_height=max_height,
        input_dir_for_relpath=input_dir_split, 
        output_dir_for_image=output_dir_split, 
        bbox_data_map_for_split=bbox_data_map_for_split
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(worker_func_with_args, image_files)

        for i, (img_path_result, success, error_msg) in enumerate(results_iterator):
            if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                print(f"進捗 ({input_dir_split}): {i + 1}/{len(image_files)}")

            if success:
                processed_count += 1
            else:
                error_count += 1
                print(f"  エラー: ファイル '{img_path_result}' の処理中に問題: {error_msg}", file=sys.stderr)
    
    print(f"\n--- スプリット '{input_dir_split}' の処理が完了しました ---")
    print(f"処理成功: {processed_count} 個")
    print(f"処理失敗: {error_count} 個")


def unicode_to_char(unicode_str):
    """Unicode ID（例: 'U+601D'）から文字に変換する"""
    # 'U+'を削除して16進数に変換
    try:
        code_point = int(unicode_str.replace("U+", ""), 16)
        return chr(code_point)
    except:
        print(f"変換エラー: {unicode_str}")
        return ""


def process_dataset(dataset_type):
    """指定されたデータセット（train/val/test）のラベルを処理する"""
    base_dir = f"data/column_dataset_padded/{dataset_type}"
    csv_path = f"{base_dir}/{dataset_type}_column_info.csv"
    labels_dir = f"{base_dir}/labels"

    # ラベルディレクトリを作成
    ensure_dir(labels_dir)

    # CSVファイルを読み込む
    df = pd.read_csv(csv_path)

    print(f"{dataset_type}データセットの処理を開始: {len(df)}行")

    for idx, row in df.iterrows():
        # 画像ファイル名を取得（拡張子を除く）
        image_path = row["column_image"]
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # book_id を image_name から抽出 (例: 100249416_00002_2_column_001 -> 100249416)
        book_id = image_name.split("_")[0]

        # book_id を含む新しいラベル出力ディレクトリパスを作成
        book_labels_dir = os.path.join(labels_dir, book_id)
        ensure_dir(book_labels_dir)

        # Unicode IDのリストを取得して文字に変換
        unicode_ids = eval(row["unicode_ids"])  # 文字列リストを実際のリストに変換
        characters = [unicode_to_char(uid) for uid in unicode_ids]
        text = "".join(characters)

        # ラベルファイルに保存
        label_path = os.path.join(book_labels_dir, f"{image_name}.txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(text)

        # 進捗表示
        if (idx + 1) % 100 == 0:
            print(f"{idx + 1}/{len(df)} 処理完了")

    print(f"{dataset_type}データセットの処理が完了しました")


def main():
    """メイン処理"""
    print("パディング処理とバウンディングボックス情報作成処理を開始します")

    for split in ["train", "val", "test"]:
        print(f"\n--- Processing {split} dataset ---")
        input_split_dir = os.path.join(BASE_INPUT_DIR, split)
        output_split_dir = os.path.join(BASE_OUTPUT_DIR, split)

        if not os.path.isdir(input_split_dir):
            print(f"警告: 入力ディレクトリが見つかりません: {input_split_dir}。このデータセットをスキップします。", file=sys.stderr)
            continue

        os.makedirs(output_split_dir, exist_ok=True)

        csv_path = os.path.join(input_split_dir, f"{split}_column_info.csv")
        bbox_data_map = {}
        if os.path.exists(csv_path):
            try:
                df_bounding_boxes = pd.read_csv(csv_path)
                df_bounding_boxes.dropna(subset=['char_boxes_in_column'], inplace=True)
                bbox_data_map = pd.Series(
                    df_bounding_boxes.char_boxes_in_column.values,
                    index=df_bounding_boxes.column_image.apply(lambda x: os.path.splitext(os.path.basename(x))[0])
                ).to_dict()
                print(f"{len(bbox_data_map)} 件のバウンディングボックス情報を {csv_path} から読み込みました")
            except Exception as e:
                print(f"エラー: CSVファイル {csv_path} の読み込みまたは解析中にエラーが発生しました: {e}。このデータセットのバウンディングボックス処理はスキップされます。", file=sys.stderr)
                bbox_data_map = {} 
        else:
            print(f"警告: バウンディングボックスCSVファイルが見つかりません: {csv_path}。このデータセットのバウンディングボックス処理はスキップされます。", file=sys.stderr)

        process_images_parallel(input_split_dir, output_split_dir, TARGET_WIDTH, bbox_data_map)

    print("\n--- 元の _column_info.csv ファイルのコピー処理を開始 ---")
    for split in ["train", "val", "test"]:
        src_csv_path = os.path.join(BASE_INPUT_DIR, split, f"{split}_column_info.csv")
        dst_split_dir = os.path.join(BASE_OUTPUT_DIR, split) 
        
        dst_csv_path = os.path.join(dst_split_dir, f"{split}_column_info.csv")
        
        if os.path.exists(src_csv_path):
            try:
                shutil.copy(src_csv_path, dst_csv_path)
                print(f"コピー完了: {src_csv_path} -> {dst_csv_path}")
            except Exception as e:
                print(f"エラー: CSVファイル {src_csv_path} のコピー中にエラーが発生しました ({dst_csv_path}): {e}", file=sys.stderr)
        else:
            print(f"警告: コピー元のCSVファイルが見つかりません: {src_csv_path}。コピーをスキップします。", file=sys.stderr)

    print("\nすべてのバウンディングボックス情報作成処理が完了しました")

    print("ラベル作成処理を開始します")

    # 各データセットを処理
    for dataset_type in ["train", "val", "test"]:
        # 対応するCSVファイルが存在するか確認
        csv_path = f"data/column_dataset_padded/{dataset_type}/{dataset_type}_column_info.csv"
        if os.path.exists(csv_path):
            process_dataset(dataset_type)
        else:
            print(f"{csv_path} が見つかりません。スキップします。")

    print("すべての処理が完了しました")


if __name__ == "__main__":
    main()
