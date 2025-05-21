"""
画像を指定された幅にリサイズし、背景色でパディングを追加するスクリプト
- 画像の平均色を計算し、その色でパディングを追加
- 画像のアスペクト比を維持
- 指定された幅にリサイズ
- メモリ上で処理 (一時ファイル不使用)
- concurrent.futures を用いた並列処理を導入
"""

import concurrent.futures
import os
import shutil
import sys
from functools import partial

import cv2
import numpy as np
import yaml  # 追加: YAMLファイル読み込み用
from PIL import Image

# --- 設定 ---
INPUT_DIR = "data/column_dataset"  # 画像が格納されているディレクトリ
OUTPUT_DIR = "data/column_dataset_padded"  # 出力先ディレクトリ
TARGET_WIDTH = 64
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")  # 処理対象の拡張子
# NOT_USE_YAML_PATH = "data/raw/not_use_data.yaml"

# --- 関数定義 ---


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
            # このケースは実際には稀だが、念のためNoneを返すかデフォルト色を返す
            return (255, 255, 255)  # デフォルト白
        gray = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        background_mask = mask == 255
        if np.sum(background_mask) == 0:
            # print(f"Warning: No background pixels found in {original_image_path}", file=sys.stderr) # 詳細すぎる場合がある
            return calculate_average_color(pil_image)  # 全体の平均色をフォールバック
        background_pixels = img_bgr_np[background_mask]
        if background_pixels.size == 0:
            # print(f"Warning: No background pixels found (after mask) in {original_image_path}", file=sys.stderr)
            return calculate_average_color(pil_image)  # 全体の平均色をフォールバック
        average_color_bgr = np.mean(background_pixels, axis=0)
        average_color_rgb = tuple(map(int, average_color_bgr[::-1]))
        return average_color_rgb
    except Exception as e:
        print(
            f"Error calculating average background color (Otsu) for {original_image_path}: {e}. Falling back to overall average.",
            file=sys.stderr,
        )
        try:
            return calculate_average_color(pil_image)  # エラー時も全体の平均色
        except Exception as e_fallback:
            print(f"Error calculating fallback average color for {original_image_path}: {e_fallback}", file=sys.stderr)
            return (255, 255, 255)  # 最悪ケースは白


def load_not_use_dirs(yaml_path: str) -> set:
    if not os.path.isfile(yaml_path):
        print(f"Warning: not_use_data.yaml が見つかりません: {yaml_path}")
        return set()
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
        dirs = data.get("dirs", [])
        return set(dirs)


def _process_single_image(
    image_path: str, target_width: int, max_height: int, input_dir: str, output_dir: str
) -> tuple[str, bool, str | None]:
    """
    単一の画像を処理するワーカー関数。
    Returns: (画像パス, 成功フラグ, エラーメッセージ or None)
    """
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size

            avg_color = calculate_average_background_color_otsu(img, original_image_path=image_path)
            # calculate_average_background_color_otsu はフォールバックにより None を返さない想定

            aspect_ratio = original_height / original_width
            new_height = int(target_width * aspect_ratio)
            # LANCZOSは高品質だが処理が重い場合がある。速度優先ならANTIALIASやBICUBICも検討
            resized_img_pil = img.resize((target_width, new_height), Image.Resampling.LANCZOS)

            final_size = (target_width, max_height)
            background = Image.new("RGB", final_size, avg_color)

            if resized_img_pil.mode == "RGBA":
                background.paste(resized_img_pil, (0, 0), resized_img_pil)
            else:
                background.paste(resized_img_pil, (0, 0))

            relative_path = os.path.relpath(image_path, input_dir)

            # --- 出力パス変更ロジック START ---
            filename = os.path.basename(relative_path)
            parent_dir = os.path.dirname(relative_path)

            if parent_dir and parent_dir != ".":  # 親ディレクトリが存在し、カレントディレクトリでない場合
                grandparent_dir = os.path.dirname(parent_dir)
                # grandparent_dir が空文字列になるのは、parent_dir がルート直下のディレクトリの場合
                # 例: relative_path = "subdir1/file.jpg" -> parent_dir = "subdir1", grandparent_dir = ""
                # この場合、出力は OUTPUT_DIR/file.jpg となるべき
                # relative_path = "subdir1/subdir2/file.jpg" -> parent_dir = "subdir1/subdir2", grandparent_dir = "subdir1"
                # この場合、出力は OUTPUT_DIR/subdir1/file.jpg となるべき
                if grandparent_dir:  # grandparent_dir が空でない場合のみ結合
                    modified_relative_path = os.path.join(grandparent_dir, filename)
                else:  # grandparent_dir が空 = 元のパスが input_dir 直下より1階層深いだけだった場合
                    modified_relative_path = filename  # ファイル名のみ（OUTPUT_DIR 直下に配置）
            else:  # 親ディレクトリがない、またはカレントディレクトリの場合 (例: input_dir 直下のファイル)
                modified_relative_path = filename
            # --- 出力パス変更ロジック END ---

            output_path = os.path.join(output_dir, modified_relative_path)  # 変更！
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            output_format_str = os.path.splitext(image_path)[1].lower()
            pil_output_format = Image.registered_extensions().get(output_format_str)

            if pil_output_format:
                background.save(output_path, format=pil_output_format)
            else:
                output_path_jpeg = os.path.splitext(output_path)[0] + ".jpg"
                background.save(output_path_jpeg, format="JPEG", quality=95)
                # if output_path != output_path_jpeg:
                #     print(f"  情報: 不明な出力フォーマット '{output_format_str}' のためJPEGで保存: {output_path_jpeg}")

            # Ensure images are closed
            resized_img_pil.close()
            background.close()
        return image_path, True, None
    except Exception as e:
        # print(f"Error in _process_single_image for {image_path}: {e}", file=sys.stderr) # For debugging worker errors
        return image_path, False, str(e)


def process_images_parallel(input_dir: str, output_dir: str, target_width: int):
    if not os.path.isdir(input_dir):
        print(f"エラー: 入力ディレクトリ '{input_dir}' が見つかりません。", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"出力先ディレクトリ: '{output_dir}'")

    image_files = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(os.path.join(root, filename))

    if not image_files:
        print("エラー: 指定されたディレクトリ内に処理対象の画像が見つかりませんでした。", file=sys.stderr)
        sys.exit(1)
    print(f"{len(image_files)} 個の画像ファイルが見つかりました。")

    # --- 1. max_height の事前計算 ---
    print("\n--- 最大高さの計算を開始 ---")
    calculated_heights = []
    for img_path_for_height in image_files:  # 変数名を変更して衝突を避ける
        try:
            with Image.open(img_path_for_height) as img_for_height:  # 変数名を変更
                original_width, original_height = img_for_height.size
                if original_width == 0:  # 幅が0の場合はスキップ
                    print(f"警告: ファイル '{img_path_for_height}' の幅が0です。高さ計算をスキップします。", file=sys.stderr)
                    continue
                aspect_ratio = original_height / original_width
                new_height = int(target_width * aspect_ratio)
                calculated_heights.append(new_height)
        except Exception as e:
            print(f"警告: ファイル '{img_path_for_height}' の高さ計算中にエラー: {e}。スキップします。", file=sys.stderr)
            continue

    if not calculated_heights:
        print("エラー: 有効な画像の高さを計算できませんでした。", file=sys.stderr)
        sys.exit(1)

    max_height = max(calculated_heights) if calculated_heights else 16  # 空の場合のデフォルト
    if max_height % 16 != 0:
        max_height += 16 - (max_height % 16)
    print(f"最大高さを16の倍数に調整: {max_height} ピクセル")

    # --- 2. 並列処理による画像処理 ---
    print("\n--- 並列画像処理を開始 ---")
    # CPUコア数に応じてワーカー数を調整 (Noneでos.cpu_count()が使われる)
    # メモリ使用量も考慮し、コア数より少なめに設定することも有効
    # max_workers = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1
    max_workers = None  # デフォルトの動作に任せる (推奨)

    processed_count = 0
    error_count = 0

    # functools.partial を使って固定引数をワーカー関数に渡す
    worker_func_with_args = partial(
        _process_single_image, target_width=target_width, max_height=max_height, input_dir=input_dir, output_dir=output_dir
    )

    # tqdm を使う場合はここでインポートし、executor.map の結果をラップする
    # from tqdm import tqdm
    # results = executor.map(worker_func_with_args, image_files)
    # for img_path, success, error_msg in tqdm(results, total=len(image_files), desc="Processing images"):

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map は結果を投入順に返す
        # future_to_path = {executor.submit(worker_func_with_args, img_file): img_file for img_file in image_files}
        # for future in concurrent.futures.as_completed(future_to_path):
        #     img_path_res = future_to_path[future]
        #     try:
        #         _, success, error_msg = future.result()
        #         # ... (as_completed を使う場合の処理)
        #     except Exception as exc:
        #         # ...

        # map を使う方がシンプル
        results_iterator = executor.map(worker_func_with_args, image_files)

        for i, (img_path_result, success, error_msg) in enumerate(results_iterator):
            # print(f"処理中 ({i+1}/{len(image_files)}): {img_path_result}") # 逐次表示は遅い場合がある
            if (i + 1) % 100 == 0 or (i + 1) == len(image_files):  # 100件ごと、または最後に進捗表示
                print(f"進捗: {i + 1}/{len(image_files)}")

            if success:
                processed_count += 1
            else:
                error_count += 1
                print(f"  エラー: ファイル '{img_path_result}' の処理中に問題: {error_msg}", file=sys.stderr)
    
    # 最後に data/column_dataset/{train/val/test/{train/val/test}_column_info.csvをコピーする
    for split in ["train", "val", "test"]:
        src_path = os.path.join(INPUT_DIR, split, f"{split}_column_info.csv")
        dst_path = os.path.join(OUTPUT_DIR, split, f"{split}_column_info.csv")
        shutil.copy(src_path, dst_path)

    print("\n--- すべての処理が完了しました ---")
    print(f"処理成功: {processed_count} 個")
    print(f"処理失敗: {error_count} 個")
    print(f"最大高さ: {max_height} ピクセル")
    print(f"処理結果は '{output_dir}' に保存されました。")


# --- スクリプトの実行 ---
if __name__ == "__main__":
    # process_images_memory_efficient(INPUT_DIR, OUTPUT_DIR, TARGET_WIDTH) # 旧関数
    process_images_parallel(INPUT_DIR, OUTPUT_DIR, TARGET_WIDTH)
