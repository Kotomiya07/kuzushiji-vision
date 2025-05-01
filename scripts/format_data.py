import os
import sys
import tempfile
import shutil
from PIL import Image
import numpy as np
import json # 処理情報を保存するため
import cv2
import yaml  # 追加: YAMLファイル読み込み用

# --- 設定 ---
INPUT_DIR = "data/processed"  # 画像が格納されているディレクトリ
OUTPUT_DIR = "data/processed_padded" # 出力先ディレクトリ
TARGET_WIDTH = 192
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff') # 処理対象の拡張子
# 一時ファイルに処理情報を保存する場合のファイル名
TEMP_INFO_FILE = "temp_image_processing_info.json"
# 除外リストのパス
NOT_USE_YAML_PATH = "data/raw/not_use_data.yaml"

# --- 関数定義 ---

def calculate_average_color(image: Image.Image) -> tuple[int, int, int]:
    """画像の平均色を計算する関数"""
    img_rgb = image.convert('RGB')
    np_img = np.array(img_rgb)
    avg_color_float = np.mean(np_img, axis=(0, 1))
    avg_color_int = tuple(avg_color_float.astype(int))
    return avg_color_int

def calculate_average_background_color_otsu(image_path: str) -> tuple[int, int, int] | None:
    """
    大津の二値化で背景・前景を分離し、背景ピクセルのみの平均色（RGB）を返す関数。
    Args:
        image_path (str): 画像ファイルのパス。
    Returns:
        tuple[int, int, int] | None: 背景ピクセルの平均色 (R, G, B)。背景がなければNone。
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error: Could not read image {image_path}", file=sys.stderr)
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 大津の二値化で背景マスク作成
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 背景は255（白）側と仮定
        background_mask = mask == 255
        if np.sum(background_mask) == 0:
            print(f"Warning: No background pixels found in {image_path}", file=sys.stderr)
            return None
        background_pixels = img[background_mask]
        if background_pixels.size == 0:
            print(f"Warning: No background pixels found in {image_path}", file=sys.stderr)
            return None
        # OpenCVはBGRなのでRGBに変換
        average_color_bgr = np.mean(background_pixels, axis=0)
        average_color_rgb = tuple(map(int, average_color_bgr[::-1]))
        return average_color_rgb
    except Exception as e:
        print(f"Error calculating average background color (Otsu) for {image_path}: {e}", file=sys.stderr)
        return None


def load_not_use_dirs(yaml_path: str) -> set:
    """
    YAMLファイルから除外ディレクトリ名リストを読み込む
    """
    if not os.path.isfile(yaml_path):
        print(f"Warning: not_use_data.yaml が見つかりません: {yaml_path}")
        return set()
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        dirs = data.get('dirs', [])
        return set(dirs)

def process_images_memory_efficient(input_dir: str, output_dir: str, target_width: int):
    """
    指定されたディレクトリ内の画像をメモリ効率良く処理するメイン関数
    """
    # 除外ディレクトリリストを読み込み
    not_use_dirs = load_not_use_dirs(NOT_USE_YAML_PATH)

    if not os.path.isdir(input_dir):
        print(f"エラー: 入力ディレクトリ '{input_dir}' が見つかりません。", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"出力先ディレクトリ: '{output_dir}'")

    image_files = []
    # 1. ディレクトリを再帰的に探索し、画像ファイルのパスを収集
    print("画像ファイルを探索中...")
    for root, _, files in os.walk(input_dir):
        # 除外ディレクトリに含まれていればスキップ
        rel_root = os.path.relpath(root, input_dir)
        # ルート直下 or サブディレクトリ名が not_use_dirs に含まれていれば除外
        root_parts = rel_root.split(os.sep)
        if root_parts[0] in not_use_dirs:
            continue
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(os.path.join(root, filename))

    if not image_files:
        print("エラー: 指定されたディレクトリ内に処理対象の画像が見つかりませんでした。", file=sys.stderr)
        sys.exit(1)

    print(f"{len(image_files)} 個の画像ファイルが見つかりました。")

    # --- 1段階目: 情報収集と一時ファイルへのリサイズ画像保存 ---
    print("\n--- 段階1: 情報収集と一時リサイズを開始 ---")
    processed_info_list = []
    max_height = 0

    # 安全な一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"一時ディレクトリを作成: {temp_dir}")

        for i, img_path in enumerate(image_files):
            print(f"段階1 処理中 ({i+1}/{len(image_files)}): {img_path}")
            try:
                with Image.open(img_path) as img:
                    original_width, original_height = img.size

                    # 元画像の平均色を計算 (パディングに使うため元画像から)
                    #avg_color = calculate_average_color(img)
                    avg_color = calculate_average_background_color_otsu(img_path)

                    # アスペクト比を維持して新しい高さを計算
                    aspect_ratio = original_height / original_width
                    new_height = int(target_width * aspect_ratio)

                    # リサイズ
                    resized_img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)

                    # 最大高さを更新
                    max_height = max(max_height, new_height)

                    # 一意な一時ファイル名を生成 (元のファイル名の一部を使うなど)
                    # 注意: ファイル名が衝突しないように、より堅牢な方法が必要な場合がある
                    base_name = os.path.basename(img_path)
                    temp_filename = f"{i}_{base_name}"
                    temp_resized_path = os.path.join(temp_dir, temp_filename)

                    # リサイズ画像を一時ディレクトリに保存 (元のフォーマットを維持するか、PNG等に統一するか選べる)
                    # ここでは元の拡張子を見て判断（より堅牢にするなら常にPNGが良いかも）
                    img_format = Image.registered_extensions().get(os.path.splitext(img_path)[1].lower())
                    if img_format:
                         # Pillowがサポートするフォーマットで保存
                        resized_img.save(temp_resized_path, format=img_format)
                    else:
                         # 不明な場合はPNGとして保存
                         temp_resized_path += ".png" # 拡張子を追加
                         resized_img.save(temp_resized_path, format='PNG')


                    # 処理情報をリストに追加 (メモリ上の画像オブジェクトは保持しない)
                    processed_info_list.append({
                        'original_path': img_path,
                        'temp_resized_path': temp_resized_path, # 一時ファイルのパスを記録
                        'avg_color': avg_color,
                        'new_height': new_height
                    })

                    # メモリ解放のため閉じる
                    resized_img.close()

            except Exception as e:
                print(f"  エラー (段階1): ファイル '{img_path}' の処理中に問題: {e}", file=sys.stderr)
                # エラーが発生したファイルはスキップ
                continue

        if not processed_info_list:
            print("エラー: 有効な画像を処理できませんでした。", file=sys.stderr)
            # 一時ディレクトリは with ブロック終了時に自動削除される
            sys.exit(1)

        print(f"\n--- 段階1 完了。最大高さ: {max_height} ピクセル ---")
        print("--- 段階2: パディングと最終保存を開始 ---")

        # --- 2段階目: パディングと最終保存 ---
        for i, info in enumerate(processed_info_list):
            original_path = info['original_path']
            temp_resized_path = info['temp_resized_path']
            avg_color = info['avg_color']
            current_height = info['new_height']

            print(f"段階2 処理中 ({i+1}/{len(processed_info_list)}): {original_path}")

            try:
                # 一時保存されたリサイズ済み画像を読み込む
                with Image.open(temp_resized_path) as resized_img:
                    # 最終的な出力画像のサイズ
                    final_size = (target_width, max_height)

                    # 平均色で塗りつぶした背景画像を作成
                    background = Image.new('RGB', final_size, avg_color)

                    # リサイズした画像を背景の上部 (0, 0) に貼り付け
                    # RGBA画像の場合、透過部分を考慮して貼り付け
                    if resized_img.mode == 'RGBA':
                         background.paste(resized_img, (0, 0), resized_img) # マスクとして自身を指定
                    else:
                         background.paste(resized_img, (0, 0))

                    # 出力パスを構築 (入力ディレクトリ構造を維持)
                    relative_path = os.path.relpath(original_path, input_dir)
                    output_path = os.path.join(output_dir, relative_path)

                    # 出力先のサブディレクトリが存在しない場合は作成
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # 画像を保存 (元のフォーマットを引き継ぐか、統一するか)
                    # ここでは元の拡張子から判断して保存
                    output_format = Image.registered_extensions().get(os.path.splitext(original_path)[1].lower())
                    if output_format:
                         background.save(output_path, format=output_format)
                    else:
                         # 不明な場合はJPEGとして保存 (品質指定可能)
                         # background = background.convert('RGB') # JPEGはRGBのみ
                         background.save(output_path, format='JPEG', quality=95)


                    # メモリ解放のため閉じる
                    background.close()

            except FileNotFoundError:
                 print(f"  エラー (段階2): 一時ファイルが見つかりません: {temp_resized_path}", file=sys.stderr)
            except Exception as e:
                 print(f"  エラー (段階2): ファイル '{original_path}' (出力先: {output_path}) の処理中に問題: {e}", file=sys.stderr)

        print("\n--- 段階2 完了 ---")

        # column_info.csvファイルのcolumn_image列にあるpathをprocessedからprocessed_paddedに変更してコピーする
        # column_info.csvファイルのcolumn_image列にあるpathをprocessedからprocessed_paddedに変更してコピー
        column_info_path = os.path.join(input_dir, 'column_info.csv')
        output_column_info_path = os.path.join(output_dir, 'column_info.csv')
        with open(column_info_path, 'r') as f:
            with open(output_column_info_path, 'w') as fo:
                for line in f:
                    line = line.replace('processed', 'processed_padded')
                    fo.write(line)

        # 一時ディレクトリとその内容は with ブロックを抜ける際に自動的に削除されます
        print(f"一時ディレクトリ {temp_dir} をクリーンアップしました。")

    print("\n--- すべての処理が完了しました ---")
    print(f"\n--- 最大高さ: {max_height} ピクセル ---")
    print(f"処理結果は '{output_dir}' に保存されました。")


# --- スクリプトの実行 ---
if __name__ == "__main__":
    process_images_memory_efficient(INPUT_DIR, OUTPUT_DIR, TARGET_WIDTH)
