# coding: utf-8
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

def calculate_mean_std(base_dir: Path):
    """
    指定されたベースディレクトリ以下の '*/images/' パターンに一致する
    ディレクトリ内のすべての画像から、チャネルごとのピクセル値の
    平均と標準偏差を計算します。

    Args:
        base_dir (Path): データセットのベースディレクトリのパス
                         (例: Path("data/raw/dataset"))。
                         この下の '*/images/' 内の画像を検索します。

    Returns:
        tuple: (mean, std) のタプル。
               mean (np.ndarray): チャネルごとの平均値 (例: [R_mean, G_mean, B_mean])
               std (np.ndarray): チャネルごとの標準偏差 (例: [R_std, G_std, B_std])
               None: 画像が見つからなかった場合、または処理中にエラーが発生した場合。
    """
    image_paths = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    # base_dir の直下のサブディレクトリ(*)の中の'images'ディレクトリの中のファイル(*)を検索
    search_pattern = '*/images/*'

    print(f"Searching for images in: {base_dir} using pattern '{search_pattern}'")

    # 1. pathlib.globを使って画像ファイルをリストアップ
    potential_files = list(base_dir.glob(search_pattern))

    for file_path in potential_files:
        # ファイルであり、かつ有効な拡張子を持つかチェック
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            image_paths.append(file_path)

    if not image_paths:
        print(f"Error: No valid image files found matching the pattern '{base_dir / search_pattern}'.")
        print("Please ensure the directory structure is like 'data/raw/dataset/some_subdir/images/image.jpg'")
        return None

    print(f"Found {len(image_paths)} images. Calculating statistics...")

    # 累算用の変数を初期化 (高精度のためfloat64を使用)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0  # 全画像の総ピクセル数 (height * width)

    # 2. 各画像を処理して統計情報を累算
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # 画像を開き、RGBに変換 (Pathオブジェクトを渡せる)
            img = Image.open(img_path).convert('RGB')
            # NumPy配列に変換し、値を[0, 1]の範囲に正規化 (float32で十分)
            img_np = np.array(img, dtype=np.float32) / 255.0

            # 予期せぬ形状の画像をスキップ (例: グレースケール画像が紛れ込んでいる場合など)
            if img_np.ndim != 3 or img_np.shape[2] != 3:
                 print(f"Warning: Skipping image with unexpected shape {img_np.shape}: {img_path}", file=sys.stderr)
                 continue

            h, w, c = img_np.shape
            current_pixels = h * w

            # チャネルごとの合計値を加算
            channel_sum += np.sum(img_np, axis=(0, 1))
            # チャネルごとの二乗和を加算
            channel_sum_sq += np.sum(img_np**2, axis=(0, 1))
            # 総ピクセル数を加算
            pixel_count += current_pixels

        except Exception as e:
            # エラー発生時もPathオブジェクトを文字列に変換して表示
            print(f"\nWarning: Could not process image {str(img_path)}. Error: {e}", file=sys.stderr)

    if pixel_count == 0:
        print("Error: Could not process any pixels from the found images.")
        return None

    # 3. 平均と標準偏差を計算
    mean = channel_sum / pixel_count
    # 分散 = E[X^2] - (E[X])^2
    variance = (channel_sum_sq / pixel_count) - (mean ** 2)

    # 浮動小数点誤差により分散が微小な負の値になる可能性に対処
    variance = np.maximum(variance, 0) # 負の値を0にクリップ

    # 標準偏差 = sqrt(分散)
    std = np.sqrt(variance)

    # ゼロ除算や数値不安定性を防ぐため、非常に小さい標準偏差をクリップ
    std = np.maximum(std, 1e-7)

    return mean, std

# --- メインの実行部分 ---
if __name__ == "__main__":
    # ここに対象のベースディレクトリを指定してください
    # スクリプトは data_directory/*/images/* 内の画像を探します。
    # 例: data_directory = Path("./data/raw/dataset")
    data_directory = Path("data/raw/dataset")

    if not data_directory.is_dir():
        print(f"Error: Base directory not found or is not a directory: {data_directory}")
        sys.exit(1) # エラーで終了

    results = calculate_mean_std(data_directory)

    if results:
        mean, std = results
        print("\n--- Calculation Complete ---")
        # 結果を小数点以下5桁で表示
        np.set_printoptions(precision=5, suppress=True)
        print(f"Calculated Mean (R, G, B): {mean}")
        print(f"Calculated Std Dev (R, G, B): {std}")

        print("\nUsage Example (PyTorch):")
        print("```python")
        print("from torchvision import transforms")
        print(f"mean = {mean.tolist()}") # リスト形式で表示
        print(f"std = {std.tolist()}") # リスト形式で表示
        print("normalize = transforms.Normalize(mean=mean, std=std)")
        print("```")

        print("\nUsage Example (TensorFlow/Keras):")
        print("```python")
        print("import tensorflow as tf")
        print(f"mean = tf.constant({mean.tolist()}, dtype=tf.float32)")
        print(f"std = tf.constant({std.tolist()}, dtype=tf.float32)")
        print("# Assuming input_tensor values are in [0, 1]")
        print("normalized_tensor = (input_tensor - mean) / std")
        print("# Or using tf.keras.layers.Normalization:")
        print("# Note: layers.Normalization expects mean and variance (std**2)")
        print(f"# variance = { (std**2).tolist() }")
        print("# For image data (batch, height, width, channel): axis=[1, 2]")
        print("# layer = tf.keras.layers.Normalization(mean=mean, variance=std**2, axis=[1, 2])")
        print("```")
