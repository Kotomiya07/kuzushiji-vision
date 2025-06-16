"""
TrOCR用の画像リサイズモジュール
アスペクト比を保ったまま幅を64にリサイズし、高さ384ごとに分割して横に並べる
"""

import random
from pathlib import Path

from PIL import Image


def get_random_column_image(image_dir: str | Path = "../data/column_dataset/train/images") -> Path:
    """
    指定ディレクトリから画像をランダムに1枚取得する

    Args:
        image_dir: 画像ディレクトリのパス

    Returns:
        ランダムに選択された画像ファイルのPath

    Raises:
        FileNotFoundError: 画像ディレクトリが存在しない、または画像が見つからない場合
    """
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # 再帰的にjpgファイルを検索
    image_files = list(image_dir.glob("**/*.jpg"))

    if not image_files:
        raise FileNotFoundError(f"No jpg images found in {image_dir}")

    return random.choice(image_files)


def resize_preserve_aspect_ratio(image: Image.Image, target_width: int = 64) -> Image.Image:
    """
    アスペクト比を保ったまま指定幅にリサイズする

    Args:
        image: PIL Imageオブジェクト
        target_width: リサイズ後の幅

    Returns:
        リサイズされたPIL Imageオブジェクト
    """
    original_width, original_height = image.size

    # アスペクト比を保ったまま新しい高さを計算
    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)

    # リサイズを実行
    resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return resized_image


def split_and_arrange_image(image: Image.Image, segment_height: int = 384) -> Image.Image:
    """
    画像を指定高さごとに分割し、90度回転させて上から左から右に格子状に配置する

    Args:
        image: PIL Imageオブジェクト
        segment_height: 分割する高さ

    Returns:
        分割・配置された正方形のPIL Imageオブジェクト（segment_height × segment_height）
    """
    width, height = image.size

    # 分割数を計算
    num_segments = (height + segment_height - 1) // segment_height  # 切り上げ除算

    # 正方形のキャンバスサイズを分割する高さに設定
    canvas_size = segment_height

    # 黒いキャンバスを作成
    canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))

    # 回転後のセグメントサイズを計算（90度回転するとwidth × segment_heightがsegment_height × widthになる）
    rotated_width = segment_height
    rotated_height = width

    # キャンバス内に配置できる行数と列数を計算
    cols_per_row = canvas_size // rotated_width
    rows_available = canvas_size // rotated_height

    if cols_per_row == 0 or rows_available == 0:
        # 回転後のセグメントがキャンバスに収まらない場合
        print(
            f"Warning: Rotated segment ({rotated_width}x{rotated_height}) is too large for canvas ({canvas_size}x{canvas_size})"
        )
        return canvas

    # 各セグメントを処理
    for i in range(min(num_segments, cols_per_row * rows_available)):
        y_start = i * segment_height
        y_end = min(y_start + segment_height, height)

        # セグメントを切り出し
        segment = image.crop((0, y_start, width, y_end))

        # 90度回転（時計回り）
        rotated_segment = segment.rotate(90, expand=True)

        # 格子状の配置位置を計算
        row = i // cols_per_row
        col = i % cols_per_row

        x_position = col * rotated_width
        y_position = row * rotated_height

        # キャンバスに配置
        canvas.paste(rotated_segment, (x_position, y_position))

    return canvas


def process_column_image_for_trocr(
    image_path: str | Path = None, target_width: int = 64, segment_height: int = 384
) -> Image.Image:
    """
    TrOCR用に列画像を処理する関数

    1. 画像をランダムに選択（image_pathが指定されていない場合）
    2. アスペクト比を保ったまま幅64にリサイズ
    3. 高さ384ごとに分割し、90度回転させて上から左から右に格子状に配置
    4. segment_height × segment_heightの正方形画像に変形
    5. 余った部分は黒で塗りつぶし

    Args:
        image_path: 処理する画像のパス（Noneの場合はランダム選択）
        target_width: リサイズ後の幅（デフォルト: 64）
        segment_height: 分割する高さ（デフォルト: 384）

    Returns:
        処理されたsegment_height × segment_heightの正方形PILImageオブジェクト

    Examples:
        >>> # ランダムに画像を選択して処理
        >>> processed_image = process_column_image_for_trocr()

        >>> # 特定の画像を処理
        >>> processed_image = process_column_image_for_trocr("path/to/image.jpg")

        >>> # パラメータを変更
        >>> processed_image = process_column_image_for_trocr(
        ...     target_width=32, segment_height=256
        ... )
    """
    try:
        # 画像の取得
        if image_path is None:
            image_path = get_random_column_image()
            print(f"Using random image: {image_path}")
        else:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

        # 画像の読み込み
        image = Image.open(image_path).convert("RGB")

        # 1. アスペクト比を保ったまま幅をリサイズ
        resized_image = resize_preserve_aspect_ratio(image, target_width)

        # 2. 分割・配置
        final_image = split_and_arrange_image(resized_image, segment_height)

        return final_image

    except Exception as e:
        raise RuntimeError(f"Failed to process image: {e}") from e


if __name__ == "__main__":
    # テスト実行
    try:
        print("Testing image processing function...")

        # ランダム画像での処理テスト
        processed_image = process_column_image_for_trocr()
        print(f"Processed image size: {processed_image.size}")

        # 結果を保存
        output_path = "test_processed_image.png"
        processed_image.save(output_path)
        print(f"Test image saved to: {output_path}")

    except Exception as e:
        print(f"Test failed: {e}")
