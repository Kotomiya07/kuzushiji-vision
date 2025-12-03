"""
列アノテーションと文字アノテーションを突き合わせて1行の画像と文字列をペアにしたデータセットを作成するスクリプト

データのカラム:
- image: data/kuzushiji-columnのアノテーションを使用して切り出した1行の画像
- text: 該当する部分の文字列（2つのアノテーションを突き合わせて作成）

データの形式: parquet形式で1shardファイルが300MB未満になるように分割
"""

import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# 設定
COLUMN_ANNOTATION_DIR = Path("data/kuzushiji-column")
CHAR_ANNOTATION_DIR = Path("datasets/raw/dataset")
OUTPUT_DIR = Path("data/line_dataset")
MAX_SHARD_SIZE_MB = 300


def extract_book_id_and_image_name(image_path: Path) -> tuple[str | None, str | None]:
    """画像パスから書籍IDと画像名を抽出

    Args:
        image_path: 画像ファイルのパス

    Returns:
        (book_id, image_name) のタプル。抽出できない場合は (None, None)
    """
    # 例: 200014685_00026_2_jpg.rf.6624a5b5f04e99f65cc0418eed7a75c6.jpg
    # -> book_id: 200014685, image_name: 200014685_00026_2
    stem = image_path.stem
    # .rf. より前の部分を取得
    match = re.match(r"^(\d+)_(\d+_\d+).*", stem)
    if match:
        book_id = match.group(1)
        image_name = f"{book_id}_{match.group(2)}"
        return book_id, image_name
    return None, None


def load_yolo_annotation(label_path: Path, image_width: int, image_height: int) -> list[tuple[float, float, float, float]]:
    """YOLO形式のアノテーションファイルを読み込み、列のバウンディングボックスを返す

    Args:
        label_path: ラベルファイルのパス
        image_width: 画像の幅
        image_height: 画像の高さ

    Returns:
        列のバウンディングボックスのリスト [(x1, y1, x2, y2), ...]
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                # YOLO形式: class_id center_x center_y width height (正規化座標)
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # 絶対座標に変換
                x1 = (center_x - width / 2) * image_width
                y1 = (center_y - height / 2) * image_height
                x2 = (center_x + width / 2) * image_width
                y2 = (center_y + height / 2) * image_height

                boxes.append((x1, y1, x2, y2))

    return boxes


def load_character_annotations(book_id: str, image_name: str) -> pd.DataFrame:
    """文字アノテーションファイルを読み込む

    Args:
        book_id: 書籍ID
        image_name: 画像名（拡張子なし）

    Returns:
        文字アノテーションのDataFrame
    """
    coord_file = CHAR_ANNOTATION_DIR / book_id / f"{book_id}_coordinate.csv"
    if not coord_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(coord_file)
    # 画像名が一致する行のみ抽出
    df = df[df["Image"] == image_name]
    return df


def extract_lines_from_column(
    column_image: Image.Image,
    column_box: tuple[float, float, float, float],
    char_df: pd.DataFrame,
    original_image_size: tuple[int, int],
) -> list[tuple[Image.Image, str]]:
    """列画像から行を抽出し、各行の画像と文字列のペアを返す

    Args:
        column_image: 列画像
        column_box: 列のバウンディングボックス (x1, y1, x2, y2) 元画像座標系
        char_df: 文字アノテーションのDataFrame
        original_image_size: 元画像のサイズ (width, height)

    Returns:
        [(行画像, 文字列), ...] のリスト
    """
    if char_df.empty:
        return []

    # 列のバウンディングボックス内の文字を抽出
    col_x1, col_y1, col_x2, col_y2 = column_box

    # 列内の文字をフィルタリング
    char_in_column = []
    for _, row in char_df.iterrows():
        x = float(row["X"])
        y = float(row["Y"])
        w = float(row["Width"])
        h = float(row["Height"])

        # 文字の中心が列内にあるかチェック
        char_center_x = x + w / 2
        char_center_y = y + h / 2

        if col_x1 <= char_center_x <= col_x2 and col_y1 <= char_center_y <= col_y2:
            # 列画像内の相対座標に変換
            rel_x = x - col_x1
            rel_y = y - col_y1
            char_id = str(row.get("Char ID", ""))
            char_in_column.append(
                {
                    "x": rel_x,
                    "y": rel_y,
                    "width": w,
                    "height": h,
                    "unicode": str(row["Unicode"]),
                    "char_id": char_id,
                }
            )

    if not char_in_column:
        return []

    # Y座標でソート
    char_in_column.sort(key=lambda c: c["y"])

    # 行に分割（Y座標のギャップで判定）
    lines = []
    current_line = [char_in_column[0]]

    # 文字高の中央値を計算
    heights = [c["height"] for c in char_in_column if c["height"] > 0]
    median_height = np.median(heights) if heights else 50
    gap_threshold = median_height * 1.5  # ギャップ閾値

    for i in range(1, len(char_in_column)):
        char_above = char_in_column[i - 1]
        char_below = char_in_column[i]

        # 垂直ギャップを計算
        gap = char_below["y"] - (char_above["y"] + char_above["height"])

        if gap > gap_threshold:
            # 新しい行を開始
            lines.append(current_line)
            current_line = [char_below]
        else:
            # 同じ行に追加
            current_line.append(char_below)

    # 最後の行を追加
    if current_line:
        lines.append(current_line)

    # 各行の画像と文字列を生成
    result = []
    for line_chars in lines:
        if not line_chars:
            continue

        # 行のバウンディングボックスを計算
        line_y1 = min(c["y"] for c in line_chars)
        line_y2 = max(c["y"] + c["height"] for c in line_chars)

        # マージンを追加
        margin = 5
        line_y1 = max(0, line_y1 - margin)
        line_y2 = min(column_image.height, line_y2 + margin)

        # 行画像を切り出し
        line_image = column_image.crop((0, int(line_y1), column_image.width, int(line_y2)))

        # 文字列を生成（Char IDでソート）
        def extract_char_id_number(char_info: dict) -> int:
            """Char IDから数値部分を抽出して返す（例: C0001 -> 1）"""
            char_id = char_info.get("char_id", "")
            if not char_id:
                return 0
            # C0001のような形式から数値部分を抽出
            match = re.search(r"\d+", char_id)
            if match:
                return int(match.group())
            return 0

        line_chars_sorted = sorted(line_chars, key=extract_char_id_number)
        text_parts = []
        for char_info in line_chars_sorted:
            unicode_str = char_info["unicode"]
            if unicode_str.startswith("U+"):
                try:
                    char_code = int(unicode_str[2:], 16)
                    text_parts.append(chr(char_code))
                except ValueError:
                    pass

        text = "".join(text_parts)

        if text:  # 空文字列でない場合のみ追加
            result.append((line_image, text))

    return result


def image_to_bytes(image: Image.Image) -> bytes:
    """PIL画像をバイト列に変換"""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def estimate_row_size(row: dict[str, Any]) -> int:
    """行のサイズを推定（バイト単位）"""
    size = 0
    if "image" in row:
        size += len(row["image"])
    if "text" in row:
        size += len(row["text"].encode("utf-8"))
    return size


def process_dataset(split: str = "train") -> None:
    """データセットを処理してparquetファイルを作成

    Args:
        split: データセットの分割（train, val, test）
    """
    print(f"Processing {split} dataset...")

    # ディレクトリパス
    image_dir = COLUMN_ANNOTATION_DIR / split / "images"
    label_dir = COLUMN_ANNOTATION_DIR / split / "labels"

    if not image_dir.exists():
        print(f"Warning: Image directory not found: {image_dir}")
        return

    # 画像ファイルを取得
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")

    # データを収集
    all_rows = []
    current_shard_rows = []
    current_shard_size = 0
    shard_index = 0

    for image_path in tqdm(image_files, desc=f"Processing {split}"):
        # 書籍IDと画像名を抽出
        book_id, image_name = extract_book_id_and_image_name(image_path)
        if book_id is None or image_name is None:
            print(f"Warning: Could not extract book_id/image_name from {image_path}")
            continue

        # 画像を読み込み
        try:
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            continue

        # 列アノテーションを読み込み
        label_path = label_dir / f"{image_path.stem}.txt"
        column_boxes = load_yolo_annotation(label_path, image_width, image_height)

        if not column_boxes:
            continue

        # 文字アノテーションを読み込み
        char_df = load_character_annotations(book_id, image_name)

        if char_df.empty:
            continue

        # 各列を処理
        for col_idx, column_box in enumerate(column_boxes):
            # 列画像を切り出し
            x1, y1, x2, y2 = map(int, column_box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_width, x2)
            y2 = min(image_height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            try:
                column_image = image.crop((x1, y1, x2, y2))
            except Exception as e:
                print(f"Warning: Could not crop column from {image_path}: {e}")
                continue

            # 行を抽出
            lines = extract_lines_from_column(column_image, column_box, char_df, (image_width, image_height))

            # 各行をデータに追加
            for line_image, text in lines:
                # 画像をバイト列に変換
                image_bytes = image_to_bytes(line_image)

                row = {
                    "image": image_bytes,
                    "text": text,
                }

                row_size = estimate_row_size(row)

                # 現在のshardに追加
                current_shard_rows.append(row)
                current_shard_size += row_size

                # shardサイズが上限を超えたら保存
                if current_shard_size > MAX_SHARD_SIZE_MB * 1024 * 1024:
                    # shardを保存
                    save_shard(current_shard_rows, split, shard_index)
                    shard_index += 1
                    current_shard_rows = []
                    current_shard_size = 0

    # 残りのデータを保存
    if current_shard_rows:
        save_shard(current_shard_rows, split, shard_index)

    print(f"Completed processing {split} dataset. Created {shard_index + 1} shard(s)")


def save_shard(rows: list[dict[str, Any]], split: str, shard_index: int) -> None:
    """shardをparquetファイルとして保存

    Args:
        rows: データ行のリスト
        split: データセットの分割
        shard_index: shardのインデックス
    """
    if not rows:
        return

    # 出力ディレクトリを作成
    output_split_dir = OUTPUT_DIR / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    # DataFrameを作成
    df = pd.DataFrame(rows)

    # parquetファイルとして保存
    output_path = output_split_dir / f"shard_{shard_index:05d}.parquet"
    df.to_parquet(output_path, index=False, engine="pyarrow")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved shard {shard_index} to {output_path} ({file_size_mb:.2f} MB, {len(rows)} rows)")


def main():
    """メイン処理"""
    print("Starting line dataset creation...")
    print(f"Column annotation directory: {COLUMN_ANNOTATION_DIR}")
    print(f"Character annotation directory: {CHAR_ANNOTATION_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max shard size: {MAX_SHARD_SIZE_MB} MB")

    # 出力ディレクトリを作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 各分割を処理
    for split in ["train", "val", "test"]:
        process_dataset(split)

    print("Dataset creation completed!")


if __name__ == "__main__":
    main()
