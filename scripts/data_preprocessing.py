# import cv2  # cv2 を削除
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image  # Pillow をインポート
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from utils.util import EasyDict


@dataclass
class CharacterBox:
    """文字のバウンディングボックス情報"""

    x1: float
    y1: float
    x2: float
    y2: float
    unicode_id: str

    @property
    def center_y(self) -> float:
        """y座標の中心を取得"""
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> float:
        """高さを取得"""
        return self.y2 - self.y1

    @property
    def width(self) -> float:
        """幅を取得"""
        return self.x2 - self.x1


def detect_text_columns(
    char_boxes: list[CharacterBox], eps_ratio: float = 0.2, min_samples: int = 1
) -> list[list[CharacterBox]]:
    """DBSCANを使用して文字を列ごとにクラスタリング

    Args:
        char_boxes (List[CharacterBox]): 文字のバウンディングボックスのリスト
        eps_ratio (float, optional): 文字の幅の中央値に対するepsの割合. Defaults to 0.2. # デフォルト値を0.5から0.2に変更 (元のコードコメントと合わせる)
        min_samples (int, optional): DBSCANのmin_samples. Defaults to 1.

    Returns:
        List[List[CharacterBox]]: 列ごとにグループ化された文字のリスト
    """
    if not char_boxes:
        return []

    # 各文字の幅の中央値を計算
    widths = [box.width for box in char_boxes]
    # 幅が0のボックスを除外して中央値を計算
    valid_widths = [w for w in widths if w > 0]
    if not valid_widths:  # 有効な幅がない場合、デフォルトのepsを使用
        median_width = 10  # 仮のデフォルト値
    else:
        median_width = np.median(valid_widths)
    eps = median_width * eps_ratio

    # 各文字の中心のx座標を計算
    x_centers = np.array([(box.x1 + box.x2) / 2 for box in char_boxes]).reshape(-1, 1)

    # DBSCANによるクラスタリング
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x_centers)
    labels = clustering.labels_

    # クラスタごとに文字をグループ化
    columns: dict[int, list[CharacterBox]] = {}  # 型ヒントを修正
    noise_label_start = max(labels) + 1 if any(label != -1 for label in labels) else 0  # ノイズラベルの開始番号  # noqa: E741
    noise_count = 0
    for label, box in zip(labels, char_boxes, strict=False):
        current_label = label
        if label == -1:
            # ノイズとして検出された場合は新しいグループを作成
            current_label = noise_label_start + noise_count
            noise_count += 1
        if current_label not in columns:
            columns[current_label] = []
        columns[current_label].append(box)

    # 各列内の文字をy座標でソート（上から下）
    text_columns = []
    # 列が存在しない場合のエラーを防ぐため、キーが存在するか確認
    sorted_labels = sorted([label for label in columns.keys() if columns[label]])
    for label in sorted_labels:
        column = sorted(columns[label], key=lambda box: box.y1)
        text_columns.append(column)

    # 列を左から右にソート (列が空でないことを確認)
    text_columns.sort(key=lambda column: min(box.x1 for box in column) if column else float("inf"))

    return text_columns


def process_page_image(
    image_path: str, coordinate_file: str, output_dir: str, target_size: int = 640
) -> tuple[str, list[dict]]:
    """ページ画像から縦方向の列を検出して切り出す (Pillowを使用)

    Args:
        image_path (str): 画像ファイルのパス
        coordinate_file (str): 文字座標のCSVファイルパス
        output_dir (str): 出力ディレクトリ
        target_size (int, optional): 未使用. Defaults to 640.

    Returns:
        Tuple[str, List[Dict]]:
            - 出力ディレクトリのパス
            - 列ごとの情報のリスト
    """
    try:
        # 画像の読み込み (Pillow)
        image = Image.open(image_path).convert("RGB")  # RGBに変換
        img_width, img_height = image.size  # 幅と高さを取得

        # 文字座標の読み込み
        df = pd.read_csv(coordinate_file)

        # 画像名が一致する行のみ抽出
        image_stem = Path(image_path).stem  # 拡張子なしのファイル名
        df = df[df["Image"] == image_stem]

        # 文字のバウンディングボックスを作成
        char_boxes = []
        for _, row in df.iterrows():
            # 座標やサイズが数値であることを確認
            try:
                x = float(row["X"])
                y = float(row["Y"])
                w = float(row["Width"])
                h = float(row["Height"])
                if w <= 0 or h <= 0:  # 幅や高さが0以下のデータはスキップ
                    # print(f"Warning: Invalid box dimensions for {row['Unicode']} in {image_stem}: W={w}, H={h}. Skipping.")
                    continue
                char_boxes.append(CharacterBox(x1=x, y1=y, x2=x + w, y2=y + h, unicode_id=str(row["Unicode"])))
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid coordinate data for {row.get('Unicode', 'N/A')} in {image_stem}: {e}. Skipping row.")
                continue

        # 列の検出
        text_columns = detect_text_columns(char_boxes)

        # 出力ディレクトリの作成
        image_id_parts = image_stem.split("_")
        # アンダースコアが含まれていない場合のエラーハンドリング
        doc_id = image_id_parts[0] if image_id_parts else image_stem
        output_subdir = Path(output_dir) / doc_id / image_stem
        output_subdir.mkdir(parents=True, exist_ok=True)

        # 列ごとの処理
        column_info = []
        for i, column in enumerate(text_columns, 1):
            if not column:  # 空の列はスキップ
                continue

            # 列の領域を計算
            x1 = min(char.x1 for char in column)
            y1 = min(char.y1 for char in column)
            x2 = max(char.x2 for char in column)
            y2 = max(char.y2 for char in column)

            # マージンを追加（文字の幅の中央値の20%）
            widths = [char.width for char in column if char.width > 0]  # 幅ゼロを除外
            if not widths:  # 有効な幅がない場合、デフォルトマージン
                margin = 5  # 仮のデフォルト値
            else:
                median_width = np.median(widths)
                margin = median_width * 0.2

            # 座標を整数にし、画像範囲内に収める (Pillow crop 用)
            crop_x1 = max(0, int(x1 - margin))
            crop_y1 = max(0, int(y1 - margin))
            crop_x2 = min(img_width, int(x2 + margin))
            crop_y2 = min(img_height, int(y2 + margin))

            # crop する領域のサイズが正であることを確認
            if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                # print(f"Warning: Invalid crop dimensions for column {i} in {image_stem}. Skipping.")
                continue

            # 列画像の切り出し (Pillow)
            column_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # 列画像の保存 (Pillow)
            output_path = output_subdir / f"{image_stem}_column_{i:03d}.jpg"
            column_image.save(str(output_path), "JPEG")  # JPEG形式で保存

            # 列の情報を保存 (座標は切り出した画像内の相対座標に)
            column_info.append(
                {
                    "column_image": str(output_path.relative_to(Path(output_dir).parent.parent)),  # data/ からの相対パスを想定
                    "original_image": image_path,  # 元画像のパスも保持
                    "box_in_original": [crop_x1, crop_y1, crop_x2, crop_y2],  # 元画像における列の絶対座標
                    "char_boxes_in_column": [  # 列画像内の相対座標
                        [
                            max(0, char.x1 - crop_x1),
                            max(0, char.y1 - crop_y1),
                            min(crop_x2 - crop_x1, char.x2 - crop_x1),
                            min(crop_y2 - crop_y1, char.y2 - crop_y1),
                        ]
                        for char in column
                    ],
                    "unicode_ids": [char.unicode_id for char in column],
                }
            )

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return str(output_dir), []  # エラー時は空リストを返す
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        # 必要に応じて、より詳細なエラーハンドリングやロギングを追加
        import traceback

        traceback.print_exc()
        return str(output_dir), []  # エラー時は空リストを返す

    return str(output_subdir), column_info


def main():
    # 設定の読み込み
    config_path = Path("config/preprocessing.yaml")  # configs ディレクトリに変更
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        return
    try:
        with open(config_path) as f:
            config = EasyDict(yaml.safe_load(f))  # EasyDictでラップ
    except yaml.YAMLError as e:
        print(f"Error loading YAML configuration: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the config file: {e}")
        return

    # 設定値の取得とデフォルト値の設定
    data_dir_str = getattr(config, "data_dir", "data")  # デフォルト値を 'data' に
    output_dir_name = getattr(config, "output_dir_name", "column_images")  # 出力ディレクトリ名を指定可能に
    raw_dataset_name = getattr(config, "raw_dataset_name", "dataset")  # rawデータセット名

    # 入力ディレクトリの設定
    data_dir = Path(data_dir_str)
    raw_dir = data_dir / "raw" / raw_dataset_name
    if not raw_dir.exists():
        print(f"Error: Raw data directory not found at {raw_dir}")
        return

    # 出力ディレクトリの設定と作成
    processed_base_dir = data_dir / "processed"  # processed をベースとする
    processed_base_dir.mkdir(parents=True, exist_ok=True)
    column_images_dir = processed_base_dir / output_dir_name
    column_images_dir.mkdir(parents=True, exist_ok=True)

    # 画像の処理
    all_column_info = []
    doc_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    print(f"Found {len(doc_dirs)} document directories in {raw_dir}")

    for doc_dir in tqdm(doc_dirs, desc="Processing documents"):
        # 画像ディレクトリとcoordinate.csvの確認
        images_dir = doc_dir / "images"
        coord_file = doc_dir / f"{doc_dir.name}_coordinate.csv"  # ファイル名を修正

        if not images_dir.exists():
            # print(f"Warning: Images directory not found in {doc_dir}. Skipping.")
            continue
        if not coord_file.exists():
            # print(f"Warning: Coordinate file not found in {doc_dir}. Skipping.")
            continue

        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))  # pngもサポート
        # print(f"Processing {len(image_files)} images in {doc_dir.name}...")

        # 各画像の処理
        for image_path in tqdm(image_files, desc=f"Images in {doc_dir.name}", leave=False):
            try:
                output_dir, column_info = process_page_image(str(image_path), str(coord_file), str(column_images_dir))
                # 列情報の保存
                all_column_info.extend(column_info)
            except Exception as e:
                print(f"\nError processing image {image_path}: {e}")
                import traceback

                traceback.print_exc()

    # 列情報をCSVファイルとして保存
    if all_column_info:
        column_info_df = pd.DataFrame(all_column_info)
        # CSV内のパスを processed_base_dir からの相対パスにする
        column_info_df["column_image"] = column_info_df["column_image"].apply(
            lambda p: str(Path(p).relative_to(processed_base_dir)) if Path(p).is_absolute() or Path(p).anchor else p
        )
        output_csv_path = processed_base_dir / "column_info.csv"
        column_info_df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully processed {len(all_column_info)} columns.")
        print(f"Column information saved to {output_csv_path}")
    else:
        print("\nNo columns were processed or extracted.")


if __name__ == "__main__":
    # Pillow がインストールされているか確認 (オプション)
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow library not found.")
        print("Please install it using: pip install Pillow")
        exit(1)
    main()
