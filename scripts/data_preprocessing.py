import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.cluster import DBSCAN


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
    char_boxes: List[CharacterBox], eps_ratio: float = 0.1, min_samples: int = 1
) -> List[List[CharacterBox]]:
    """DBSCANを使用して文字を列ごとにクラスタリング

    Args:
        char_boxes (List[CharacterBox]): 文字のバウンディングボックスのリスト
        eps_ratio (float, optional): 文字の幅の中央値に対するepsの割合. Defaults to 0.5.
        min_samples (int, optional): DBSCANのmin_samples. Defaults to 1.

    Returns:
        List[List[CharacterBox]]: 列ごとにグループ化された文字のリスト
    """
    if not char_boxes:
        return []

    # 各文字の幅の中央値を計算
    widths = [box.width for box in char_boxes]
    median_width = np.median(widths)
    eps = median_width * eps_ratio

    # 各文字の中心のx座標を計算
    x_centers = np.array([(box.x1 + box.x2) / 2 for box in char_boxes]).reshape(-1, 1)

    # DBSCANによるクラスタリング
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x_centers)
    labels = clustering.labels_

    # クラスタごとに文字をグループ化
    columns = {}
    for label, box in zip(labels, char_boxes):
        if label == -1:
            # ノイズとして検出された場合は新しいグループを作成
            label = max(labels) + 1 if len(labels) > 0 else 0
        if label not in columns:
            columns[label] = []
        columns[label].append(box)

    # 各列内の文字をy座標でソート（上から下）
    text_columns = []
    for label in sorted(columns.keys()):
        column = sorted(columns[label], key=lambda box: box.y1)
        text_columns.append(column)

    # 列を左から右にソート
    text_columns.sort(key=lambda column: min(box.x1 for box in column))

    return text_columns


def process_page_image(
    image_path: str, coordinate_file: str, output_dir: str, target_size: int = 640
) -> Tuple[str, List[Dict]]:
    """ページ画像から縦方向の列を検出して切り出す

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
    # 画像の読み込み
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 文字座標の読み込み
    df = pd.read_csv(coordinate_file)

    # 画像名が一致する行のみ抽出
    df = df[df["Image"] == image_path.split("/")[-1].split(".")[0]]

    # 文字のバウンディングボックスを作成
    char_boxes = []
    for _, row in df.iterrows():
        char_boxes.append(
            CharacterBox(
                x1=row["X"], y1=row["Y"], x2=row["X"] + row["Width"], y2=row["Y"] + row["Height"], unicode_id=row["Unicode"]
            )
        )

    # 列の検出
    text_columns = detect_text_columns(char_boxes)

    # 出力ディレクトリの作成
    image_id = Path(image_path).stem
    output_subdir = Path(output_dir) / image_id
    output_subdir.mkdir(parents=True, exist_ok=True)

    # 列ごとの処理
    column_info = []
    for i, column in enumerate(text_columns, 1):
        # 列の領域を計算
        x1 = min(char.x1 for char in column)
        y1 = min(char.y1 for char in column)
        x2 = max(char.x2 for char in column)
        y2 = max(char.y2 for char in column)

        # マージンを追加（文字の幅の中央値の20%）
        widths = [char.width for char in column]
        median_width = np.median(widths)
        margin = median_width * 0.2

        y1 = max(0, int(y1 - margin))
        y2 = min(image.shape[0], int(y2 + margin))
        x1 = max(0, int(x1 - margin))
        x2 = min(image.shape[1], int(x2 + margin))

        # 列画像の切り出し
        column_image = image[y1:y2, x1:x2]

        # 列画像の保存
        output_path = output_subdir / f"{image_id}_column_{i:03d}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(column_image, cv2.COLOR_RGB2BGR))

        # 列の情報を保存
        column_info.append(
            {
                "column_image": str(output_path),
                "original_image": image_path,
                "box": [x1, y1, x2, y2],
                "char_boxes": [[char.x1 - x1, char.y1 - y1, char.x2 - x1, char.y2 - y1] for char in column],
                "unicode_ids": [char.unicode_id for char in column],
            }
        )

    return str(output_subdir), column_info


def main():
    # 設定の読み込み
    with open("config/preprocessing.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 入力ディレクトリの設定
    data_dir = Path(config["data_dir"])
    raw_dir = data_dir / "raw" / "dataset"

    # 出力ディレクトリの設定と作成
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    column_images_dir = processed_dir / "column_images"
    column_images_dir.mkdir(parents=True, exist_ok=True)

    # 画像の処理
    all_column_info = []
    for page_dir in tqdm(list(raw_dir.iterdir())):
        if not page_dir.is_dir():
            continue

        # 画像ディレクトリとcoordinate.csvの確認
        images_dir = page_dir / "images"
        coord_file = page_dir / f"{page_dir.name}_coordinate.csv"

        if not images_dir.exists() or not coord_file.exists():
            continue

        # 各画像の処理
        for image_path in images_dir.glob("*.jpg"):
            output_dir, column_info = process_page_image(str(image_path), str(coord_file), str(column_images_dir))

            # 列情報の保存
            all_column_info.extend(column_info)

    # 列情報をCSVファイルとして保存
    column_info_df = pd.DataFrame(all_column_info)
    column_info_df.to_csv(processed_dir / "column_info.csv", index=False)


if __name__ == "__main__":
    main()
