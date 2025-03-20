import os
import shutil
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def convert_to_yolo_format(box, image_width, image_height):
    """バウンディングボックスをYOLOフォーマットに変換
    
    Args:
        box (list): [x1, y1, x2, y2]
        image_width (int): 画像の幅
        image_height (int): 画像の高さ
        
    Returns:
        list: [x_center, y_center, width, height]（正規化済み）
    """
    x1, y1, x2, y2 = box
    
    # 中心座標と幅・高さを計算
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    # 正規化
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height
    
    return [x_center, y_center, width, height]

def get_image_size(image_path):
    """画像サイズを取得

    Args:
        image_path (str): 画像ファイルのパス

    Returns:
        tuple: (width, height)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img.shape[1], img.shape[0]  # width, height

def prepare_yolo_dataset(
    column_info_file: str,
    output_dir: str,
    val_size: float = 0.2,
    random_state: int = 42
):
    """YOLOフォーマットのデータセットを準備
    
    Args:
        column_info_file (str): 列情報のCSVファイル
        output_dir (str): 出力ディレクトリ
        val_size (float, optional): 検証データの割合. Defaults to 0.2.
        random_state (int, optional): 乱数シード. Defaults to 42.
    """
    # 出力ディレクトリの準備
    output_dir = Path(output_dir)
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # 列情報の読み込み
    df = pd.read_csv(column_info_file)
    
    # 元画像ごとにグループ化
    image_groups = df.groupby('original_image')
    
    # 訓練データと検証データに分割
    image_paths = list(image_groups.groups.keys())
    train_paths, val_paths = train_test_split(
        image_paths,
        test_size=val_size,
        random_state=random_state
    )
    
    # データセットの準備
    for split, paths in [('train', train_paths), ('val', val_paths)]:
        print(f"Preparing {split} dataset...")
        for image_path in tqdm(paths):
            # 元画像の情報を取得
            image_data = image_groups.get_group(image_path)
            
            # 画像をコピー
            image_name = Path(image_path).name
            dst_image_path = output_dir / split / 'images' / image_name
            shutil.copy2(image_path, dst_image_path)
            
            # 画像サイズを取得
            try:
                image_width, image_height = get_image_size(image_path)
            except ValueError as e:
                print(f"Skipping {image_path}: {e}")
                continue
            
            # ラベルファイルを作成
            label_path = output_dir / split / 'labels' / f"{Path(image_name).stem}.txt"
            with open(label_path, 'w') as f:
                for _, row in image_data.iterrows():
                    # バウンディングボックスを取得
                    box = eval(row['box']) if isinstance(row['box'], str) else row['box']
                    
                    # YOLOフォーマットに変換
                    yolo_box = convert_to_yolo_format(
                        box,
                        image_width=image_width,
                        image_height=image_height
                    )
                    
                    # クラス0（列）として保存
                    f.write(f"0 {' '.join(map(str, yolo_box))}\n")

if __name__ == '__main__':
    prepare_yolo_dataset(
        column_info_file="data/processed/column_info.csv",
        output_dir="data/processed",
        val_size=0.2
    ) 