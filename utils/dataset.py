import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict
from torch.utils.data import Dataset
from .image_processing import resize_keeping_aspect_ratio, normalize_image

class ColumnDetectionDataset(Dataset):
    """列検出のためのデータセット"""
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        target_size: int = 640,
        transform = None
    ):
        """
        Args:
            image_dir (str): 画像ディレクトリのパス
            annotation_file (str): アノテーションファイルのパス
            target_size (int, optional): リサイズ後の長辺のサイズ. Defaults to 640.
            transform: データ拡張の関数. Defaults to None.
        """
        self.image_dir = image_dir
        self.annotations = pd.read_csv(annotation_file)
        self.target_size = target_size
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # アノテーション情報の取得
        row = self.annotations.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_name'])
        
        # 画像の読み込みとリサイズ
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image, scale = resize_keeping_aspect_ratio(
            image, target_size=self.target_size
        )
        
        # バウンディングボックスのスケーリング
        boxes = np.array(eval(row['column_boxes']))  # [N, 4]
        boxes = boxes * scale
        
        # 正規化
        resized_image = normalize_image(resized_image)
        
        # データ拡張
        if self.transform:
            transformed = self.transform(
                image=resized_image,
                bboxes=boxes
            )
            resized_image = transformed['image']
            boxes = np.array(transformed['bboxes'])
        
        return {
            'image': resized_image,
            'boxes': boxes,
            'scale': scale
        }

class CharacterDetectionDataset(Dataset):
    """文字検出のためのデータセット"""
    
    def __init__(
        self,
        column_image_dir: str,
        annotation_file: str,
        target_width: int = 192,
        transform = None
    ):
        """
        Args:
            column_image_dir (str): 列画像ディレクトリのパス
            annotation_file (str): アノテーションファイルのパス
            target_width (int, optional): リサイズ後の横幅. Defaults to 192.
            transform: データ拡張の関数. Defaults to None.
        """
        self.column_image_dir = column_image_dir
        self.annotations = pd.read_csv(annotation_file)
        self.target_width = target_width
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # アノテーション情報の取得
        row = self.annotations.iloc[idx]
        image_path = os.path.join(self.column_image_dir, row['column_image_name'])
        
        # 画像の読み込みとリサイズ
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image, scale = resize_keeping_aspect_ratio(
            image,
            target_size=0,  # 無視される
            target_width=self.target_width
        )
        
        # 文字のバウンディングボックスとラベルの取得
        boxes = np.array(eval(row['char_boxes']))  # [N, 4]
        labels = np.array(eval(row['char_labels']))  # [N]
        
        # バウンディングボックスのスケーリング
        boxes = boxes * scale
        
        # 正規化
        resized_image = normalize_image(resized_image)
        
        # データ拡張
        if self.transform:
            transformed = self.transform(
                image=resized_image,
                bboxes=boxes,
                labels=labels
            )
            resized_image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])
        
        return {
            'image': resized_image,
            'boxes': boxes,
            'labels': labels,
            'scale': scale
        } 