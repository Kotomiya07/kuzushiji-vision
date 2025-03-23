import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict
from torch.utils.data import Dataset
from .image_processing import resize_keeping_aspect_ratio, normalize_image
import torch
from pathlib import Path
from PIL import Image
import ast
import torchvision.transforms as T
from typing import Dict, List, Optional, Tuple


class ColumnDetectionDataset(Dataset):
    """列検出のためのデータセット"""

    def __init__(self, image_dir: str, annotation_file: str, target_size: int = 640, transform=None):
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
        image_path = os.path.join(self.image_dir, row["image_name"])

        # 画像の読み込みとリサイズ
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image, scale = resize_keeping_aspect_ratio(image, target_size=self.target_size)

        # バウンディングボックスのスケーリング
        boxes = np.array(eval(row["column_boxes"]))  # [N, 4]
        boxes = boxes * scale

        # 正規化
        resized_image = normalize_image(resized_image)

        # データ拡張
        if self.transform:
            transformed = self.transform(image=resized_image, bboxes=boxes)
            resized_image = transformed["image"]
            boxes = np.array(transformed["bboxes"])

        return {"image": resized_image, "boxes": boxes, "scale": scale}


class CharacterDetectionDataset(Dataset):
    """文字検出のデータセット"""

    def __init__(
        self,
        annotation_file: str,
        target_width: int = 192,
        transform: Optional[T.Compose] = None,
    ):
        """
        Args:
            annotation_file (str): アノテーションファイルのパス
            target_width (int, optional): リサイズ後の画像の幅. Defaults to 192.
            transform (Optional[T.Compose], optional): データ拡張. Defaults to None.
        """
        super().__init__()
        self.df = pd.read_csv(annotation_file)
        self.target_width = target_width
        self.transform = transform

        # Unicodeコードのマッピングを作成
        self.unicode_to_idx = {}
        for unicode_ids in self.df["unicode_ids"]:
            for unicode_id in ast.literal_eval(unicode_ids):
                if unicode_id not in self.unicode_to_idx:
                    self.unicode_to_idx[unicode_id] = len(self.unicode_to_idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # データの取得
        row = self.df.iloc[idx]
        image_path = row["column_image"]
        char_boxes = ast.literal_eval(row["char_boxes"])
        unicode_ids = ast.literal_eval(row["unicode_ids"])

        # 画像の読み込み
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # スケール係数の計算（幅を target_width にリサイズする際の比率）
        scale = self.target_width / w

        # バウンディングボックスのスケーリング
        boxes = []
        labels = []
        for box, unicode_id in zip(char_boxes, unicode_ids):
            x1, y1, x2, y2 = box
            x1 = x1 * scale
            y1 = y1 * scale
            x2 = x2 * scale
            y2 = y2 * scale
            boxes.append([x1, y1, x2, y2])
            labels.append(self.unicode_to_idx[unicode_id])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # 画像をテンソルに変換
        if self.transform is not None:
            # データ拡張を適用
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)
            image_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)

        # 画像のリサイズ（幅を固定し、高さはアスペクト比を保持）
        _, h, w = image_tensor.shape
        new_h = int(h * (self.target_width / w))
        image_tensor = T.Resize((new_h, self.target_width), antialias=True)(image_tensor)

        return {
            "image": image_tensor,
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
            "image_path": image_path,
            "height": new_h,  # リサイズ後の高さを追加
        }

    @property
    def num_classes(self) -> int:
        """文字クラスの数を返す"""
        return len(self.unicode_to_idx)
