import albumentations as A
import torchvision.transforms as T
from typing import Dict, Optional
import numpy as np
import torch
from PIL import Image


class AlbumentationsToTorchTransform:
    """AlbumentationsのTransformをPyTorchのTransformに変換するクラス"""
    
    def __init__(self, transform: A.Compose):
        self.transform = transform
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # テンソルをnumpy配列に変換
        image = image.permute(1, 2, 0).numpy()
        
        # Albumentationsの変換を適用
        transformed = self.transform(image=image)
        image = transformed["image"]
        
        # numpy配列をテンソルに戻す
        return torch.from_numpy(image).permute(2, 0, 1)


def get_column_detection_transforms(config: Dict) -> A.Compose:
    """列検出用のデータ拡張を取得する

    Args:
        config (Dict): 設定ファイルから読み込んだ設定

    Returns:
        A.Compose: データ拡張のパイプライン
    """
    aug_config = config["augmentation"]
    return A.Compose(
        [
            A.RandomRotate90(p=0.0),  # 日本語の文書なので90度回転は不要
            A.Rotate(limit=aug_config["rotation"], p=0.5),
            A.RandomScale(
                scale_limit=(aug_config["scale"][0] - 1.0, aug_config["scale"][1] - 1.0),
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config["brightness"],
                contrast_limit=aug_config["contrast"],
                p=0.5
            ),
            A.GaussNoise(p=0.3),  # デフォルトのパラメータを使用
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=[]),
    )


def get_character_detection_transforms(config: Dict) -> T.Compose:
    """文字検出用のデータ拡張を取得する

    Args:
        config (Dict): 設定ファイルから読み込んだ設定

    Returns:
        T.Compose: データ拡張のパイプライン
    """
    aug_config = config["augmentation"]

    # Albumentationsによる画像の拡張
    alb_transform = A.Compose([
        A.RandomRotate90(p=0.0),  # 日本語の文書なので90度回転は不要
        A.Rotate(limit=aug_config["rotation"], p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=aug_config["brightness"],
            contrast_limit=aug_config["contrast"],
            p=0.5
        ),
        A.GaussNoise(p=0.3),  # デフォルトのパラメータを使用
    ])

    # PyTorchのTransformパイプライン
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        AlbumentationsToTorchTransform(alb_transform),
        T.RandomErasing(
            p=aug_config["random_erase"],
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
            value=0
        ),
    ])
