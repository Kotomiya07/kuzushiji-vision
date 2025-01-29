#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class KuzushijiDataset(Dataset):
    """くずし字データセットのカスタムデータセット"""
    
    def __init__(
        self,
        root_dir: str,
        split: str,
        transform_config: Dict,
        max_size: Optional[int] = None
    ):
        """
        Args:
            root_dir: データセットのルートディレクトリ
            split: 'train', 'val', 'test'のいずれか
            transform_config: データ拡張の設定
            max_size: データセットサイズの制限（デバッグ用）
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform_config = transform_config
        
        # 画像とラベルのペアを読み込み
        self.samples = self._load_dataset(max_size)
        
        # データ拡張の設定
        self.transform = self._create_transforms()
        
    def _load_dataset(self, max_size: Optional[int]) -> List[Tuple[str, int]]:
        """データセットの読み込み"""
        samples = []
        
        # 文書IDごとのディレクトリをスキャン
        doc_dirs = os.listdir(os.path.join(self.root_dir, 'raw/dataset'))
        for doc_id in doc_dirs:
            doc_dir = os.path.join(self.root_dir, 'raw/dataset', doc_id)
            if not os.path.isdir(doc_dir):
                continue
                
            # 座標情報ファイルの読み込み
            coord_file = os.path.join(doc_dir, f'{doc_id}_coordinate.csv')
            if not os.path.exists(coord_file):
                continue
                
            # charactersディレクトリ内の文字画像を処理
            char_dir = os.path.join(doc_dir, 'characters')
            if not os.path.exists(char_dir):
                continue
                
            unicode_dirs = os.listdir(char_dir)
            for unicode_dir in unicode_dirs:
                char_dir_path = os.path.join(char_dir, unicode_dir)
                if not os.path.isdir(char_dir_path):
                    continue
                    
                char_images = os.listdir(char_dir_path)
                for img_name in char_images:
                    img_path = os.path.join(char_dir_path, img_name)
                    # Unicodeコードポイントを数値インデックスに変換
                    label = int(unicode_dir[2:], 16)  # "U+XXXX" -> int
                    samples.append((img_path, label))
        
        # データセットサイズの制限（指定がある場合）
        if max_size is not None:
            samples = samples[:max_size]
        
        return samples
    
    def _create_transforms(self) -> A.Compose:
        """Albumentationsによるデータ拡張パイプラインの作成"""
        if self.split == 'train':
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=10, p=0.7),
                A.RandomResizedCrop(
                    height=224,
                    width=224,
                    scale=(0.8, 1.0),
                    p=1.0
                ),
                A.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    p=0.7
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # 画像の読み込み
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # データ拡張の適用
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

def create_data_loaders(
    config: Dict,
    root_dir: str
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """データローダーの作成"""
    
    # データセットの作成
    train_dataset = KuzushijiDataset(
        root_dir=root_dir,
        split='train',
        transform_config=config['augmentation']['train']
    )
    
    val_dataset = KuzushijiDataset(
        root_dir=root_dir,
        split='val',
        transform_config=config['augmentation']['val']
    )
    
    test_dataset = KuzushijiDataset(
        root_dir=root_dir,
        split='test',
        transform_config=config['augmentation']['val']  # テストも検証と同じ変換
    )
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader