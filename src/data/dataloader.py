#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lmdb
from PIL import Image

# albumentationsのインポート
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available. Data augmentation will be disabled.")

class KuzushijiDataset(Dataset):
    """くずし字データセット"""
    def __init__(
        self,
        config: Dict,
        data_dir: str,
        split: str,
        transform_config: Optional[Dict] = None,
        use_lmdb: bool = True
    ):
        self.config = config
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_lmdb = use_lmdb
        
        # transformの設定
        if transform_config and ALBUMENTATIONS_AVAILABLE:
            self.transform = self._create_default_transforms()
        else:
            self.transform = None
        
        # データの読み込み
        self.samples = self._load_dataset()
        
        # LMDBの設定
        if use_lmdb:
            self.env = lmdb.open(
                str(self.data_dir / f"lmdb_{split}"),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            self.txn = self.env.begin(write=False)
    
    def _create_default_transforms(self) -> Optional[A.Compose]:
        """デフォルトのtransform"""
        if not ALBUMENTATIONS_AVAILABLE:
            return None
            
        if self.split == 'train':
            return A.Compose([
                A.RandomRotate90(p=0.0),  # 縦書きなので回転は制限
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
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
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def _load_dataset(self) -> List[Dict]:
        """データセットの読み込み"""
        samples = []
        dataset_dir = self.data_dir / "raw/dataset"
        
        for doc_dir in dataset_dir.glob("*"):
            if not doc_dir.is_dir():
                continue
            
            # coordinate.csvの読み込み
            coord_file = doc_dir / f"{doc_dir.name}_coordinate.csv"
            if not coord_file.exists():
                continue
            
            # 画像ディレクトリの確認
            image_dir = doc_dir / "images"
            if not image_dir.exists():
                continue
            
            # 座標情報の読み込み
            with open(coord_file, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                chars = []
                for row in reader:
                    chars.append({
                        'unicode': row['Unicode'],
                        'image_name': row['Image'],
                        'x': float(row['X']),
                        'y': float(row['Y']),
                        'width': float(row['Width']),
                        'height': float(row['Height']),
                        'block_id': row['Block ID'],
                        'char_id': row['Char ID']
                    })
            
            # 画像ごとにグループ化
            char_groups = {}
            for char in chars:
                image_name = char['image_name']
                if image_name not in char_groups:
                    char_groups[image_name] = []
                char_groups[image_name].append(char)
            
            # サンプルの作成
            for image_name, chars in char_groups.items():
                image_path = image_dir / f"{image_name}.jpg"
                if not image_path.exists():
                    continue
                
                # 画像サイズの取得
                with Image.open(image_path) as img:
                    W, H = img.size
                
                # 座標の正規化
                normalized_chars = []
                for char in chars:
                    normalized_chars.append({
                        'unicode': char['unicode'],
                        'x': char['x'] / W,
                        'y': char['y'] / H,
                        'width': char['width'] / W,
                        'height': char['height'] / H,
                        'block_id': char['block_id'],
                        'char_id': char['char_id']
                    })
                
                # 右から左、上から下の順にソート
                normalized_chars.sort(key=lambda x: (-x['x'], x['y']))
                
                samples.append({
                    'image_path': str(image_path),
                    'doc_id': doc_dir.name,
                    'chars': normalized_chars
                })
        
        return samples
    
    def _prepare_target(
        self,
        chars: List[Dict],
        max_chars: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ターゲットの準備"""
        # 位置情報の準備
        positions = torch.zeros(max_chars, 4)
        # Unicode情報の準備
        unicodes = torch.zeros(max_chars, dtype=torch.long)
        
        for i, char in enumerate(chars[:max_chars]):
            # 位置情報: [x, y, width, height]
            positions[i] = torch.tensor([
                char['x'],
                char['y'],
                char['width'],
                char['height']
            ])
            # Unicode: U+XXXXの形式から数値に変換
            unicode_value = int(char['unicode'][2:], 16)
            unicodes[i] = unicode_value
        
        return positions, unicodes
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 画像の読み込み
        if self.use_lmdb:
            # LMDBからの読み込み
            key = f"{idx}".encode()
            image_data = self.txn.get(key)
            if image_data is None:
                raise ValueError(f"Key {idx} not found in LMDB")
            image = pickle.loads(image_data)
        else:
            # 直接ファイルから読み込み
            image = Image.open(sample['image_path']).convert('RGB')
            image = np.array(image)
        
        # 画像の前処理
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # transformがない場合の基本的な前処理
            image = torch.from_numpy(image.transpose(2, 0, 1))
            image = image.float() / 255.0
        
        # ターゲットの準備
        max_chars = self.config['output']['max_chars_per_page']
        positions, unicodes = self._prepare_target(sample['chars'], max_chars)
        
        return {
            'image': image,
            'positions': positions,
            'unicodes': unicodes,
            'num_chars': len(sample['chars']),
            'doc_id': sample['doc_id']
        }

def create_lmdb_dataset(
    config: Dict,
    data_dir: str,
    split: str
) -> None:
    """LMDBデータセットの作成"""
    dataset = KuzushijiDataset(
        config=config,
        data_dir=data_dir,
        split=split,
        use_lmdb=False
    )
    
    # LMDBの設定
    lmdb_path = Path(data_dir) / f"lmdb_{split}"
    env = lmdb.open(
        str(lmdb_path),
        map_size=1099511627776  # 1TB
    )
    
    # データの書き込み
    with env.begin(write=True) as txn:
        for idx in range(len(dataset)):
            sample = dataset.samples[idx]
            # 画像の読み込み
            image = Image.open(sample['image_path']).convert('RGB')
            image = np.array(image)
            # シリアライズ
            key = f"{idx}".encode()
            value = pickle.dumps(image)
            # 保存
            txn.put(key, value)

def create_data_loaders(
    config: Dict,
    data_dir: str,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """データローダーの作成"""
    if batch_size is None:
        batch_size = config['training']['batch_size']
    if num_workers is None:
        num_workers = config['data']['num_workers']
    
    # データセットの作成
    train_dataset = KuzushijiDataset(
        config=config,
        data_dir=data_dir,
        split='train',
        transform_config=config['augmentation']['train']
    )
    
    val_dataset = KuzushijiDataset(
        config=config,
        data_dir=data_dir,
        split='val',
        transform_config=config['augmentation']['val']
    )
    
    test_dataset = KuzushijiDataset(
        config=config,
        data_dir=data_dir,
        split='test',
        transform_config=config['augmentation']['val']
    )
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader