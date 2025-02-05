#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import csv

import numpy as np
from PIL import Image
import lmdb
import yaml
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

def setup_logger(log_dir: str) -> logging.Logger:
    """ロガーの設定"""
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    
    os.makedirs(log_dir, exist_ok=True)
    
    fh = logging.FileHandler(os.path.join(log_dir, "preprocess.log"))
    ch = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def save_to_lmdb(samples: List[Dict], output_path: Path, logger: logging.Logger, split: str) -> List[Dict]:
    """データをLMDBに保存"""
    logger.info(f"{split}用のLMDBデータセットを作成")
    
    env = lmdb.open(str(output_path), map_size=1099511627776)  # 1TB
    metadata = []
    
    with env.begin(write=True) as txn:
        for idx, sample in enumerate(tqdm(samples, desc=f"Creating {split} LMDB")):
            # 画像の読み込みと保存
            image = np.array(Image.open(sample['image_path']).convert('RGB'))
            image_key = f"image_{idx}".encode()
            txn.put(image_key, pickle.dumps(image))
            
            # メタデータの作成
            metadata.append({
                'image_key': f"image_{idx}",
                'doc_id': sample['doc_id'],
                'chars': sample['chars'],
                'width': sample['width'],
                'height': sample['height']
            })
    
    return metadata

class DatasetPreprocessor:
    """くずし字データセットの前処理クラス"""
    def __init__(self, config: Dict, data_dir: str, output_dir: str, logger: logging.Logger):
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.logger = logger
        
        # 出力ディレクトリの作成
        self.processed_dir = self.output_dir / "processed"
        self.splits_dir = self.output_dir / "splits"
        self.lmdb_dir = self.output_dir / "lmdb"
        
        for dir_path in [self.processed_dir, self.splits_dir, self.lmdb_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 文字マッピングの初期化
        self.char_to_idx = {}
        self.idx_to_char = {}
    
    def process_coordinate_file(self, coord_file: Path) -> List[Dict]:
        """座標ファイルの処理"""
        chars = []
        with open(coord_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                unicode_value = row['Unicode']
                if unicode_value not in self.char_to_idx:
                    idx = len(self.char_to_idx)
                    self.char_to_idx[unicode_value] = idx
                    self.idx_to_char[idx] = unicode_value
                
                chars.append({
                    'unicode': unicode_value,
                    'image_name': row['Image'],
                    'x': float(row['X']),
                    'y': float(row['Y']),
                    'width': float(row['Width']),
                    'height': float(row['Height']),
                    'block_id': row['Block ID'],
                    'char_id': row['Char ID']
                })
        return chars
    
    def load_document(self, doc_dir: Path) -> List[Dict]:
        """1つの文書からデータを読み込む"""
        samples = []
        
        # ファイルの存在確認
        coord_file = doc_dir / f"{doc_dir.name}_coordinate.csv"
        image_dir = doc_dir / "images"
        if not coord_file.exists() or not image_dir.exists():
            self.logger.warning(f"必要なファイルが見つかりません: {doc_dir}")
            return samples
        
        # 座標情報の読み込み
        chars = self.process_coordinate_file(coord_file)
        
        # 画像ごとにグループ化
        char_groups = {}
        for char in chars:
            image_name = char['image_name']
            if image_name not in char_groups:
                char_groups[image_name] = []
            char_groups[image_name].append(char)
        
        # サンプルの作成
        for image_name, page_chars in char_groups.items():
            image_path = image_dir / f"{image_name}.jpg"
            if not image_path.exists():
                continue
            
            # 画像サイズの取得と座標の正規化
            with Image.open(image_path) as img:
                W, H = img.size
                normalized_chars = []
                for char in page_chars:
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
                'chars': normalized_chars,
                'width': W,
                'height': H
            })
        
        return samples
    
    def split_dataset(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """データセットの分割"""
        self.logger.info("データセットの分割を開始")
        
        # 文書IDの抽出
        doc_ids = [sample['doc_id'] for sample in samples]
        
        # 訓練データとその他のデータに分割
        train_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, temp_idx = next(train_splitter.split(samples, groups=doc_ids))
        
        # 検証データとテストデータに分割
        val_test_samples = [samples[i] for i in temp_idx]
        val_test_doc_ids = [doc_ids[i] for i in temp_idx]
        
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_idx, test_idx = next(val_splitter.split(val_test_samples, groups=val_test_doc_ids))
        
        # データセットの分割
        train_samples = [samples[i] for i in train_idx]
        val_samples = [val_test_samples[i] for i in val_idx]
        test_samples = [val_test_samples[i] for i in test_idx]
        
        self.logger.info(f"訓練データ: {len(train_samples)}ページ")
        self.logger.info(f"検証データ: {len(val_samples)}ページ")
        self.logger.info(f"テストデータ: {len(test_samples)}ページ")
        
        return train_samples, val_samples, test_samples
    
    def save_char_mapping(self) -> None:
        """文字マッピングの保存"""
        mapping_path = self.processed_dir / "char_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char
            }, f, ensure_ascii=False, indent=2)
    
    def run(self) -> None:
        """前処理の実行"""
        try:
            # 全データの読み込み
            self.logger.info("データの読み込みを開始")
            samples = []
            dataset_dir = self.data_dir / "dataset"
            
            for doc_dir in tqdm(list(dataset_dir.glob("*")), desc="Loading documents"):
                if doc_dir.is_dir():
                    samples.extend(self.load_document(doc_dir))
            
            self.logger.info(f"合計{len(samples)}ページを読み込みました")
            self.logger.info(f"文字クラス数: {len(self.char_to_idx)}")
            
            # データセットの分割とLMDB作成
            train_samples, val_samples, test_samples = self.split_dataset(samples)
            
            # LMDBデータセットの作成
            splits = [
                ("train", train_samples),
                ("val", val_samples),
                ("test", test_samples)
            ]
            
            for split_name, split_samples in splits:
                # LMDBの作成
                lmdb_path = self.lmdb_dir / f"lmdb_{split_name}"
                metadata = save_to_lmdb(split_samples, lmdb_path, self.logger, split_name)
                
                # メタデータの保存
                metadata_path = self.splits_dir / f"{split_name}_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 文字マッピングの保存
            self.save_char_mapping()
            
            self.logger.info("前処理が完了しました")
            
        except Exception as e:
            self.logger.error(f"前処理中にエラーが発生しました: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="くずし字データセットの前処理")
    parser.add_argument("--config", type=str, required=True, help="設定ファイルのパス")
    parser.add_argument("--data-dir", type=str, required=True, help="生データのディレクトリ")
    parser.add_argument("--output-dir", type=str, required=True, help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    logger = setup_logger(args.output_dir)
    preprocessor = DatasetPreprocessor(config, args.data_dir, args.output_dir, logger)
    preprocessor.run()

if __name__ == "__main__":
    main()