#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import csv
import multiprocessing as mp
from functools import partial

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

def resize_image(img: Image.Image, max_size: int) -> Image.Image:
    """アスペクト比を保持したままリサイズ"""
    w, h = img.size
    if w > h:
        if w > max_size:
            h = int(h * max_size / w)
            w = max_size
    else:
        if h > max_size:
            w = int(w * max_size / h)
            h = max_size
    return img.resize((w, h), Image.Resampling.LANCZOS)

def process_image(
    image_path: str,
    max_size: Optional[int] = None,
    return_array: bool = True
) -> np.ndarray:
    """画像の読み込みと前処理"""
    try:
        with Image.open(image_path) as img:
            if max_size:
                img = resize_image(img, max_size)
            if return_array:
                return np.array(img.convert('RGB'))
            return img
    except Exception as e:
        raise RuntimeError(f"画像の処理中にエラーが発生: {image_path}, {str(e)}")

def save_to_lmdb_batch(
    samples: List[Dict],
    output_path: Path,
    batch_size: int = 100,
    max_image_size: Optional[int] = None
) -> List[Dict]:
    """データをLMDBにバッチ単位で保存"""
    env = lmdb.open(str(output_path), map_size=1099511627776)  # 1TB
    metadata = []
    
    for i in tqdm(range(0, len(samples), batch_size), desc="Processing batches"):
        batch_samples = samples[i:i + batch_size]
        with env.begin(write=True) as txn:
            for idx, sample in enumerate(batch_samples):
                try:
                    # 画像の読み込みと前処理
                    image = process_image(sample['image_path'], max_size=max_image_size)
                    
                    # 画像データの保存
                    global_idx = i + idx
                    image_key = f"image_{global_idx}".encode()
                    txn.put(image_key, pickle.dumps(image))
                    
                    # メタデータの作成
                    metadata.append({
                        'image_key': f"image_{global_idx}",
                        'doc_id': sample['doc_id'],
                        'chars': sample['chars'],
                        'width': image.shape[1],
                        'height': image.shape[0]
                    })
                    
                    # メモリ解放
                    del image
                except Exception as e:
                    print(f"Warning: Failed to process {sample['image_path']}: {str(e)}")
                    continue
    
    env.close()
    return metadata

def process_document(doc_dir: Path, temp_dir: Path) -> List[Dict]:
    """1つの文書を処理"""
    samples = []
    
    # ファイルの存在確認
    coord_file = doc_dir / f"{doc_dir.name}_coordinate.csv"
    image_dir = doc_dir / "images"
    if not coord_file.exists() or not image_dir.exists():
        return samples
    
    # 座標情報の読み込みと画像ごとのグループ化
    char_groups = {}
    with open(coord_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row['Image']
            if image_name not in char_groups:
                char_groups[image_name] = []
            char_groups[image_name].append({
                'unicode': row['Unicode'],
                'x': float(row['X']),
                'y': float(row['Y']),
                'width': float(row['Width']),
                'height': float(row['Height']),
                'block_id': row['Block ID'],
                'char_id': row['Char ID']
            })
    
    # サンプルの作成
    for image_name, chars in char_groups.items():
        image_path = image_dir / f"{image_name}.jpg"
        if not image_path.exists():
            continue
        
        try:
            # 画像サイズの取得
            with Image.open(image_path) as img:
                W, H = img.size
            
            # 座標の正規化
            normalized_chars = [{
                'unicode': char['unicode'],
                'x': char['x'] / W,
                'y': char['y'] / H,
                'width': char['width'] / W,
                'height': char['height'] / H,
                'block_id': char['block_id'],
                'char_id': char['char_id']
            } for char in chars]
            
            # 右から左、上から下の順にソート
            normalized_chars.sort(key=lambda x: (-x['x'], x['y']))
            
            samples.append({
                'image_path': str(image_path),
                'doc_id': doc_dir.name,
                'chars': normalized_chars,
                'width': W,
                'height': H
            })
        except Exception as e:
            print(f"Warning: Failed to process {image_path}: {str(e)}")
            continue
    
    return samples

def process_documents_parallel(
    doc_dirs: List[Path],
    temp_dir: Path,
    num_processes: Optional[int] = None
) -> List[Dict]:
    """文書の並列処理"""
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(
                partial(process_document, temp_dir=temp_dir),
                doc_dirs,
                chunksize=1
            ),
            total=len(doc_dirs),
            desc="Processing documents"
        ))
    
    # 結果の統合
    samples = []
    for result in results:
        samples.extend(result)
    
    return samples

class DatasetPreprocessor:
    def __init__(self, config: Dict, data_dir: str, output_dir: str, logger: logging.Logger):
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.logger = logger
        
        # 出力ディレクトリの作成
        self.processed_dir = self.output_dir / "processed"
        self.splits_dir = self.output_dir / "splits"
        self.lmdb_dir = self.output_dir / "lmdb"
        self.temp_dir = self.output_dir / "temp"
        
        for dir_path in [self.processed_dir, self.splits_dir, self.lmdb_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 文字マッピングの初期化
        self.char_to_idx = {}
        self.idx_to_char = {}
    
    def build_char_mapping(self, samples: List[Dict]) -> None:
        """文字マッピングの構築"""
        for sample in samples:
            for char in sample['chars']:
                unicode_value = char['unicode']
                if unicode_value not in self.char_to_idx:
                    idx = len(self.char_to_idx)
                    self.char_to_idx[unicode_value] = idx
                    self.idx_to_char[idx] = unicode_value
    
    def split_dataset(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """データセットの分割"""
        self.logger.info("データセットの分割を開始")
        
        doc_ids = [sample['doc_id'] for sample in samples]
        train_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, temp_idx = next(train_splitter.split(samples, groups=doc_ids))
        
        val_test_samples = [samples[i] for i in temp_idx]
        val_test_doc_ids = [doc_ids[i] for i in temp_idx]
        
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_idx, test_idx = next(val_splitter.split(val_test_samples, groups=val_test_doc_ids))
        
        train_samples = [samples[i] for i in train_idx]
        val_samples = [val_test_samples[i] for i in val_idx]
        test_samples = [val_test_samples[i] for i in test_idx]
        
        self.logger.info(f"訓練データ: {len(train_samples)}ページ")
        self.logger.info(f"検証データ: {len(val_samples)}ページ")
        self.logger.info(f"テストデータ: {len(test_samples)}ページ")
        
        return train_samples, val_samples, test_samples
    
    def run(self) -> None:
        """前処理の実行"""
        try:
            # データの並列読み込み
            dataset_dir = self.data_dir / "dataset"
            doc_dirs = [d for d in dataset_dir.glob("*") if d.is_dir()]
            
            self.logger.info("データの読み込みを開始")
            samples = process_documents_parallel(doc_dirs, self.temp_dir)
            self.logger.info(f"合計{len(samples)}ページを読み込みました")
            
            # 文字マッピングの構築
            self.build_char_mapping(samples)
            self.logger.info(f"文字クラス数: {len(self.char_to_idx)}")
            
            # データセットの分割
            train_samples, val_samples, test_samples = self.split_dataset(samples)
            
            # LMDBデータセットの作成
            splits = [
                ("train", train_samples),
                ("val", val_samples),
                ("test", test_samples)
            ]
            
            max_image_size = self.config.get('input', {}).get('max_size', 2048)
            batch_size = self.config.get('data', {}).get('batch_size', 100)
            
            for split_name, split_samples in splits:
                self.logger.info(f"{split_name}用のLMDBデータセットを作成")
                
                # LMDBの作成
                lmdb_path = self.lmdb_dir / f"lmdb_{split_name}"
                metadata = save_to_lmdb_batch(
                    split_samples,
                    lmdb_path,
                    batch_size=batch_size,
                    max_image_size=max_image_size
                )
                
                # メタデータの保存
                metadata_path = self.splits_dir / f"{split_name}_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 文字マッピングの保存
            mapping_path = self.processed_dir / "char_mapping.json"
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'char_to_idx': self.char_to_idx,
                    'idx_to_char': self.idx_to_char
                }, f, ensure_ascii=False, indent=2)
            
            self.logger.info("前処理が完了しました")
            
        except Exception as e:
            self.logger.error(f"前処理中にエラーが発生しました: {e}")
            raise
        finally:
            # 一時ディレクトリの削除
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)

def main():
    parser = argparse.ArgumentParser(description="くずし字データセットの前処理")
    parser.add_argument("--config", type=str, required=True, help="設定ファイルのパス")
    parser.add_argument("--data-dir", type=str, required=True, help="生データのディレクトリ")
    parser.add_argument("--output-dir", type=str, required=True, help="出力ディレクトリ")
    parser.add_argument("--max-image-size", type=int, help="画像の最大サイズ")
    parser.add_argument("--batch-size", type=int, default=100, help="バッチサイズ")
    parser.add_argument("--num-workers", type=int, help="並列処理のワーカー数")
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # コマンドライン引数で設定を上書き
    if args.max_image_size:
        config.setdefault('input', {})['max_size'] = args.max_image_size
    if args.batch_size:
        config.setdefault('data', {})['batch_size'] = args.batch_size
    
    logger = setup_logger(args.output_dir)
    preprocessor = DatasetPreprocessor(config, args.data_dir, args.output_dir, logger)
    preprocessor.run()

if __name__ == "__main__":
    main()