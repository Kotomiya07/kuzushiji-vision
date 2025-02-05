#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import json
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split

def setup_logger(log_dir: str) -> logging.Logger:
    """ロガーの設定"""
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    fh = logging.FileHandler(os.path.join(log_dir, "preprocess.log"))
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

class DataPreprocessor:
    """くずし字データセットの前処理クラス"""
    
    def __init__(
        self,
        config_path: str,
        data_dir: str,
        output_dir: str,
        logger: logging.Logger
    ):
        self.logger = logger
        
        # 設定の読み込み
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # パスの設定
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.processed_dir = self.output_dir / "processed"
        self.splits_dir = self.output_dir / "splits"
        
        # 出力ディレクトリの作成
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像サイズの設定
        self.image_size = tuple(self.config["input"]["size"])
        
        # 文字クラスの管理
        self.char_to_idx = {}
        self.idx_to_char = {}
    
    def load_and_verify_data(self) -> List[Dict]:
        """データの読み込みと検証"""
        self.logger.info("データの読み込みと検証を開始")
        
        samples = []
        doc_dirs = list(self.data_dir.glob("*/"))
        
        for doc_dir in tqdm(doc_dirs, desc="Loading documents"):
            if not doc_dir.is_dir():
                continue
            
            coord_file = doc_dir / f"{doc_dir.name}_coordinate.csv"
            if not coord_file.exists():
                self.logger.warning(f"座標ファイルが見つかりません: {coord_file}")
                continue
            
            char_dir = doc_dir / "characters"
            if not char_dir.exists():
                self.logger.warning(f"charactersディレクトリが見つかりません: {char_dir}")
                continue
            
            for unicode_dir in char_dir.glob("*/"):
                if not unicode_dir.is_dir():
                    continue
                
                char = chr(int(unicode_dir.name[2:], 16))
                if char not in self.char_to_idx:
                    idx = len(self.char_to_idx)
                    self.char_to_idx[char] = idx
                    self.idx_to_char[idx] = char
                
                for img_path in unicode_dir.glob("*.jpg"):
                    samples.append({
                        "image_path": str(img_path),
                        "char": char,
                        "label": self.char_to_idx[char],
                        "doc_id": doc_dir.name
                    })
        
        self.logger.info(f"読み込んだサンプル数: {len(samples)}")
        self.logger.info(f"文字クラス数: {len(self.char_to_idx)}")
        
        return samples
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """画像の前処理"""
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size, Image.LANCZOS)
        
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        
        mean = np.array(self.config["input"]["normalize"]["mean"])
        std = np.array(self.config["input"]["normalize"]["std"])
        image = (image - mean) / std
        
        return image
    
    def split_dataset(
        self,
        samples: List[Dict],
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """データセットの分割"""
        self.logger.info("データセットの分割を開始")
        
        doc_ids = list(set(s["doc_id"] for s in samples))
        
        train_docs, temp_docs = train_test_split(
            doc_ids,
            test_size=(val_size + test_size),
            random_state=random_state
        )
        
        val_docs, test_docs = train_test_split(
            temp_docs,
            test_size=test_size/(val_size + test_size),
            random_state=random_state
        )
        
        train_samples = [s for s in samples if s["doc_id"] in train_docs]
        val_samples = [s for s in samples if s["doc_id"] in val_docs]
        test_samples = [s for s in samples if s["doc_id"] in test_docs]
        
        self.logger.info(f"訓練サンプル数: {len(train_samples)}")
        self.logger.info(f"検証サンプル数: {len(val_samples)}")
        self.logger.info(f"テストサンプル数: {len(test_samples)}")
        
        return train_samples, val_samples, test_samples
    
    def save_processed_data(
        self,
        split_name: str,
        samples: List[Dict]
    ) -> None:
        """前処理済みデータの保存"""
        output_images_dir = self.processed_dir / "images" / split_name
        output_images_dir.mkdir(parents=True, exist_ok=True)
        
        processed_samples = []
        for sample in tqdm(samples, desc=f"Processing {split_name}"):
            image = self.preprocess_image(sample["image_path"])
            image_filename = f"{len(processed_samples):08d}.npy"
            image_path = output_images_dir / image_filename
            np.save(str(image_path), image)
            
            processed_sample = {
                "processed_image_path": str(image_path),
                "original_image_path": sample["image_path"],
                "char": sample["char"],
                "label": sample["label"],
                "doc_id": sample["doc_id"]
            }
            processed_samples.append(processed_sample)
        
        metadata_path = self.splits_dir / f"{split_name}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(processed_samples, f, ensure_ascii=False, indent=2)
    
    def save_char_mapping(self) -> None:
        """文字マッピングの保存"""
        mapping_path = self.processed_dir / "char_mapping.json"
        mapping = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char
        }
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    def run(self) -> None:
        """前処理の実行"""
        try:
            samples = self.load_and_verify_data()
            train_samples, val_samples, test_samples = self.split_dataset(samples)
            
            self.save_processed_data("train", train_samples)
            self.save_processed_data("val", val_samples)
            self.save_processed_data("test", test_samples)
            self.save_char_mapping()
            
            self.logger.info("前処理が完了しました")
        
        except Exception as e:
            self.logger.error(f"前処理中にエラーが発生しました: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="くずし字データセットの前処理")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="設定ファイルのパス"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="生データのディレクトリ"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="出力ディレクトリ"
    )
    
    args = parser.parse_args()
    logger = setup_logger(args.output_dir)
    
    preprocessor = DataPreprocessor(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        logger=logger
    )
    preprocessor.run()

if __name__ == "__main__":
    main()