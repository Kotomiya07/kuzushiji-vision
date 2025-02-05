#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml
from tqdm import tqdm

from src.models.arch.kuzushiji_recognizer import load_model

def load_image(image_path: str) -> torch.Tensor:
    """画像の読み込みと前処理"""
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # 正規化
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # バッチ次元の追加
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

def denormalize_coordinates(
    coords: List[float],
    image_size: Tuple[int, int]
) -> List[int]:
    """正規化された座標を元のスケールに戻す"""
    W, H = image_size
    x, y, w, h = coords
    return [
        int(x * W),
        int(y * H),
        int(w * W),
        int(h * H)
    ]

def get_font(font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
    """フォントの取得"""
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size=24)
        except Exception:
            return ImageFont.load_default()
    return ImageFont.load_default()

def visualize_results(
    image_path: str,
    results: List[Dict],
    output_path: str,
    font_path: Optional[str] = None
) -> None:
    """結果の可視化"""
    # 画像の読み込み
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # フォントの設定
    font = get_font(font_path)
    
    # 文字と位置の描画
    for result in results:
        x, y, w, h = denormalize_coordinates(
            result['position'],
            image.size
        )
        
        # 文字位置の矩形を描画
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline='red',
            width=2
        )
        
        # Unicodeから文字に変換
        char = chr(result['unicode'])
        
        # 文字とConfidenceを描画
        text = f"{char} ({result['confidence']:.2f})"
        draw.text(
            (x, y - 24),
            text,
            fill='red',
            font=font
        )
    
    # 結果の保存
    image.save(output_path)

def infer_page(
    model: torch.nn.Module,
    image_path: str,
    confidence_threshold: float = 0.5,
    output_dir: Optional[str] = None,
    visualize: bool = False,
    font_path: Optional[str] = None,
    device: str = 'cuda'
) -> List[Dict]:
    """1ページの推論を実行"""
    # 画像の読み込み
    image = load_image(image_path).to(device)
    
    # 推論の実行
    with torch.no_grad():
        results = model.decode_page(
            image,
            confidence_threshold=confidence_threshold
        )
    
    # 結果の整形
    formatted_results = []
    for result in results:
        char = chr(result['unicode'])
        formatted_results.append({
            'unicode': result['unicode'],
            'char': char,
            'position': result['position'],
            'confidence': result['confidence']
        })
    
    # 結果の可視化（オプション）
    if visualize and output_dir:
        output_name = Path(image_path).stem + '_viz.png'
        output_path = os.path.join(output_dir, output_name)
        visualize_results(
            image_path,
            formatted_results,
            output_path,
            font_path
        )
    
    return formatted_results

def process_file(
    model: torch.nn.Module,
    image_path: Path,
    output_dir: str,
    confidence_threshold: float,
    visualize: bool,
    font_path: Optional[str],
    device: str
) -> None:
    """単一ファイルの処理"""
    results = infer_page(
        model,
        str(image_path),
        confidence_threshold=confidence_threshold,
        output_dir=output_dir if visualize else None,
        visualize=visualize,
        font_path=font_path,
        device=device
    )
    
    # 結果の保存
    output_name = image_path.stem + '.json'
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def process_directory(
    model: torch.nn.Module,
    input_dir: Path,
    output_dir: str,
    confidence_threshold: float,
    visualize: bool,
    font_path: Optional[str],
    device: str
) -> None:
    """ディレクトリ内の全画像を処理"""
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    for image_file in tqdm(image_files, desc='Processing pages'):
        process_file(
            model,
            image_file,
            output_dir,
            confidence_threshold,
            visualize,
            font_path,
            device
        )

def main():
    parser = argparse.ArgumentParser(description='くずし字ページ認識の推論')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='モデルチェックポイントのパス'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='入力画像または画像ディレクトリのパス'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='出力ディレクトリ'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='検出の確信度閾値'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='結果を可視化するかどうか'
    )
    parser.add_argument(
        '--font-path',
        type=str,
        help='可視化用フォントのパス'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='使用するデバイス（cuda/cpu）'
    )
    
    args = parser.parse_args()
    
    # 設定の読み込み
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # モデルの読み込み
    model = load_model(args.checkpoint, config).to(args.device)
    model.eval()
    
    # 入力パスの処理
    input_path = Path(args.input)
    if input_path.is_file():
        process_file(
            model,
            input_path,
            args.output_dir,
            args.confidence_threshold,
            args.visualize,
            args.font_path,
            args.device
        )
    else:
        process_directory(
            model,
            input_path,
            args.output_dir,
            args.confidence_threshold,
            args.visualize,
            args.font_path,
            args.device
        )

if __name__ == '__main__':
    main()