# Line Extraction モデルテスト計画

## 概要
学習済みのline extractionモデルの性能を評価し、結果を可視化するためのnotebookを実装します。

## 実装構成

### 1. 実験設定の読み込み
- 実験ディレクトリパスの設定（例：experiments/line_extraction/20250320_194943/）
- config.yamlの読み込み
- args.yamlの読み込み（訓練時の設定を再利用）
- 評価パラメータの設定（IoU閾値、バッチサイズなど）

```python
experiment_dir = "experiments/line_extraction/20250320_194943/"
model_path = os.path.join(experiment_dir, "weights/best.pt")
config_path = os.path.join(experiment_dir, "config.yaml")
```

### 2. データ準備
- テスト画像の読み込み
- アノテーションデータの読み込み
- 訓練時と同じ前処理パイプラインの適用
  - リサイズ（640x640）
  - 正規化
  - バッチ処理の準備

### 3. モデルのセットアップ
- 実験ディレクトリからbest.ptの読み込み
- 訓練時のモデル設定を使用
- 推論モードへの切り替え
- GPUの設定

### 4. 推論と評価
- バッチ処理による推論実行
- 評価指標の計算：
  - IoUスコア
  - Precision
  - Recall
  - F1-score
- 結果の集計とサマリー

### 5. 結果の可視化と分析
- 検出結果のプロット
  - 予測された列領域の表示
  - グラウンドトゥルースとの比較
- エラー分析
  - 誤検出ケースの表示
  - 未検出ケースの表示
- 性能指標の可視化
  - Precision-Recallカーブ
  - 信頼度スコアの分布

## 必要なライブラリ
```python
import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
from torchvision import transforms
```

## 評価指標
- Mean IoU
- Precision@IoU=0.5
- Recall@IoU=0.5
- F1-score@IoU=0.5

## 出力
1. 定量的評価結果のサマリー
2. 可視化された検出結果（成功例と失敗例）
3. エラー分析レポート

## 次のステップ
1. Codeモードに切り替えてnotebookを実装
2. 実装したnotebookでテストを実行
3. 結果に基づくモデルの改善提案