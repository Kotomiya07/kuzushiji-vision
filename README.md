# Kuzushiji Vision: 古典籍くずし字認識モデル

[English](README_en.md) | 日本語

## 概要

このプロジェクトは、日本の古典籍に含まれるくずし字を認識するための深層学習モデルを実装したものです。ResNet50とTransformerを組み合わせたエンコーダー・デコーダーモデルにより、高精度な文字認識を実現します。

## 特徴

- **ハイブリッドアーキテクチャ**
  - ResNet50 + FPNによる効率的な特徴抽出
  - Transformerエンコーダー・デコーダーによる文字認識
  - マルチスケール特徴処理による高精度化

- **最適化された学習プロセス**
  - PyTorch Lightningベースの実装
  - 混合精度学習のサポート
  - 効率的なデータローディングと拡張

- **充実した実験管理**
  - 設定ファイルベースの実験管理
  - TensorBoardによる学習過程の可視化
  - チェックポイントと早期停止の実装

## 必要環境

- Python 3.8+
- CUDA 11.0+（GPU使用時）
- 12GB以上のGPU VRAM推奨

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/kuzushiji-vision.git
cd kuzushiji-vision

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
.\venv\Scripts\activate  # Windows

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### データセットの準備

1. くずし字データセットを `data/raw/dataset` に配置
2. データの前処理を実行:
```bash
python scripts/preprocess_data.py
```

### 学習の実行

```bash
python src/training/trainer.py \
    --config configs/model/kuzushiji_recognizer.yaml \
    --data-dir data \
    --exp-dir experiments/exp_001 \
    --max-epochs 100
```

### 推論の実行

```bash
python scripts/infer.py \
    --config configs/model/kuzushiji_recognizer.yaml \
    --checkpoint experiments/exp_001/checkpoints/best.ckpt \
    --input-image path/to/image.jpg
```

## プロジェクト構造

```
kuzushiji-vision/
├── data/                  # データ管理
│   ├── raw/               # 生データ（変更不可）
│   ├── processed/         # 前処理済みデータ
│   └── splits/            # データ分割設定
├── configs/               # 実験設定
├── src/                   # ソースコード
│   ├── data/             # データ処理
│   ├── models/           # モデル定義
│   ├── training/         # 訓練関連
│   └── evaluation/       # 評価関連
├── experiments/          # 実験記録
├── docs/                 # ドキュメント
└── scripts/              # 実行スクリプト
```

詳細なディレクトリ構造は [project_structure_ja.md](project_structure_ja.md) を参照してください。

## モデルアーキテクチャ

![モデルアーキテクチャ](docs/images/model_architecture.png)

モデルの詳細な説明は [docs/model_architecture.md](docs/model_architecture.md) を参照してください。

## 実験結果


## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。


## 謝辞

- くずし字データセットを提供していただいた関係者の皆様に感謝いたします
- PyTorch、PyTorch Lightning、その他のオープンソースプロジェクトのコミュニティに感謝いたします

## お問い合わせ

- Issue Trackerをご利用ください
- または、[email@example.com](mailto:email@example.com) までご連絡ください

## コントリビューション

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/awesome-feature`)
3. 変更をコミット (`git commit -am 'Add awesome feature'`)
4. ブランチをプッシュ (`git push origin feature/awesome-feature`)
5. プルリクエストを作成

## 更新履歴

- **v1.0.0** (2025-01-29)
  - 初回リリース
  - 基本的なモデルアーキテクチャの実装
  - 訓練・評価スクリプトの追加