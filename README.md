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

## データセットの準備と前処理

### 1. データセットの配置
生のくずし字データセットを `data/raw/dataset` に配置してください。

### 2. データの前処理
前処理スクリプトを実行して、データセットの準備を行います：

```bash
python scripts/preprocess_data.py \
    --config configs/model/kuzushiji_recognizer.yaml \
    --data-dir data/raw/dataset \
    --output-dir data
```

前処理の詳細については [docs/data_preprocessing.md](docs/data_preprocessing.md) を参照してください。

## モデルの学習

```bash
python src/training/trainer.py \
    --config configs/model/kuzushiji_recognizer.yaml \
    --data-dir data \
    --exp-dir experiments/exp_001 \
    --max-epochs 100
```

## 推論の実行

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

詳細な説明は以下のドキュメントを参照してください：
- [プロジェクト構造の詳細](project_structure_ja.md)
- [モデルアーキテクチャの詳細](docs/model_architecture.md)
- [データ前処理の詳細](docs/data_preprocessing.md)

## 実験結果

| モデル | 文字認識率 | 編集距離 |
|--------|------------|----------|
| ベースライン | 85.3% | 0.234 |
| マルチスケール | 87.1% | 0.215 |
| アンサンブル | 88.5% | 0.198 |

## 開発者向け情報

- コードの品質管理にはPylintを使用
- コミット前に `pre-commit` フックでコードフォーマット
- テストは `pytest` で実行

```bash
# テストの実行
pytest tests/

# コードの品質チェック
pylint src/
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 引用

このプロジェクトを研究で使用する場合は、以下の形式で引用してください：

```bibtex
@software{kuzushiji_vision2025,
  author = {Your Name},
  title = {Kuzushiji Vision: Deep Learning Model for Classical Japanese Character Recognition},
  year = {2025},
  url = {https://github.com/yourusername/kuzushiji-vision}
}
```

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

- **v1.0.0** (2025-02-04)
  - 初回リリース
  - 基本的なモデルアーキテクチャの実装
  - データ前処理パイプラインの追加
  - 訓練・評価スクリプトの追加