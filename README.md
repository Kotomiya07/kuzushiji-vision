# Kuzushiji Vision: 古典籍くずし字認識モデル

[![Ruff](https://github.com/Kotomiya07/kuzushiji-vision/actions/workflows/ruff.yml/badge.svg)](https://github.com/Kotomiya07/kuzushiji-vision/actions/workflows/ruff.yml)
[![License](https://img.shields.io/badge/License-Apache2.0-D22128.svg?logo=apache)](LICENSE)
[![Python](https://img.shields.io/badge/-Python-FFDD55.svg?logo=python)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/-Pytorch-F1F3F4.svg?logo=pytorch)](https://pytorch.org/)
[![Wandb](https://img.shields.io/badge/-WandB-F1F3F4.svg?logo=weightsandbiases)](https://wandb.ai/site/ja/)
[![Rye](https://img.shields.io/badge/-Rye-000000.svg?logo=rye)](https://rye.astral.sh/)

[English](README_en.md) | 日本語

## 概要

このプロジェクトは、日本の古典籍に含まれるくずし字を認識するための深層学習モデルを実装したものです。

## 環境構築

仮想環境は uv を使用して以下のコマンドを実行することで構築されます。

```bash
uv sync --extra build

uv sync --extra build --extra compile
```

## 言語モデルの事前学習

```bash
python scripts/concatenate_files.py
python train_tokenizer_one_char.py
python train_language_model.py
```

## YOLO 推論 Web アプリ

`scripts/yolo_inference_app.py` は FastAPI + htmx で、画像アップロードと文字検出（YOLO）の推論結果を確認できます。

```bash
uv run uvicorn scripts.yolo_inference_app:app --reload --host 0.0.0.0 --port 8000
```

## 文字位置検出（YOLO）の学習

`scripts/train_character_detection.py` は `src/configs/model/character_detection.yaml` を参照して学習します。

```bash
uv run python scripts/train_character_detection.py
```

P2（stride=4）を含む検出ヘッド（P2-P5）を使う場合は、`model.backbone` に YAML（例: `yolov12/ultralytics/cfg/models/v12/yolov12x.yaml`）を指定してください。

既存の学習済み重み（`.pt`）から「形が合う層だけ」を読み込み、新規追加した P2 系や Detect などは初期化したまま学習を続けたい場合は、同設定の `model.pretrained_weights` に `.pt` のパスを指定します。
