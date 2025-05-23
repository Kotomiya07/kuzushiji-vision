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
仮想環境はuvを使用して以下のコマンドを実行することで構築されます。

```bash
# dev と flash-attn のグループを抜いて sync　する
uv sync --no-group dev --no-group flash-attn

# その後 dev のグループを sync する (実行環境の場合はなくても OK)
uv sync --group dev

# 最後に flash-attn のグループを sync する
uv sync --group flash-attn
```

## 言語モデルの事前学習
```bash
python scripts/concatenate_files.py
python train_tokenizer_one_char.py
python train_language_model.py
```
