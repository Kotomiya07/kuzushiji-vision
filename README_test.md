# OCRモデル テスト・可視化スクリプト

## 基本使用方法

```bash
python test_and_visualize.py \
    --model_path experiments/ocr_model/best_model.ckpt \
    --data_dir data/column_dataset_padded
```

## 主要オプション

- `--model_path`: 学習済みモデルのチェックポイントパス（必須）
- `--data_dir`: テストデータディレクトリ（必須）
- `--output_dir`: 結果出力ディレクトリ（デフォルト: test_results）
- `--max_samples`: 処理する最大サンプル数（デフォルト: 全て）

## 出力

- 文字認識結果の可視化画像
- バウンディングボックス比較画像
- 評価結果CSV
- サマリーレポート 
