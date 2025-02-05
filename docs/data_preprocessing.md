# データ前処理の詳細

## 1. 概要

くずし字データセットの前処理パイプラインは以下の手順で実行されます：

1. データの読み込みと検証
2. データセットの分割（訓練/検証/テスト）
3. LMDB形式でのデータ保存
4. 文字マッピングの生成

## 2. 前処理の実行

```bash
python scripts/preprocess_data.py \
    --config configs/model/kuzushiji_recognizer.yaml \
    --data-dir data/raw \
    --output-dir data
```

### 引数の説明

- `--config`: モデル設定ファイルのパス
- `--data-dir`: 生データセットのディレクトリパス
- `--output-dir`: 前処理済みデータの出力ディレクトリ

## 3. 出力ディレクトリ構造

```
data/
├── processed/
│   └── char_mapping.json  # 文字とインデックスのマッピング
├── splits/
│   ├── train_metadata.json  # 訓練データのメタデータ
│   ├── val_metadata.json    # 検証データのメタデータ
│   └── test_metadata.json   # テストデータのメタデータ
├── lmdb/
│   ├── lmdb_train/  # 訓練データのLMDBデータベース
│   ├── lmdb_val/    # 検証データのLMDBデータベース
│   └── lmdb_test/   # テストデータのLMDBデータベース
└── preprocess.log    # 前処理のログ
```

## 4. 前処理の詳細

### 4.1 データの読み込みと検証

- 文書IDごとのディレクトリをスキャン
- 座標情報ファイルの存在確認
- 文字画像の収集とラベル付け
- 文字とインデックスのマッピング作成

### 4.2 データセットの分割

- 文書IDレベルでの分割（同じ文書が異なる分割に入らないように）
- デフォルトの分割比率:
  - 訓練: 80%
  - 検証: 10%
  - テスト: 10%

### 4.3 LMDB形式でのデータ保存

LMDBを使用する利点：
- 高速なデータアクセス
- メモリ効率の良い処理
- ディスクI/Oの最適化

保存されるデータ：
1. 画像データ（バイナリ形式）
2. メタデータ（JSON形式）：
   ```json
   {
     "image_key": "image_0",
     "doc_id": "100241706",
     "chars": [
       {
         "unicode": "U+4E00",
         "x": 0.85,
         "y": 0.12,
         "width": 0.05,
         "height": 0.08,
         "block_id": "B0001",
         "char_id": "C0001"
       },
       ...
     ],
     "width": 2000,
     "height": 3000
   }
   ```

### 4.4 文字マッピング

`char_mapping.json`に保存される情報：
```json
{
  "char_to_idx": {
    "U+4E00": 0,
    ...
  },
  "idx_to_char": {
    "0": "U+4E00",
    ...
  }
}
```

## 5. データ拡張

訓練時のデータ拡張：
```python
transforms = A.Compose([
    A.RandomRotate90(p=0.0),  # 縦書きなので回転は制限
    A.RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.5
    ),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## 6. エラー処理

- 不正なデータ構造の検出
- 破損画像のスキップ
- 処理失敗時のエラーログ
- クリーンアップ処理

## 7. メモリ使用量の最適化

- バッチ処理による効率的なデータ処理
- LMDBによるメモリ効率の向上
- 大規模データセット対応の設計

## 8. ログ出力

前処理の進捗とエラーは`preprocess.log`に記録：
- データ読み込みの統計
- 分割サイズの情報
- エラーや警告メッセージ
- 処理完了の確認