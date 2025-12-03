# 学習済みモデルのテストスクリプト

このディレクトリには、学習済みのくずし字言語モデルをテストするためのスクリプトが含まれています。

## ファイル構成

- `test_trained_model.py`: 対話式テストスクリプト（メイン）
- `demo_test_model.py`: デモテストスクリプト（予定義された例文でテスト）
- `quick_test.py`: クイックテストスクリプト（簡単な使用方法）

## 使用方法

### 1. 対話式テスト

```bash
python test_trained_model.py --model_path [モデルパス] --tokenizer_path [トークナイザーパス]
```

**例:**

```bash
# 基本的な使用
python test_trained_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model

# トークナイザーを別途指定
python test_trained_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model --tokenizer_path experiments/kuzushiji_tokenizer_one_char

# 上位10位まで表示
python test_trained_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model --top_k 10
```

### 2. デモテスト

```bash
python demo_test_model.py --model_path [モデルパス]
```

**例:**

```bash
python demo_test_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model
```

### 3. クイックテスト

```bash
python quick_test.py
# または
python quick_test.py --usage  # 使用方法を表示
```

## マスクトークンの使用方法

文中に `[MASK]` を含めることで、その部分をマスクトークンとして処理できます。

**例:**

- `今日は[MASK]い天気です`
- `この[MASK]は美しい`
- `[MASK]が好きです`
- `昔[MASK]あるところに`
- `[MASK]は[MASK]です` (複数マスク)

## 出力形式

スクリプトは以下の情報を表示します：

1. **入力文**: 元の入力文
2. **処理後文**: トークナイザーで処理された文
3. **マスク数**: マスクされたトークンの数
4. **各マスク位置の予測**: 各マスク位置での Top-K 予測結果
5. **復元された文**: Top-K 予測で復元された文

## オプション

### test_trained_model.py のオプション

- `--model_path`: 学習済みモデルのパス（必須）
- `--tokenizer_path`: トークナイザーのパス（省略時は model_path と同じ）
- `--top_k`: 予測結果の上位何位まで表示するか（デフォルト: 5）
- `--batch_mode`: バッチモード（複数の文を一度に処理）

### demo_test_model.py のオプション

- `--model_path`: 学習済みモデルのパス（必須）
- `--tokenizer_path`: トークナイザーのパス（省略時は model_path と同じ）
- `--top_k`: 予測結果の上位何位まで表示するか（デフォルト: 5）

## 実行例

### 対話式テスト

```
$ python test_trained_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model

くずし字言語モデル テスター
================================================================================
使用方法:
  - [MASK]を含む文を入力してください
  - 例: 今日は[MASK]い天気です
  - 終了するには 'quit' または 'exit' を入力してください
================================================================================

対話モード: 文を一つずつ入力してください

文を入力: 今日は[MASK]い天気です

================================================================================
マスクトークン予測結果
================================================================================
入力文:     今日は[MASK]い天気です
処理後文:   今日は<mask>い天気です
マスク数:   1

------------------------------------------------------------
各マスク位置の予測:
------------------------------------------------------------

マスク位置 1 (トークン位置: 3):
  1. '良' (確率: 0.4521)
  2. '悪' (確率: 0.2134)
  3. '美' (確率: 0.1876)
  4. '素' (確率: 0.0987)
  5. '暖' (確率: 0.0482)

------------------------------------------------------------
復元された文:
------------------------------------------------------------
Top1: 今日は良い天気です
Top2: 今日は悪い天気です
Top3: 今日は美い天気です
Top4: 今日は素い天気です
Top5: 今日は暖い天気です

================================================================================
```

### デモテスト

```
$ python demo_test_model.py --model_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model

くずし字言語モデル デモテスト
================================================================================
モデルパス: experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model
トークナイザーパス: experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20240101_120000/final_model
テスト文数: 23
表示する予測数: 5
================================================================================

==================== テスト  1/23 ====================
...
```

## 注意事項

1. **モデルパス**: 学習済みモデルのパスを正しく指定してください
2. **トークナイザー**: トークナイザーのパスが異なる場合は明示的に指定してください
3. **CUDA**: GPU が利用可能な場合は自動的に CUDA を使用します
4. **メモリ**: 大きなモデルの場合、十分なメモリが必要です

## トラブルシューティング

### モデルが見つからない場合

```bash
# モデルディレクトリを確認
ls -la experiments/pretrain_language_model/

# 最新のモデルを確認
find experiments/pretrain_language_model/ -name "final_model" -type d
```

### CUDA 関連のエラー

```python
# CPUを強制的に使用する場合
import torch
torch.cuda.is_available = lambda: False
```

### メモリ不足エラー

- より小さな batch_size を使用
- CPU モードで実行
- より小さなモデルを使用

## 開発者向け情報

### KuzushijiModelTester クラス

メインのテストクラスで、以下の機能を提供：

- `__init__(model_path, tokenizer_path)`: モデルとトークナイザーの初期化
- `predict_masked_tokens(text, top_k)`: マスクトークンの予測
- `print_prediction_results(results, top_display)`: 結果の表示

### 拡張方法

カスタムテストを追加する場合：

```python
from test_trained_model import KuzushijiModelTester

# テスターを初期化
tester = KuzushijiModelTester(model_path, tokenizer_path)

# カスタムテスト
results = tester.predict_masked_tokens("カスタム[MASK]テスト", top_k=10)
tester.print_prediction_results(results)
```
