
# `train_character_detection.py` 再実装ガイド

## 1. 概要

このスクリプト (`trainer/train_character_detection.py`) は、くずし字画像内の文字を検出するための深層学習モデルを訓練することを目的としています。主な機能は以下の通りです。

*   設定ファイルに基づいた訓練パイプラインの実行。
*   文字検出モデル (`CharacterDetectionModel`) の訓練。
*   訓練データと検証データを用いたモデルの評価 (mAP, 文字精度)。
*   Hugging Face Accelerate を利用した混合精度訓練のサポート。
*   Weights & Biases (WandB) を用いた実験追跡とロギング。
*   Cosine Annealing 学習率スケジューラと早期停止の実装。
*   検証時の予測結果の可視化。

## 2. 依存関係

このスクリプトを実行するには、以下の主要なライブラリが必要です。

*   `torch`: 深層学習フレームワーク。
*   `torchvision`: 画像処理と物体検出モデル。
*   `accelerate`: Hugging Face の混合精度訓練ライブラリ。
*   `timm`: 画像モデルと学習率スケジューラ。
*   `numpy`: 数値計算ライブラリ。
*   `matplotlib`: グラフ描画、予測結果の可視化。
*   `pyyaml`: 設定ファイルの読み込み。
*   `tqdm`: プログレスバー表示。
*   `wandb`: 実験追跡とロギング。
*   `collections.abc`: 型チェック用。
*   `pathlib`: ファイルパス操作。

プロジェクト内のカスタムモジュール:

*   `models.character_detection.model.CharacterDetectionModel`: 文字検出モデル。
*   `utils.augmentation.get_character_detection_transforms`: データ拡張関数。
*   `utils.dataset.CharacterDetectionDataset`: データセットクラス。
*   `utils.metrics`: 評価指標計算関数 (`compute_map`, `compute_character_accuracy`)。
*   `utils.util.EasyDict`: 辞書ライクな属性アクセスを提供するクラス。

## 3. 設定ファイル

訓練の挙動は `config/model/character_detection.yaml` ファイルによって制御されます。このファイルには以下の主要なセクションが含まれます。

*   `seed`: 乱数シード。
*   `data`: データセット関連の設定。
    *   `train_annotation`: 訓練用アノテーションファイルパス。
    *   `val_annotation`: 検証用アノテーションファイルパス。
    *   `test_annotation`: テスト用アノテーションファイルパス。
    *   `unicode_dict`: Unicodeからidへの辞書ファイルパス。
*   `model`: モデルアーキテクチャとパラメータ。
    *   `input_size`: モデルへの入力画像サイズ。
    *   `confidence_threshold`: 予測時の信頼度スコア閾値。
    *   その他モデル固有のパラメータ (例: バックボーン、ヘッド)。
*   `training`: 訓練パラメータ。
    *   `batch_size`: バッチサイズ。
    *   `num_workers`: データローダーのワーカ数。
    *   `learning_rate`: 初期学習率。
    *   `weight_decay`: 重み減衰。
    *   `early_stopping_patience`: 早期停止のpatience。
    *   `scheduler`: 学習率スケジューラの設定。
        *   `t_initial`: CosineLRScheduler のエポック数。
        *   `lr_min`: 最小学習率。
        *   `warmup_t`: ウォームアップエポック数。
        *   `warmup_lr_init`: ウォームアップ開始時の学習率。
*   `evaluation`: 評価時の設定。
    *   `iou_threshold`: mAP や精度計算時の IoU 閾値。
    *   `num_visualizations`: 保存する可視化サンプル数。
*   `experiment`: 実験結果の保存設定。
    *   `save_dir`: 実験結果の保存先ディレクトリパターン。
    *   `checkpoint_dir`: モデルチェックポイントのサブディレクトリ名。
    *   `eval_dir`: 評価結果（可視化画像など）のサブディレクトリ名。
*   `wandb`: Weights & Biases の設定。
    *   `project`: プロジェクト名。
    *   `entity`: WandB のエンティティ名 (ユーザー名またはチーム名)。
    *   `name_format`: 実行名のフォーマット。

スクリプト内では、YAML から読み込んだ辞書を `EasyDict` オブジェクトに変換し、属性アクセス (`config.training.batch_size`) を可能にしています。WandB に設定を記録する際には `recursive_to_dict` で通常の辞書に戻します。

## 4. 主要コンポーネント

### 4.1. `main()` 関数

スクリプトのエントリーポイントであり、全体のワークフローを制御します。

1.  **Accelerator 初期化**: `Accelerator(log_with="wandb")` で初期化し、ロギングの準備をします。
2.  **設定読み込み**: `config/model/character_detection.yaml` を読み込み、`EasyDict` に変換します。
3.  **乱数シード設定**: `set_seed()` で再現性を確保します。
4.  **データセットとデータローダー準備**:
    *   `get_character_detection_transforms()` でデータ拡張を定義します。
    *   `CharacterDetectionDataset` で訓練・検証データセットを作成します。
    *   `DataLoader` でデータローダーを作成します。
5.  **モデル初期化**: `CharacterDetectionModel(config)` でモデルをインスタンス化します。
6.  **Optimizer と Scheduler 設定**:
    *   `AdamW` オプティマイザを定義します。
    *   `timm.scheduler.CosineLRScheduler` を設定ファイルに基づいて定義します。
7.  **Accelerator 準備**: `accelerator.prepare()` でモデル、オプティマイザ、データローダーをラップし、分散学習に対応させます。
8.  **WandB 初期化**: メインプロセスでのみ `accelerator.init_trackers()` を呼び出し、WandB でのロギングを開始します。設定情報は `recursive_to_dict` で辞書に変換してから渡します。
9.  **実験ディレクトリ作成**: メインプロセスでのみ、設定に基づいて実験結果やチェックポイントを保存するディレクトリを作成します。
10. **早期停止準備**: `EarlyStopping` クラスをインスタンス化します。
11. **訓練ループ**:
    *   設定されたエポック数だけループします。
    *   `train_epoch()` を呼び出して1エポック分の訓練を実行します。
    *   `validate()` を呼び出して検証データで評価します。
    *   学習率スケジューラ (`scheduler.step()`) を更新します。
    *   訓練損失、検証メトリクス (mAP, 精度)、学習率を `accelerator.log()` で WandB に記録します。
    *   検証 mAP が改善した場合、メインプロセスで `accelerator.save()` を用いてモデルの `state_dict` を保存します。
    *   `early_stopping()` を呼び出し、早期停止条件をチェックします。
12. **終了処理**: `accelerator.end_training()` で WandB トラッカーなどを終了します。

### 4.2. `train_epoch()` 関数

1エポック分の訓練ロジックを担当します。

1.  **モデルを訓練モードに**: `model.train()` を呼び出します。
2.  **損失記録用辞書初期化**: `defaultdict(float)` を使って、損失の種類ごとに合計値を記録する辞書を初期化します。
3.  **データローダーでループ**: `tqdm` でプログレスバーを表示しながら、訓練データローダーからバッチを取得します。
4.  **データ準備**: バッチから画像とターゲット (ボックス、ラベル) を取り出します。Accelerator が自動的に適切なデバイスに転送します。
5.  **勾配累積**: `accelerator.accumulate(model)` コンテキスト内で以下の処理を行います。
    *   **順伝播**: `loss_dict = model(images, targets)` を実行し、損失の辞書を取得します。モデルは内部で損失計算まで行います。
    *   **損失取得**: `loss = loss_dict["loss"]` で逆伝播に使用する合計損失を取得します。
    *   **逆伝播**: `accelerator.backward(loss)` で勾配を計算します。
    *   **勾配同期とクリッピング**: `accelerator.sync_gradients` が True の場合 (勾配累積の最後)、`accelerator.clip_grad_norm_()` で勾配クリッピングを行います。
    *   **パラメータ更新**: `optimizer.step()` でモデルのパラメータを更新します。
    *   **勾配リセット**: `optimizer.zero_grad()` で次のイテレーションのために勾配をリセットします。
6.  **損失集約**: `accelerator.gather()` を使って、全プロセスから各損失値 (Tensor) を集約し、平均を計算して `.item()` で Python の float に変換します。
7.  **エポック損失記録**: 集約した損失値を `total_losses` 辞書に加算します。
8.  **プログレスバー更新**: `progress_bar.set_postfix()` で現在のバッチの合計損失を表示します。
9.  **エポック平均損失計算**: ループ終了後、`total_losses` をバッチ数で割り、エポック全体の平均損失辞書を計算します。
10. **結果返却**: 平均損失の辞書を返します。

### 4.3. `validate()` 関数

検証データセットを用いてモデルの評価を行います。

1.  **モデルを評価モードに**: `model.eval()` を呼び出します。
2.  **評価設定取得**: `config.evaluation` から IoU 閾値や可視化サンプル数を取得します。
3.  **結果格納用リスト初期化**: 各プロセス *ローカル* で予測結果 (ボックス、スコア、ラベル) と正解データ (ボックス、ラベル) を格納するためのリストを初期化します。可視化用のデータリスト (`viz_data_list`) もメインプロセス用に初期化します。
4.  **データローダーでループ**: `@torch.no_grad()` デコレータにより勾配計算を無効化し、検証データローダーからバッチを取得します。
5.  **推論実行**: `predictions = model(images)` を実行します。モデルは推論モードでは、ボックス、スコア、ラベルのリストを含む辞書を返します。
6.  **ローカル結果蓄積**: 各プロセスのバッチに対する予測結果と正解データを `.cpu()` でCPUに転送し、ローカルリストに追加します。
7.  **可視化データ収集 (メインプロセスのみ)**:
    *   メインプロセス (`accelerator.is_local_main_process`) で、かつ可視化が必要な場合、設定された `num_visualizations` に達するまで、現在のバッチから可視化に必要なデータ (画像、予測、正解) をCPUに転送し、`viz_data_list` に追加します。
8.  **全プロセスから結果収集**: ループ終了後、`accelerator.gather_object()` を使用して、各プロセスに蓄積されたローカルリスト (予測ボックス、スコア、ラベル、正解ボックス、ラベル) をメインプロセスに集約します。`gather_object` はリストのリストを返すため、これをフラットなリストに変換します。
9.  **メトリクス計算 (メインプロセスのみ)**:
    *   集約された予測結果と正解データのリストを用いて `compute_map()` と `compute_character_accuracy()` を呼び出し、mAP と文字精度を計算します。
    *   リストが空の場合や計算中にエラーが発生した場合は、メトリクスを 0.0 とし、警告やエラーログを出力します。
10. **可視化実行 (メインプロセスのみ)**:
    *   `viz_data_list` に収集されたデータを使って `visualize_predictions()` を呼び出し、予測結果を画像として保存します。
    *   WandB が有効な場合、生成された画像を `wandb.Image` としてログに記録します。
11. **結果のブロードキャスト**: `accelerator.wait_for_everyone()` で全プロセスが同期するのを待った後、メインプロセスで計算された mAP と精度を Tensor に変換し、`accelerator.broadcast()` を使って他の全プロセスに送信します。
12. **結果返却**: 全プロセスが同じ mAP と精度の値 (float) を返すようにします。

### 4.4. `visualize_predictions()` 関数

単一の画像に対する予測結果と正解データを可視化し、ファイルに保存します。

1.  **画像前処理**: 入力画像テンソル (正規化済み) を `numpy` 配列に変換し、逆正規化して [0, 1] の範囲にクリップします。
2.  **プロット準備**: `matplotlib.pyplot.subplots` で Figure と Axes を作成し、画像を表示します。
3.  **予測ボックス描画 (赤色)**:
    *   予測ボックス、ラベル、スコアを `.cpu().numpy()` で NumPy 配列に変換します。
    *   設定された `score_threshold` を超える予測についてループします。
    *   `matplotlib.patches.Rectangle` を使ってバウンディングボックスを描画します。
    *   `ax.text` でボックスの左上にラベルとスコアを表示します。
    *   無効なボックス (幅や高さが0以下) はスキップします。エラー発生時も警告を出力して継続します。
4.  **正解ボックス描画 (緑色)**:
    *   正解ボックスとラベルを NumPy 配列に変換します。
    *   同様に `Rectangle` と `ax.text` を使って正解ボックスとラベルを描画します。
    *   無効なボックスはスキップし、エラー発生時も警告を出力して継続します。
5.  **プロット整形**: 軸の目盛りや枠線を非表示にします (`ax.axis('off')`)。
6.  **保存**: `save_path` で指定されたパスに画像を保存します。`plt.savefig()` を使用し、`bbox_inches='tight', pad_inches=0` で余白を最小限にします。保存先ディレクトリが存在しない場合は作成します。
7.  **メモリ解放**: `plt.close(fig)` で Figure オブジェクトを閉じてメモリを解放します。

### 4.5. `collate_fn()` 関数

`DataLoader` がミニバッチを作成する際に使用されるカスタム関数です。`CharacterDetectionDataset` から返される辞書のリストを受け取り、モデル入力に適した形式の単一の辞書にまとめます。

1.  **データ抽出**: バッチ内の各アイテム (辞書) から "image", "boxes", "labels", "image_id", "image_path" を抽出し、対応するリストに追加します。キーが存在しない、または型が不正なアイテムはスキップし、警告を出力します。
2.  **空バッチ処理**: 有効なアイテムが一つもなかった場合、空のテンソルやリストを含む辞書を返します。
3.  **パディング**:
    *   バッチ内の画像の最大高 (`max_h`) を見つけます。
    *   各画像について、高さが `max_h` になるように `torch.nn.functional.pad` を使って下側に 0 でパディングします。幅は変更しません。
4.  **スタッキング**: パディングされた画像のリストを `torch.stack()` で結合し、単一のバッチテンソル `[B, C, H_max, W]` を作成します。スタック時にエラーが発生した場合は、エラーログを出力し、空のバッチを返します。
5.  **結果返却**: 以下のキーを持つ辞書を返します。
    *   `image`: パディング・スタックされた画像テンソル。
    *   `boxes`: ボックス座標テンソルのリスト (パディングなし)。
    *   `labels`: ラベルテンソルのリスト (パディングなし)。
    *   `image_id`: 画像 ID のテンソル。
    *   `image_path`: 画像パスのリスト。

### 4.6. `EarlyStopping` クラス

検証セットでの性能向上に基づいて訓練を早期に終了させるためのクラスです。

*   `__init__(patience, min_delta)`: `patience` (性能が改善しないことを許容するエポック数) と `min_delta` (改善とみなす最小の変化量) で初期化します。
*   `__call__(score)`: 現在のエポックのスコア (例: mAP) を受け取ります。
    *   最初の呼び出しでは、`best_score` を現在のスコアで初期化します。
    *   現在のスコアが `best_score + min_delta` より大きい場合、`best_score` を更新し、`counter` をリセットします。
    *   スコアが改善しない場合、`counter` をインクリメントします。
    *   `counter` が `patience` に達した場合、`should_stop` フラグを True に設定します。
    *   `should_stop` フラグ (True なら停止) を返します。

### 4.7. ヘルパー関数

*   `recursive_easydict(d)`: ネストされた辞書を再帰的に `EasyDict` に変換します。リストやタプル内の辞書も変換します。
*   `recursive_to_dict(d)`: `EasyDict` オブジェクトを再帰的に通常の Python 辞書に変換します。WandB に設定を渡す際に使用されます。

## 5. データ準備

### 5.1. `CharacterDetectionDataset`

*   アノテーションファイル (CSV) と画像ベースディレクトリを読み込み、画像パス、バウンディングボックス、文字ラベルを提供します。
*   `__getitem__` メソッドは、指定されたインデックスに対応する画像、ボックス、ラベルを含む辞書を返します。画像は指定された `target_width` にリサイズされ、アスペクト比を維持した上で高さが調整されます。
*   データ拡張 (`transform`) が指定されていれば、画像とボックスに適用します。
*   文字クラスの数 (`num_classes`) を自動的に計算します。

### 5.2. `get_character_detection_transforms`

*   設定ファイル (`config.model.augmentation`) に基づいて、訓練用のデータ拡張パイプライン (例: `albumentations` ライブラリを使用) を作成します。
*   一般的な拡張には、ランダムな輝度・コントラスト調整、ガウシアンノイズ、水平反転などが含まれる可能性があります。
*   最後に、画像を Tensor に変換し、正規化する処理が含まれます。

## 6. モデル

### 6.1. `CharacterDetectionModel`

*   設定ファイル (`config.model`) に基づいて文字検出モデルを構築します。
*   通常、事前学習済みのバックボーン (例: ResNet, EfficientNet) と、物体検出ヘッド (例: Faster R-CNN, RetinaNet, またはカスタムヘッド) から構成されます。`torchvision.models.detection` のモデルを利用するか、カスタム実装を使用します。
*   `forward(images, targets=None)` メソッド:
    *   訓練時 (`targets` が提供される場合): 画像とターゲットを受け取り、損失計算まで行い、損失の辞書 (`{'loss': total_loss, 'detection_loss': ..., 'classification_loss': ...}`) を返します。
    *   推論時 (`targets` が None の場合): 画像のみを受け取り、予測結果のリスト (`[{'boxes': tensor, 'scores': tensor, 'labels': tensor}, ...]`) を返します。各要素はバッチ内の1画像に対応します。
*   `set_epoch()` メソッドを持つ場合があり、エポック番号に依存する処理 (例: 特定のスケジューリング) を行うために訓練ループから呼び出されます。

## 7. 訓練プロセス

### 7.1. Accelerator

*   分散学習 (Multi-GPU, TPU) と混合精度訓練 (AMP) を容易にします。
*   `accelerator.prepare()`: モデル、オプティマイザ、データローダーをラップします。
*   `accelerator.backward()`: 混合精度や分散学習を考慮して勾配計算を行います。
*   `accelerator.gather()` / `accelerator.gather_object()`: 異なるプロセスからテンソルやオブジェクトを集約します。
*   `accelerator.is_local_main_process`: 現在のプロセスがメインプロセスかどうかを判定します (ファイル書き込みやロギングの制御に使用)。
*   `accelerator.log()`: 接続されたトラッカー (WandB など) にメトリクスを記録します。
*   `accelerator.save()` / `accelerator.load_state()`: 分散環境で安全にモデルの状態を保存・読み込みします。

### 7.2. Optimizer と Scheduler

*   **Optimizer**: `AdamW` が使用され、学習率と重み減衰は設定ファイルから取得されます。
*   **Scheduler**: `timm.scheduler.CosineLRScheduler` が使用されます。
    *   エポックごとにコサインカーブに従って学習率を減衰させます。
    *   オプションでウォームアップ期間を設定できます。
    *   `scheduler.step(epoch)` のように、エポック番号を渡して訓練ループ内で手動で更新されます (Accelerator による自動準備は行われません)。

### 7.3. WandB ロギング

*   `accelerator = Accelerator(log_with="wandb")` で WandB との連携が有効になります。
*   `accelerator.init_trackers()`: WandB の実行 (`wandb.init`) を初期化します。プロジェクト名、実行名、設定 (`config`) が渡されます。
*   `accelerator.log(log_data, step=epoch_num)`: 訓練損失、検証メトリクス、学習率などの辞書を WandB に記録します。`step` 引数でエポック番号を指定します。
*   検証時の可視化画像も `wandb.log({"val_predictions/sample_X.png": wandb.Image(...)})` のように記録されます。

### 7.4. チェックポイント

*   検証 mAP が過去最高を記録した場合に、メインプロセスでモデルの `state_dict` が保存されます。
*   `accelerator.unwrap_model(model)` で Accelerator によってラップされた元のモデルを取得し、その `state_dict` を `accelerator.save()` を使って保存します。これにより、分散環境でも正しく保存されます。
*   保存パスは設定ファイルの `experiment.save_dir` と `experiment.checkpoint_dir` に基づいて決定されます。

## 8. 評価

### 8.1. メトリクス

*   **mAP (mean Average Precision)**: 物体検出の標準的な評価指標。`utils.metrics.compute_map` で計算されます。異なる IoU 閾値での Average Precision を計算し、その平均を取ります (このスクリプトでは単一の IoU 閾値を使用)。
*   **Character Accuracy**: 検出された文字のクラス分類精度。`utils.metrics.compute_character_accuracy` で計算されます。正しく検出されたボックス (IoU 閾値以上) の中で、ラベルが一致する割合を計算するなどの方法が考えられます (実装の詳細によります)。

### 8.2. 可視化

*   `validate` 関数内で、設定された数の検証サンプルについて予測結果と正解データを重ねて描画した画像が生成されます (`visualize_predictions` を使用)。
*   これらの画像は実験ディレクトリ内の `evaluation/epoch_X` フォルダに保存され、WandB にもアップロードされます。

## 9. 再実装のためのステップ

1.  **環境設定**: 必要なライブラリ (`requirements.txt` や `pyproject.toml` を参照) をインストールします。
2.  **設定ファイルの準備**: `config/model/character_detection.yaml` を作成またはコピーし、データパス、モデルパラメータ、訓練設定などを適切に構成します。
3.  **ユーティリティ関数の実装**:
    *   `EasyDict` クラス (または `collections.namedtuple` や `types.SimpleNamespace` で代用)。
    *   `recursive_easydict`, `recursive_to_dict` 関数。
    *   `EarlyStopping` クラス。
    *   `collate_fn` 関数 (パディングロジックを含む)。
    *   `visualize_predictions` 関数。
    *   評価指標関数 (`compute_map`, `compute_character_accuracy`)。これらは既存のライブラリ (例: `torchmetrics`) を利用するか、カスタム実装が必要です。
4.  **データ準備の実装**:
    *   `CharacterDetectionDataset` クラスを実装します。CSV アノテーションの読み込み、画像のリサイズ、ターゲット形式の整形を行います。
    *   `get_character_detection_transforms` 関数を実装します (`albumentations` などを使用)。
5.  **モデルの実装**:
    *   `CharacterDetectionModel` クラスを実装します。設定に基づいてバックボーンとヘッドを組み合わせ、`forward` メソッドで訓練時の損失計算と推論時の予測出力を行います。
6.  **訓練・検証関数の実装**:
    *   `train_epoch` 関数を実装します (ループ、順伝播、逆伝播、損失記録)。
    *   `validate` 関数を実装します (ループ、推論、結果収集、メトリクス計算、可視化呼び出し)。
7.  **メイン関数の実装 (`main`)**:
    *   上記コンポーネントを組み合わせて、設定の読み込みから訓練ループ、結果の保存、ロギングまでの全体のパイプラインを構築します。Accelerator の初期化と利用を正しく組み込みます。
8.  **テストとデバッグ**: 小規模なデータセットや短いエポック数で動作確認を行い、各コンポーネントが期待通りに機能するかを検証します。特に Accelerator を使用した分散学習環境での動作確認が重要です。
