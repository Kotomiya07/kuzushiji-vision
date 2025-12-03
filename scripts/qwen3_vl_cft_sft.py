# unslothは必ず一番初めにインポートする
from unsloth import FastVisionModel, UnslothTrainer, UnslothTrainingArguments  # noqa: I001
from unsloth.trainer import UnslothVisionDataCollator  # noqa: I001

import evaluate  # noqa: I001
from trl import SFTConfig, SFTTrainer  # noqa: I001

from datasets import load_dataset  # noqa: I001
import torch  # noqa: I001
import gc  # noqa: I001

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",
    "unsloth/Pixtral-12B-2409-bnb-4bit",
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
]

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)


train_dataset = load_dataset("Kotomiya07/kuzushiji-column", split="train")
# テストデータセットもここで読み込んでおく
# test_dataset = load_dataset("Kotomiya07/kuzushiji-column", split="test")


instruction = "Transcribe the Kuzushiji in the image."


def convert_to_conversation(sample):
    # 'text'フィールドから正解ラベルを抽出
    ground_truth = sample["text"]
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image", "image": sample["image"]}]},
        # 評価時にはアシスタントの応答は不要なため、空にするか、あるいは前処理で分離します
        {"role": "assistant", "content": [{"type": "text", "text": ground_truth}]},
    ]
    # compute_metricsで使いやすいように正解ラベルを別途保持
    return {"messages": conversation, "ground_truth": ground_truth}


# 学習用・テスト用を対話形式に変換
converted_train_dataset = [convert_to_conversation(sample) for sample in train_dataset]
# converted_test_dataset = [convert_to_conversation(sample) for sample in test_dataset]


############################ 追加部分 ############################
#
# CERを計算するための `compute_metrics` 関数を定義します。
# この関数は trainer.evaluate() が呼び出されるたびに実行されます。
#
# cer_metric = evaluate.load("cer")


# def compute_metrics(eval_preds, eval_dataset=None):
#     """
#     評価時にCERを計算する関数
#     """
#     preds, labels = eval_preds

#     # preds: モデルが生成したトークンID
#     # labels: データローダーが提供するラベル (今回は使用しない)

#     # 生成されたトークンIDをテキストにデコード
#     # パディングトークンはデコード時にスキップする
#     pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)

#     # 評価されたサンプルの正解ラベルを取得
#     # eval_datasetが渡されている場合はそれを使用、そうでなければconverted_test_datasetを使用
#     if eval_dataset is not None:
#         label_str = [ex["ground_truth"] for ex in eval_dataset]
#     else:
#         # フォールバック: 評価されたサンプル数分だけ取得
#         label_str = [ex["ground_truth"] for ex in converted_test_dataset[: len(pred_str)]]

#     # CERを計算
#     cer = cer_metric.compute(predictions=pred_str, references=label_str)

#     # 結果を辞書形式で返す
#     return {"cer": cer}


# def _create_chunk_compute_metrics(chunk_dataset, chunk_predictions, chunk_references):
#     """
#     チャンク用のcompute_metrics関数を作成するヘルパー関数
#     ループ変数のバインド問題を回避するため、関数を外側で定義
#     """

#     def compute_metrics_chunk(eval_preds):
#         preds, labels = eval_preds
#         pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
#         label_str = [ex["ground_truth"] for ex in chunk_dataset]
#         # 予測と参照を保存
#         chunk_predictions.extend(pred_str)
#         chunk_references.extend(label_str)
#         # チャンク単位のCERも計算（参考用）
#         cer = cer_metric.compute(predictions=pred_str, references=label_str)
#         return {"cer": cer}

#     return compute_metrics_chunk


# def evaluate_in_chunks(trainer, eval_dataset, chunk_size=10, model_name="model"):
#     """
#     評価データセットをチャンクに分割して評価し、結果を集約する関数
#     メモリ不足を防ぐために使用
#     """
#     print(f"\n{model_name}の評価を開始します（チャンクサイズ: {chunk_size}）...")

#     all_predictions = []
#     all_references = []

#     # データセットをチャンクに分割
#     num_chunks = (len(eval_dataset) + chunk_size - 1) // chunk_size

#     # 評価前の設定を保存
#     original_eval_dataset = trainer.eval_dataset
#     original_compute_metrics = trainer.compute_metrics
#     original_per_device_eval_batch_size = trainer.args.per_device_eval_batch_size
#     original_gradient_accumulation_steps = trainer.args.gradient_accumulation_steps

#     # 評価用の設定を適用
#     trainer.args.per_device_eval_batch_size = 1
#     trainer.args.gradient_accumulation_steps = 1

#     try:
#         for chunk_idx in range(num_chunks):
#             start_idx = chunk_idx * chunk_size
#             end_idx = min(start_idx + chunk_size, len(eval_dataset))
#             chunk_dataset = eval_dataset[start_idx:end_idx]

#             print(f"  チャンク {chunk_idx + 1}/{num_chunks} を評価中... (サンプル {start_idx}-{end_idx - 1})")

#             # チャンク用の予測と参照を保存するリスト
#             chunk_predictions = []
#             chunk_references = []

#             # チャンク用のcompute_metrics関数を作成
#             compute_metrics_chunk = _create_chunk_compute_metrics(chunk_dataset, chunk_predictions, chunk_references)

#             # 一時的にeval_datasetとcompute_metricsを変更
#             trainer.eval_dataset = chunk_dataset
#             trainer.compute_metrics = compute_metrics_chunk

#             # チャンクを評価
#             trainer.evaluate()

#             # 収集した予測と参照を全体のリストに追加
#             all_predictions.extend(chunk_predictions)
#             all_references.extend(chunk_references)

#             # GPUメモリをクリア
#             torch.cuda.empty_cache()
#             gc.collect()

#         # 全チャンクの結果を集約して最終的なCERを計算
#         final_cer = cer_metric.compute(predictions=all_predictions, references=all_references)
#         final_results = {"cer": final_cer}

#         print(f"{model_name}の評価が完了しました。")
#         print(f"  総サンプル数: {len(all_predictions)}")
#         print(f"  最終CER: {final_cer:.4f}")

#         return final_results

#     finally:
#         # 元の設定に戻す
#         trainer.eval_dataset = original_eval_dataset
#         trainer.compute_metrics = original_compute_metrics
#         trainer.args.per_device_eval_batch_size = original_per_device_eval_batch_size
#         trainer.args.gradient_accumulation_steps = original_gradient_accumulation_steps


#
##################################################################


model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

### CPT (Continual Pre-training)

FastVisionModel.for_training(model)


def is_bf16_supported():
    try:
        import torch

        return torch.cuda.is_bf16_supported()
    except ImportError:
        return False


processor = model.processor if hasattr(model, "processor") else tokenizer

data_collator = UnslothVisionDataCollator(
    model=model,
    processor=processor,
    max_seq_length=4096,
)


trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=converted_train_dataset,
    ############################ 追加部分 ############################
    # eval_dataset=converted_test_dataset,  # 評価データセットをTrainerに渡す
    # compute_metrics=compute_metrics,  # 作成した計算関数を渡す
    ##################################################################
    args=UnslothTrainingArguments(
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=5e-5,
        embedding_learning_rate=1e-5,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs/outputs_cpt",
        report_to="wandb",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=4096,
    ),
)

############################ 追加部分 ############################
#
# UnslothTrainingArgumentsはpredict_with_generateを直接受け取らないため、
# Trainerを初期化した後に、そのargs属性に直接設定します。
#
# trainer.args.predict_with_generate = True
# trainer.args.generation_max_length = 256  # 生成するテキストの最大長も同様に設定
#
##################################################################

trainer_stats = trainer.train()

# テストデータセットで モデルを評価
############################ 修正部分 ############################
#
# メモリ不足を防ぐため、評価データセットをチャンクに分割して評価します
# GPUメモリをクリアしてから評価を開始
# torch.cuda.empty_cache()
# gc.collect()

# チャンクサイズは環境に応じて調整可能（デフォルト: 10）
# メモリが少ない場合は5や3に減らしてください
# test_results = evaluate_in_chunks(trainer=trainer, eval_dataset=converted_test_dataset, chunk_size=10, model_name="CPTモデル")
# print("CPTモデルの評価結果:")
# print(test_results)
#
##################################################################

# 最も性能の良かったモデルがロードされているので、それを保存します
model.save_pretrained("outputs/lora_model_cpt_best")
tokenizer.save_pretrained("outputs/lora_model_cpt_best")


### SFT (Supervised Fine-Tuning)
# CPTで学習した最良のモデルをロードしてSFTに進むことも可能です

model, tokenizer = FastVisionModel.from_pretrained(
    "outputs/lora_model_cpt_best",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

FastVisionModel.for_training(model)

# ステップ3: Trainer (SFT) の設定を変更
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=converted_train_dataset,
    # eval_dataset=converted_test_dataset,
    # compute_metrics=compute_metrics,
    args=SFTConfig(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs/outputs_sft",
        report_to="wandb",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_length=4096,
        # SFTConfigは 'predict_with_generate' を受け取らないため、ここから削除します
    ),
)


############################ 修正・追加部分 ############################
#
# SFTConfigはpredict_with_generateを直接受け取らないため、
# Trainerを初期化した後に、そのargs属性に直接設定します。
#
# trainer.args.predict_with_generate = True
# trainer.args.generation_max_length = 256  # 生成するテキストの最大長も同様に設定
#
##################################################################


trainer_stats = trainer.train()

# 最も性能の良かったモデルがロードされているので、それを保存します
model.save_pretrained("outputs/lora_model_cpt_sft_best")
tokenizer.save_pretrained("outputs/lora_model_cpt_sft_best")
model.save_pretrained_merged(
    "outputs/lora_model_cpt_sft_best_vllm",
    tokenizer,
)

# テストデータセットで モデルを評価
############################ 修正部分 ############################
#
# メモリ不足を防ぐため、評価データセットをチャンクに分割して評価します
# GPUメモリをクリアしてから評価を開始
# torch.cuda.empty_cache()
# gc.collect()

# チャンクサイズは環境に応じて調整可能（デフォルト: 10）
# メモリが少ない場合は5や3に減らしてください
# test_results = evaluate_in_chunks(trainer=trainer, eval_dataset=converted_test_dataset, chunk_size=10, model_name="SFTモデル")
# print("SFTモデルの評価結果:")
# print(test_results)
#
##################################################################
