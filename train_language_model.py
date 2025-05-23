"""
翻刻テキストを使用して、モデルに文字と文字のつながりを学習させるスクリプト
"""

import argparse
import glob
import os
from datetime import datetime

# typingからList, Optionalをインポート
import numpy as np
import torch
from datasets import Dataset
from numpy import dtype, ndarray

# _reconstruct を直接インポート
from numpy.core.multiarray import _reconstruct
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# 安全なグローバルとして登録
torch.serialization.add_safe_globals([_reconstruct])
torch.serialization.add_safe_globals([ndarray])
torch.serialization.add_safe_globals([dtype])

# NumPy のデータ型クラスをインポート (正確なパスは NumPy のバージョンによる可能性があります)
# 多くの場合、np.dtype('uint32') の型そのものを取得するか、
# 型クラスを直接指定します。
# まず UInt32DType を試す
try:
    from numpy import UInt32DType

    dtype_to_add = UInt32DType
except ImportError:
    # 古い NumPy や別のアクセス方法の場合
    try:
        from numpy.dtypes import UInt32DType as NumpyUInt32DType  # エイリアスを使う

        dtype_to_add = NumpyUInt32DType
    except ImportError:
        # dtypeオブジェクト自体を登録することも試せる
        dtype_to_add = np.dtype("uint32").type
        print(f"Warning: Using np.dtype('uint32').type ({dtype_to_add}) for safe globals.")

# 安全なグローバルとして登録
if dtype_to_add:
    torch.serialization.add_safe_globals([dtype_to_add])
else:
    print("Error: Could not find UInt32DType to add to safe globals.")


def restore_masked_text(model, tokenizer, input_ids, labels, max_examples=10):
    """
    マスクされたトークンをモデルのTop1予測で復元し、元の文、マスクされた文、復元された文を返す

    Args:
        model: 学習済みのAutoModelForMaskedLM
        tokenizer: 対応するトークナイザー
        input_ids: マスクされたinput_ids [batch_size, seq_len]
        labels: 元のラベル（-100でマスクされていない部分を示す） [batch_size, seq_len]
        max_examples: 表示する例の最大数

    Returns:
        list: 復元結果の辞書のリスト
            - original_text: 元の文
            - masked_text: マスクされた文
            - restored_text: 復元された文
    """
    model.eval()
    device = next(model.parameters()).device

    # バッチサイズを制限
    batch_size = min(input_ids.size(0), max_examples)
    input_ids = input_ids[:batch_size].to(device)
    labels = labels[:batch_size].to(device)

    results = []

    with torch.no_grad():
        # モデルで予測
        outputs = model(input_ids=input_ids)
        predictions = outputs.logits.argmax(dim=-1)  # Top1予測

        for i in range(batch_size):
            # 元の文を復元（labelsから-100でない部分を取得）
            original_tokens = input_ids[i].clone()
            mask_positions = labels[i] != -100
            original_tokens[mask_positions] = labels[i][mask_positions]

            # マスクされた文
            masked_tokens = input_ids[i].clone()

            # 復元された文（マスク位置のみ予測で置換）
            restored_tokens = input_ids[i].clone()
            restored_tokens[mask_positions] = predictions[i][mask_positions]

            # テキストに変換
            original_text = tokenizer.decode(original_tokens, skip_special_tokens=True)
            masked_text = tokenizer.decode(masked_tokens, skip_special_tokens=False)
            # PADトークンを削除
            masked_text = masked_text.replace(tokenizer.pad_token, "")
            restored_text = tokenizer.decode(restored_tokens, skip_special_tokens=True)

            results.append(
                {
                    "original_text": original_text,
                    "masked_text": masked_text,
                    "restored_text": restored_text,
                    "mask_count": mask_positions.sum().item(),
                }
            )

    return results


def print_restoration_examples(restoration_results, max_display=5):
    """
    復元結果を見やすい形式で表示する

    Args:
        restoration_results: restore_masked_text関数の戻り値
        max_display: 表示する例の最大数
    """
    print("\n" + "=" * 80)
    print("マスクされたトークンの復元結果 (Top1予測)")
    print("=" * 80)

    for i, result in enumerate(restoration_results[:max_display]):
        print(f"\n例 {i + 1} (マスクされたトークン数: {result['mask_count']})")
        print("-" * 60)
        print(f"元の文:     {result['original_text']}")
        print(f"マスク文:   {result['masked_text']}")
        print(f"復元文:     {result['restored_text']}")

        # 復元の正確性を簡単にチェック
        if result["original_text"] == result["restored_text"]:
            print("✓ 完全復元成功")
        else:
            print("△ 部分的復元")

    print("\n" + "=" * 80)


# CustomTrainer Class Definition
class CustomTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ):
        num_eval_samples: int = 1000  # サンプリングする件数

        # trainer.evaluate() が引数なしで呼ばれた場合は self.eval_dataset を使う
        # trainer.evaluate(my_eval_dataset) のように呼ばれた場合は my_eval_dataset を使う
        actual_eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if actual_eval_dataset is None:
            # 評価データセットがない場合は、そのまま親クラスのメソッドを呼び出す
            return super().evaluate(eval_dataset=None, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        # 通常の評価時（metric_key_prefix が 'eval'）かつ、
        # データセットのサイズが指定したサンプリング数より大きい場合にサンプリングを実行
        if metric_key_prefix == "eval" and len(actual_eval_dataset) > num_eval_samples:
            indices = np.random.choice(len(actual_eval_dataset), num_eval_samples, replace=False)
            sampled_dataset = actual_eval_dataset.select(indices)
            print(
                f"Evaluation ({metric_key_prefix}): Using a random subset of {len(sampled_dataset)} samples "
                f"({num_eval_samples} requested) from the evaluation dataset."
            )

            # evaluate メソッドが eval_dataset=None で呼ばれた場合 (self.eval_dataset を使うケース)
            # self.eval_dataset を一時的に差し替える必要がある
            if eval_dataset is None:
                original_self_eval_dataset = self.eval_dataset
                self.eval_dataset = sampled_dataset
                try:
                    results = super().evaluate(eval_dataset=None, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
                finally:
                    self.eval_dataset = original_self_eval_dataset  # 必ず元に戻す
                return results
            else:
                # evaluate メソッドが eval_dataset=some_dataset で呼ばれた場合
                return super().evaluate(
                    eval_dataset=sampled_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
                )
        else:
            # サンプリングしない場合 (データセットが小さい、または metric_key_prefix が 'eval' でないなど)
            if metric_key_prefix == "eval" and actual_eval_dataset:
                print(
                    f"Evaluation ({metric_key_prefix}): Using the full evaluation dataset ({len(actual_eval_dataset)} samples)."
                )
            return super().evaluate(
                eval_dataset=actual_eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
            )

    def show_restoration_examples(self, eval_dataset=None, num_examples=5):
        """
        評価データセットからサンプルを選択してマスクされたトークンの復元例を表示する

        Args:
            eval_dataset: 評価データセット（Noneの場合はself.eval_datasetを使用）
            num_examples: 表示する例の数
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if eval_dataset is None:
            print("評価データセットが利用できません。復元例の表示をスキップします。")
            return

        # 少数のサンプルを選択
        sample_size = min(num_examples, len(eval_dataset))
        indices = np.random.choice(len(eval_dataset), sample_size, replace=False)
        sample_dataset = eval_dataset.select(indices)

        # データコレーターを使用してマスクされたデータを生成
        sample_batch = self.data_collator([sample_dataset[i] for i in range(sample_size)])

        # デバイスに移動
        device = next(self.model.parameters()).device
        input_ids = sample_batch["input_ids"].to(device)
        labels = sample_batch["labels"].to(device)

        # 復元を実行
        restoration_results = restore_masked_text(self.model, self.tokenizer, input_ids, labels, max_examples=num_examples)

        # 結果を表示
        print_restoration_examples(restoration_results, max_display=num_examples)


# Custom Callback for training data metrics
class TrainMetricsCallback(TrainerCallback):
    def __init__(self, train_dataset_for_metrics, compute_metrics_fn, metric_key_prefix="train", max_samples_for_metrics=None):
        self.train_dataset_for_metrics = train_dataset_for_metrics
        self.compute_metrics_fn = compute_metrics_fn
        self.metric_key_prefix = metric_key_prefix
        self.trainer = None
        self.max_samples_for_metrics = max_samples_for_metrics
        self.eval_count = 0  # 評価回数をカウント

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if self.trainer is None:
            print("Warning: Trainer instance not available in TrainMetricsCallback. Skipping train metrics.")
            return

        if state.is_world_process_zero:  # Only run on the main process
            if self.train_dataset_for_metrics is None:
                print("Train dataset for metrics is not available in callback.")
                return
            if self.compute_metrics_fn is None:
                print("Compute metrics function is not available in callback.")
                return

            print(f"Evaluating on training data (or a subset) at step {state.global_step}...")

            eval_dataset = self.train_dataset_for_metrics
            if self.max_samples_for_metrics is not None and self.max_samples_for_metrics < len(self.train_dataset_for_metrics):
                # Randomly select a subset of the training data for evaluation
                indices = np.random.choice(len(self.train_dataset_for_metrics), self.max_samples_for_metrics, replace=False)
                eval_dataset = self.train_dataset_for_metrics.select(indices)
                print(f"Using a random subset of {len(eval_dataset)} samples from training data for metrics calculation.")
            else:
                print(f"Using the full training dataset ({len(eval_dataset)} samples) for metrics calculation.")

            # trainer.predict を使用して訓練データセット（またはサブセット）で予測を実行
            predictions_output = self.trainer.predict(eval_dataset, metric_key_prefix=self.metric_key_prefix)
            metrics_from_predict = predictions_output.metrics

            custom_metrics_input = (predictions_output.predictions, predictions_output.label_ids)
            custom_metrics = self.compute_metrics_fn(custom_metrics_input)

            log_metrics = {}
            if metrics_from_predict:
                for k, v in metrics_from_predict.items():
                    log_metrics[k] = v

            for k, v in custom_metrics.items():
                log_metrics[f"{self.metric_key_prefix}_{k}"] = v

            self.trainer.log(log_metrics)
            print(f"Train data metrics logged: {log_metrics}")

            # 復元例を定期的に表示（5回に1回）
            self.eval_count += 1
            if self.eval_count % 5 == 1:  # 最初の評価と5回に1回表示
                print(f"\n復元例を表示します（評価回数: {self.eval_count}）...")
                if hasattr(self.trainer, "show_restoration_examples"):
                    # 評価データセットがある場合はそれを使用、なければ訓練データセットを使用
                    display_dataset = self.trainer.eval_dataset if self.trainer.eval_dataset is not None else eval_dataset
                    self.trainer.show_restoration_examples(eval_dataset=display_dataset, num_examples=3)
                else:
                    print("復元例表示機能が利用できません。")


def main():
    parser = argparse.ArgumentParser(description="Pretrain a language model for Kuzushiji recognition.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="KoichiYasuoka/roberta-small-japanese-aozora-char",
        help="Pretrained model name or path.",
    )
    parser.add_argument(
        "--dataset_dirs",
        type=list,
        default=[
            # "ndl-minhon-ocrdataset/src/honkoku_oneline_v1",
            # "ndl-minhon-ocrdataset/src/honkoku_oneline_v2",
            # "honkoku_yatanavi/honkoku_oneline",
            # "data/oneline",
            "kokubunken_repo/text",
        ],
        help="Directory containing the text dataset files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/pretrain_language_model",
        help="Output directory for a_model_name and checkpoints.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--mask_probability", type=float, default=0.15, help="Probability of masking tokens.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Run evaluation every X updates steps.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log metrics every X updates steps.")
    parser.add_argument("--early_stopping_patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--vocab_size", type=int, default=32000, help="学習時の語彙サイズ (パスの推測用)")
    parser.add_argument(
        "--model_type", type=str, default="bpe", choices=["bpe", "unigram"], help="学習時のモデルタイプ (パスの推測用)"
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--auto_find_batch_size", action="store_true", help="Auto find batch size.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint.")

    args = parser.parse_args()

    # 1. Vocab and Tokenizer
    DEFAULT_VOCAB_SIZE = args.vocab_size
    DEFAULT_MODEL_TYPE = args.model_type
    BASE_EXPERIMENT_DIR = "experiments/kuzushiji_tokenizer_one_char"
    MODEL_SPECIFIC_DIR_NAME = f"vocab{DEFAULT_VOCAB_SIZE}_{DEFAULT_MODEL_TYPE}"
    TOKENIZER_FILE_PATH = os.path.join(BASE_EXPERIMENT_DIR, MODEL_SPECIFIC_DIR_NAME)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tokenizer = AutoTokenizer.from_pretrained(BASE_EXPERIMENT_DIR)
    # 2. Load and Preprocess Dataset
    print(f"Loading dataset from {args.dataset_dirs}")
    text_files = []
    # dataset_dirsの各ディレクトリ以下を再帰的に検索
    for dataset_dir in args.dataset_dirs:
        text_files_iterator = glob.iglob(os.path.join(dataset_dir, "**/*.txt"), recursive=True)
        count = 0
        for text_file in text_files_iterator:
            text_files.append(text_file)
            count += 1
        print(f"Found {count} text files in {dataset_dir}")

    texts = []
    for file_path in text_files:
        with open(file_path, encoding="utf-8") as f:
            texts.append(f.readline().strip())

    # Create Hugging Face Dataset
    dataset_dict = {"text": texts}
    dataset = Dataset.from_dict(dataset_dict)

    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize the texts. The tokenizer will automatically add CLS and SEP if configured.
        return tokenizer(examples["text"], truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print(f"Dataset tokenized. Number of examples: {len(tokenized_dataset)}")

    # Split dataset into train and test
    if args.test_size > 0:
        train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=args.test_size).values()
        print(f"Dataset split into train ({len(train_dataset)}) and eval ({len(eval_dataset)}) sets.")
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None  # Or a small subset if evaluation is still desired during training
        print(f"Using full dataset for training ({len(train_dataset)}). No evaluation set created.")

    # 3. Model
    print(f"Loading model: {args.model_name}")
    # model = AutoModelForMaskedLM.from_pretrained(args.model_name, attn_implementation="flash_attention_2")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    # Resize token embeddings if we used a custom tokenizer or added tokens to AutoTokenizer
    # This is crucial if the vocab size of the tokenizer is different from the model's original vocab size
    model.resize_token_embeddings(len(tokenizer))

    # 4. Data Collator
    # Data collator for MLM. It will handle dynamic masking.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mask_probability)
    print(f"Using DataCollatorForLanguageModeling with mask probability: {args.mask_probability}")

    if args.resume_from_checkpoint is None:
        output_dir = os.path.join(args.output_dir, args.model_name.split("/")[-1], datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)
        run_name = output_dir.split("/")[-1]
    else:
        output_dir = args.resume_from_checkpoint
        run_name = args.resume_from_checkpoint.split("/")[-1] + "_resume"
    # 5. Training Arguments
    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=output_dir,
        overwrite_output_dir=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=os.cpu_count(),
        torch_compile=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size if not args.auto_find_batch_size else 512,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to="wandb",
        tf32=True,
        bf16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        resume_from_checkpoint=args.resume_from_checkpoint,
        optim="schedule_free_adamw",  # Schedule-Free Optimizer の指定
        lr_scheduler_type="constant",
    )
    if args.gradient_accumulation_steps:
        training_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    print("TrainingArguments configured.")

    if args.resume_from_checkpoint is None:
        training_args.learning_rate = args.learning_rate

    # 6. Evaluation Metrics
    def compute_metrics(p):
        predictions, labels = p
        # predictions are logits, so we need to take argmax
        # For MLM, labels contain -100 for non-masked tokens
        masked_indices = labels != -100

        # Flatten and filter predictions and labels
        flat_predictions = predictions.argmax(axis=-1)[masked_indices].flatten()
        flat_labels = labels[masked_indices].flatten()

        if len(flat_labels) == 0:  # Should not happen if there are masked tokens
            return {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}

        accuracy = accuracy_score(flat_labels, flat_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_labels, flat_predictions, average="macro", zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    print("Compute_metrics function defined.")

    # EarlyStoppingCallbackの設定
    callbacks_list = []
    if args.early_stopping_patience is not None:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
        )
        callbacks_list.append(early_stopping_callback)

    # 7. Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        callbacks=callbacks_list,
    )
    print("Trainer initialized.")

    # Add custom callback for training metrics
    if trainer.train_dataset is not None and (compute_metrics if eval_dataset is not None else None) is not None:
        train_metrics_callback = TrainMetricsCallback(
            train_dataset_for_metrics=trainer.train_dataset, compute_metrics_fn=compute_metrics, max_samples_for_metrics=2000
        )
        train_metrics_callback.trainer = trainer
        trainer.add_callback(train_metrics_callback)
        print("TrainMetricsCallback registered.")
    else:
        if trainer.train_dataset is None:
            print("TrainMetricsCallback not registered because train_dataset is None.")
        if (compute_metrics if eval_dataset is not None else None) is None:
            print(
                "TrainMetricsCallback not registered because compute_metrics is None (likely no eval_dataset or eval_strategy='no')."
            )

    # 8. Train
    print("Starting training...")
    print(f"output_dir: {training_args.output_dir}")
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        print(f"最新のチェックポイント: {last_checkpoint} から再開します。")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("チェックポイントが見つかりませんでした。最初から学習します。")
        trainer.train()

    print("Training finished.")

    # 9. Save model and tokenizer
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final model and tokenizer saved to {final_model_path}")

    # 10. Evaluate (if eval_dataset exists)
    if eval_dataset:
        print("Starting final evaluation...")
        eval_results = trainer.evaluate()
        print("Evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")

        # 最終評価後に復元例を表示
        print("\n最終評価後の復元例表示:")
        trainer.show_restoration_examples(eval_dataset=eval_dataset, num_examples=5)
    else:
        print("No evaluation dataset provided. Skipping final evaluation.")


if __name__ == "__main__":
    main()

# How to run
"""
python train_language_model.py \
    --num_train_epochs 10000 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 2 \
    --learning_rate 0.0001 \
    --mask_probability 0.15 \
    --test_size 0.1 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --logging_steps 100 \
    --resume_from_checkpoint experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250511_192051
"""

"""
Found 66537 text files in ndl-minhon-ocrdataset/src/honkoku_oneline_v1
Found 360052 text files in ndl-minhon-ocrdataset/src/honkoku_oneline_v2
Found 84990 text files in honkoku_yatanavi/honkoku_oneline
Found 29603 text files in data/oneline
"""
