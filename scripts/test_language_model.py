"""
翻刻テキストを使用して、モデルに文字と文字のつながりを学習させるスクリプト
"""

import argparse
import glob
import os

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
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

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
    マスクされたトークンをモデルのTop1、Top3、Top5予測で復元し、元の文、マスクされた文、復元された文を返す

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
            - restored_text_top1: Top1で復元された文
            - restored_text_top3: Top3候補を含む復元された文
            - restored_text_top5: Top5候補を含む復元された文
            - top3_tokens: マスク位置のTop3候補トークン
            - top5_tokens: マスク位置のTop5候補トークン
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
        logits = outputs.logits
        
        # Top1予測
        predictions_top1 = logits.argmax(dim=-1)
        
        # Top3予測
        top3_predictions = torch.topk(logits, k=3, dim=-1)  # values, indices
        
        # Top5予測
        top5_predictions = torch.topk(logits, k=5, dim=-1)  # values, indices

        for i in range(batch_size):
            # 元の文を復元（labelsから-100でない部分を取得）
            original_tokens = input_ids[i].clone()
            mask_positions = labels[i] != -100
            original_tokens[mask_positions] = labels[i][mask_positions]

            # マスクされた文
            masked_tokens = input_ids[i].clone()

            # Top1で復元された文（マスク位置のみ予測で置換）
            restored_tokens_top1 = input_ids[i].clone()
            restored_tokens_top1[mask_positions] = predictions_top1[i][mask_positions]
            
            # マスク位置のTop3、Top5候補を取得
            mask_positions_list = torch.where(mask_positions)[0]
            top3_tokens_for_masks = []
            top5_tokens_for_masks = []
            
            for pos in mask_positions_list:
                # Top3候補
                top3_indices = top3_predictions.indices[i][pos]
                top3_tokens = [tokenizer.decode([idx], skip_special_tokens=True) for idx in top3_indices]
                top3_tokens_for_masks.append(top3_tokens)
                
                # Top5候補
                top5_indices = top5_predictions.indices[i][pos]
                top5_tokens = [tokenizer.decode([idx], skip_special_tokens=True) for idx in top5_indices]
                top5_tokens_for_masks.append(top5_tokens)

            # テキストに変換
            original_text = tokenizer.decode(original_tokens, skip_special_tokens=True)
            masked_text = tokenizer.decode(masked_tokens, skip_special_tokens=False)
            # PADトークンを削除
            masked_text = masked_text.replace(tokenizer.pad_token, "")
            masked_text = masked_text.replace(tokenizer.cls_token, "")
            masked_text = masked_text.replace(tokenizer.sep_token, "")
            masked_text = masked_text.replace(tokenizer.mask_token, "　")
            
            restored_text_top1 = tokenizer.decode(restored_tokens_top1, skip_special_tokens=True)
            
            # Top3候補を含む復元文の作成
            restored_text_top3 = restored_text_top1
            if top3_tokens_for_masks:
                restored_text_top3 += " [Top3: " + ", ".join([f"{'|'.join(tokens)}" for tokens in top3_tokens_for_masks]) + "]"
            
            # Top5候補を含む復元文の作成
            restored_text_top5 = restored_text_top1
            if top5_tokens_for_masks:
                restored_text_top5 += " [Top5: " + ", ".join([f"{'|'.join(tokens)}" for tokens in top5_tokens_for_masks]) + "]"

            results.append(
                {
                    "original_text": original_text,
                    "masked_text": masked_text,
                    "restored_text_top1": restored_text_top1,
                    "restored_text_top3": restored_text_top3,
                    "restored_text_top5": restored_text_top5,
                    "top3_tokens": top3_tokens_for_masks,
                    "top5_tokens": top5_tokens_for_masks,
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
    print("マスクされたトークンの復元結果 (Top1 & Top3 & Top5予測)")
    print("=" * 80)

    for i, result in enumerate(restoration_results[:max_display]):
        print(f"\n例 {i + 1} (マスクされたトークン数: {result['mask_count']})")
        print("-" * 60)
        print(f"元の文:        {result['original_text']}")
        print(f"マスク文:      {result['masked_text']}")
        print(f"復元文(Top1):  {result['restored_text_top1']}")
        
        # Top3候補を表示
        if result.get('top3_tokens') and len(result['top3_tokens']) > 0:
            print(f"Top3候補:      {result['restored_text_top3']}")
        
        # Top5候補を表示（長くなりすぎないように制御）
        if result.get('top5_tokens') and len(result['top5_tokens']) > 0:
            print(f"Top5候補:      {result['restored_text_top5']}")
        
        # 復元の正確性を簡単にチェック
        if result["original_text"] == result["restored_text_top1"]:
            print("✓ Top1完全復元成功")
        else:
            # Top3とTop5に正解が含まれているかチェック
            top3_correct = False
            top5_correct = False
            
            if result.get('top3_tokens'):
                # 各マスク位置で正解がTop3に含まれているかチェック
                original_tokens = result["original_text"]
                top3_tokens_flat = [token for mask_tokens in result['top3_tokens'] for token in mask_tokens]
                # 簡易的なチェック（完全ではないが参考になる）
                top3_correct = any(token in original_tokens for token in top3_tokens_flat)
            
            if result.get('top5_tokens'):
                # 各マスク位置で正解がTop5に含まれているかチェック
                original_tokens = result["original_text"]
                top5_tokens_flat = [token for mask_tokens in result['top5_tokens'] for token in mask_tokens]
                # 簡易的なチェック（完全ではないが参考になる）
                top5_correct = any(token in original_tokens for token in top5_tokens_flat)
            
            if top5_correct:
                print("△ Top1では不完全だが、Top5に正解候補あり")
            elif top3_correct:
                print("△ Top1では不完全だが、Top3に正解候補あり")
            else:
                print("× Top5でも不完全復元")

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
            print(f"Evaluation ({metric_key_prefix}): Using a subset of {len(actual_eval_dataset)} samples")
            all_results = []
            for i in range(int(len(actual_eval_dataset) / num_eval_samples)):
                print(f"Eval: {i+1} / {int(len(actual_eval_dataset) / num_eval_samples)}")
                # i*num_eval_samples から (i+1)*num_eval_samples までのインデックスを取得
                indices = list(range(i * num_eval_samples, (i + 1) * num_eval_samples))
                sampled_dataset = actual_eval_dataset.select(indices)

                # evaluate メソッドが eval_dataset=None で呼ばれた場合 (self.eval_dataset を使うケース)
                # self.eval_dataset を一時的に差し替える必要がある
                if eval_dataset is None:
                    original_self_eval_dataset = self.eval_dataset
                    self.eval_dataset = sampled_dataset
                    try:
                        results = super().evaluate(eval_dataset=None, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
                    finally:
                        self.eval_dataset = original_self_eval_dataset  # 必ず元に戻す
                    all_results.append(results)
                else:
                    # evaluate メソッドが eval_dataset=some_dataset で呼ばれた場合
                    results = super().evaluate(
                        eval_dataset=sampled_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
                    )
                    all_results.append(results)
            # 平均を計算
            return {k: np.mean([r[k] for r in all_results]) for k in all_results[0].keys()}

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
            "ndl-minhon-ocrdataset/src/honkoku_oneline_v1",
            "ndl-minhon-ocrdataset/src/honkoku_oneline_v2",
            "honkoku_yatanavi/honkoku_oneline",
            "data/oneline",
            "kokubunken_repo/text/azumakagami",
            "kokubunken_repo/text/eirigenji",
            "kokubunken_repo/text/nijuuichidaishuu",
            "kokubunken_repo/text/rekishimonogo",
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
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size for evaluation.")
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

    # 1. Tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
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
    model.requires_grad_(False)
    model.eval()

    # 4. Data Collator
    # Data collator for MLM. It will handle dynamic masking.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mask_probability)
    print(f"Using DataCollatorForLanguageModeling with mask probability: {args.mask_probability}")

    # 5. Training Arguments
    training_args = TrainingArguments(
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
        tf32=True,
        bf16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        resume_from_checkpoint=args.resume_from_checkpoint,
        optim="schedule_free_adamw",  # Schedule-Free Optimizer の指定
        lr_scheduler_type="constant",
        report_to="none",
        remove_unused_columns=False,
    )

    # 6. Evaluation Metrics
    def compute_metrics(p):
        predictions, labels = p
        # predictions are logits, so we need to get top-k predictions
        # For MLM, labels contain -100 for non-masked tokens
        masked_indices = labels != -100

        # Flatten and filter predictions and labels
        flat_predictions_logits = predictions[masked_indices]  # Shape: (num_masked_tokens, vocab_size)
        flat_labels = labels[masked_indices].flatten()

        if len(flat_labels) == 0:  # Should not happen if there are masked tokens
            return {
                "accuracy": 0.0, 
                "accuracy_top3": 0.0,
                "accuracy_top5": 0.0,
                "precision_macro": 0.0, 
                "recall_macro": 0.0, 
                "f1_macro": 0.0
            }

        # Top1 predictions
        flat_predictions_top1 = flat_predictions_logits.argmax(axis=-1).flatten()
        
        # Top3 predictions
        top3_predictions = np.argsort(flat_predictions_logits, axis=-1)[:, -3:]  # Get top 3 indices
        
        # Top5 predictions
        top5_predictions = np.argsort(flat_predictions_logits, axis=-1)[:, -5:]  # Get top 5 indices
        
        # Calculate Top1 accuracy
        accuracy_top1 = accuracy_score(flat_labels, flat_predictions_top1)
        
        # Calculate Top3 accuracy
        correct_top3 = 0
        for i, true_label in enumerate(flat_labels):
            if true_label in top3_predictions[i]:
                correct_top3 += 1
        accuracy_top3 = correct_top3 / len(flat_labels)
        
        # Calculate Top5 accuracy
        correct_top5 = 0
        for i, true_label in enumerate(flat_labels):
            if true_label in top5_predictions[i]:
                correct_top5 += 1
        accuracy_top5 = correct_top5 / len(flat_labels)
        
        # Calculate precision, recall, f1 for Top1 predictions
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_labels, flat_predictions_top1, average="macro", zero_division=0
        )

        return {
            "accuracy": accuracy_top1,
            "accuracy_top3": accuracy_top3,
            "accuracy_top5": accuracy_top5,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    print("Compute_metrics function defined.")

    # 7. Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )
    
    # 8. Evaluate (if eval_dataset exists)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    if eval_dataset:
        print("Starting final evaluation...")
        eval_results = trainer.evaluate()
        print("Evaluation results:")
        for key, value in eval_results.items():
            if key in ['eval_accuracy', 'eval_accuracy_top3', 'eval_accuracy_top5', 'eval_f1', 'eval_precision', 'eval_recall']:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        end_event.record()
        torch.cuda.synchronize()
        print(f"\nEvaluation time: {start_event.elapsed_time(end_event) / 1000} s")
        # Top1、Top3、Top5精度の比較を分かりやすく表示
        if 'eval_accuracy' in eval_results and 'eval_accuracy_top3' in eval_results and 'eval_accuracy_top5' in eval_results:
            top1_acc = eval_results['eval_accuracy']
            top3_acc = eval_results['eval_accuracy_top3']
            top5_acc = eval_results['eval_accuracy_top5']
            improvement_top3 = top3_acc - top1_acc
            improvement_top5 = top5_acc - top1_acc
            print("\n精度比較:")
            print(f"  Top1精度: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
            print(f"  Top3精度: {top3_acc:.4f} ({top3_acc*100:.2f}%) [+{improvement_top3:.4f} ({improvement_top3*100:.2f}ポイント)]")
            print(f"  Top5精度: {top5_acc:.4f} ({top5_acc*100:.2f}%) [+{improvement_top5:.4f} ({improvement_top5*100:.2f}ポイント)]")

        # 最終評価後に復元例を表示
        print("\n最終評価後の復元例表示:")
        # trainer.show_restoration_examples(eval_dataset=eval_dataset, num_examples=10)
    else:
        print("No evaluation dataset provided. Skipping final evaluation.")


if __name__ == "__main__":
    main()

# How to run
"""
python scripts/test_language_model.py \
--model_name experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250530_034458/checkpoint-680000
"""

"""
Found 66537 text files in ndl-minhon-ocrdataset/src/honkoku_oneline_v1
Found 360052 text files in ndl-minhon-ocrdataset/src/honkoku_oneline_v2
Found 84990 text files in honkoku_yatanavi/honkoku_oneline
Found 29603 text files in data/oneline
"""


"""
experiments/model-ver2/decoder-roberta-v3
Evaluation results:
  eval_loss: 4.457808494567871
  eval_model_preparation_time: 0.3452
  eval_accuracy: 0.22835943940643033
  eval_precision: 0.13860941453432776
  eval_recall: 0.11290581894984325
  eval_f1: 0.10923065383140683
  eval_runtime: 219.7951
  eval_samples_per_second: 4.55
  eval_steps_per_second: 0.287

Evaluation results:
  eval_loss: 4.466537952423096
  eval_model_preparation_time: 0.4389
  eval_accuracy: 0.2088635366388767
  eval_precision: 0.151376414050057
  eval_recall: 0.10999374736408715
  eval_f1: 0.11179419012560109
  eval_runtime: 259.9218
  eval_samples_per_second: 3.847
  eval_steps_per_second: 0.062
"""

"""
20250526_141154/final_model
Evaluation results:
  eval_loss: nan
  eval_model_preparation_time: 0.3338
  eval_accuracy: 0.36601578728707934
  eval_precision: 0.1405605604807511
  eval_recall: 0.13742032964513676
  eval_f1: 0.12865473236951522
  eval_runtime: 225.9711
  eval_samples_per_second: 4.425
  eval_steps_per_second: 2.213
"""

"""
KoichiYasuoka/roberta-small-japanese-aozora-char
Evaluation results:
  eval_loss: nan
  eval_model_preparation_time: 0.3458
  eval_accuracy: 0.27643630308076605
  eval_precision: 0.15856445045024362
  eval_recall: 0.13719538320773383
  eval_f1: 0.13476485081209721
  eval_runtime: 285.2448
  eval_samples_per_second: 3.506
  eval_steps_per_second: 1.753
"""
