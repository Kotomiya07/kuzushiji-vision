"""
言語モデルのAccuracyとCER（Character Error Rate）をテストするスクリプト

Usage:
    uv run python test_language_model.py \
        --checkpoint_path experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250526_141154/checkpoint-1200000 \
        --tokenizer_name experiments/kuzushiji_tokenizer_one_char \
        --dataset_name Kotomiya07/honkoku-hq \
        --test_size 0.2 \
        --mask_probability 0.15
"""

import argparse
import os

import numpy as np
import torch
from jiwer import cer
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

from datasets import load_dataset


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate (CER) を計算する（jiwerライブラリ使用）

    Args:
        reference: 正解文字列
        hypothesis: 予測文字列

    Returns:
        CER値（0.0〜1.0以上）
    """
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0

    return cer(reference, hypothesis)


def evaluate_model(
    model,
    tokenizer,
    dataset,
    data_collator,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
    num_samples: int = None,
):
    """
    モデルを評価してAccuracy, Top-k Accuracy, CERを計算する

    Args:
        model: AutoModelForMaskedLM
        tokenizer: AutoTokenizer
        dataset: 評価データセット
        data_collator: DataCollatorForLanguageModeling
        batch_size: バッチサイズ
        max_length: 最大シーケンス長
        device: デバイス
        num_samples: 評価するサンプル数（Noneの場合は全件）

    Returns:
        評価結果の辞書
    """
    model.eval()
    model.to(device)

    # サンプル数制限
    if num_samples is not None and num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = dataset.select(indices)

    print(f"評価サンプル数: {len(dataset)}")

    # メトリクス計算用
    all_predictions = []
    all_labels = []

    # Top-k accuracy用
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_masked = 0

    # サンプル別CER計算用
    sample_cer_scores = []

    # バッチ処理
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))

            # バッチデータを準備
            batch_samples = [dataset[i] for i in range(start_idx, end_idx)]

            # データコレーターでマスキング
            batch = data_collator(batch_samples)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # モデル予測
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Top-1予測
            predictions = logits.argmax(dim=-1)

            # Top-3, Top-5予測
            top3_preds = torch.topk(logits, k=3, dim=-1).indices
            top5_preds = torch.topk(logits, k=5, dim=-1).indices

            # マスク位置のみ評価
            mask_positions = labels != -100

            for i in range(input_ids.size(0)):
                sample_mask = mask_positions[i]

                if sample_mask.sum() == 0:
                    continue

                # マスク位置の予測とラベル
                sample_preds = predictions[i][sample_mask]
                sample_labels = labels[i][sample_mask]
                sample_top3 = top3_preds[i][sample_mask]
                sample_top5 = top5_preds[i][sample_mask]

                # Top-k accuracy計算
                for j in range(len(sample_labels)):
                    true_label = sample_labels[j].item()
                    pred_label = sample_preds[j].item()

                    total_masked += 1

                    if pred_label == true_label:
                        top1_correct += 1
                    if true_label in sample_top3[j].tolist():
                        top3_correct += 1
                    if true_label in sample_top5[j].tolist():
                        top5_correct += 1

                all_predictions.extend(sample_preds.cpu().numpy())
                all_labels.extend(sample_labels.cpu().numpy())

                # CER計算（マスク位置のみ）
                # 元のトークンを復元
                original_tokens = input_ids[i].clone()
                original_tokens[sample_mask] = sample_labels

                # 予測で置換したトークン
                restored_tokens = input_ids[i].clone()
                restored_tokens[sample_mask] = sample_preds

                # テキストに変換
                original_text = tokenizer.decode(original_tokens, skip_special_tokens=True)
                restored_text = tokenizer.decode(restored_tokens, skip_special_tokens=True)

                # CER計算
                cer = calculate_cer(original_text, restored_text)
                sample_cer_scores.append(cer)

    # 全体メトリクス計算
    if total_masked > 0:
        top1_accuracy = top1_correct / total_masked
        top3_accuracy = top3_correct / total_masked
        top5_accuracy = top5_correct / total_masked
    else:
        top1_accuracy = 0.0
        top3_accuracy = 0.0
        top5_accuracy = 0.0

    # Precision, Recall, F1計算
    if len(all_predictions) > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="macro", zero_division=0
        )
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0

    # 平均CER
    mean_cer = np.mean(sample_cer_scores) if sample_cer_scores else 0.0
    std_cer = np.std(sample_cer_scores) if sample_cer_scores else 0.0

    results = {
        "total_samples": len(dataset),
        "total_masked_tokens": total_masked,
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "top5_accuracy": top5_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_cer": mean_cer,
        "std_cer": std_cer,
        "sample_cer_scores": sample_cer_scores,
    }

    return results


def show_examples(
    model,
    tokenizer,
    dataset,
    data_collator,
    num_examples: int = 10,
    device: str = "cuda",
):
    """
    復元例を表示する
    """
    model.eval()
    model.to(device)

    # サンプルを選択
    indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)
    samples = [dataset[int(i)] for i in indices]

    # マスキング
    batch = data_collator(samples)

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    print("\n" + "=" * 80)
    print("マスクされたトークンの復元例")
    print("=" * 80)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        predictions = logits.argmax(dim=-1)
        top3_preds = torch.topk(logits, k=3, dim=-1).indices

        for i in range(input_ids.size(0)):
            mask_positions = labels[i] != -100
            mask_count = mask_positions.sum().item()

            if mask_count == 0:
                continue

            # 元のテキスト
            original_tokens = input_ids[i].clone()
            original_tokens[mask_positions] = labels[i][mask_positions]
            original_text = tokenizer.decode(original_tokens, skip_special_tokens=True)

            # マスクされたテキスト
            masked_text = tokenizer.decode(input_ids[i], skip_special_tokens=False)
            for special in (tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token, tokenizer.eos_token):
                if special:
                    masked_text = masked_text.replace(special, "")
            masked_text = masked_text.replace(tokenizer.mask_token, "　")

            # 復元されたテキスト
            restored_tokens = input_ids[i].clone()
            restored_tokens[mask_positions] = predictions[i][mask_positions]
            restored_text = tokenizer.decode(restored_tokens, skip_special_tokens=True)

            # CER計算
            cer = calculate_cer(original_text, restored_text)

            # 正解/不正解の判定
            correct_count = (predictions[i][mask_positions] == labels[i][mask_positions]).sum().item()

            print(f"\n例 {i + 1} (マスク数: {mask_count}, 正解数: {correct_count}/{mask_count}, CER: {cer:.4f})")
            print("-" * 60)
            print(f"元の文:     {original_text}")
            print(f"マスク文:   {masked_text}")
            print(f"復元文:     {restored_text}")

            # Top3候補を表示
            mask_indices = torch.where(mask_positions)[0]
            if len(mask_indices) > 0 and len(mask_indices) <= 5:
                print("マスク位置のTop3候補:")
                for j, pos in enumerate(mask_indices):
                    true_token = tokenizer.decode([labels[i][pos].item()])
                    pred_tokens = [tokenizer.decode([idx.item()]) for idx in top3_preds[i][pos]]
                    is_correct = "✓" if labels[i][pos].item() == predictions[i][pos].item() else "×"
                    print(f"  位置{j + 1}: 正解='{true_token}' 予測={pred_tokens} {is_correct}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test a language model for Kuzushiji recognition.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250526_141154/checkpoint-1200000",
        help="Path to the checkpoint directory (tokenizer will also be loaded from here).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Kotomiya07/honkoku-hq",
        help="Hugging Face dataset name or path.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset configuration name (optional).",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Name of the text column in the dataset. If None, will try to detect automatically.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use for testing.",
    )
    parser.add_argument(
        "--mask_probability",
        type=float,
        default=0.15,
        help="Probability of masking tokens.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples to show.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    # シード設定
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # モデルとトークナイザーを同じチェックポイントから読み込み
    print(f"Loading model and tokenizer from: {args.checkpoint_path}")
    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # max_length設定
    max_length = args.max_length
    if max_length is None:
        max_length = model.config.max_position_embeddings
        print(f"Using model's max_position_embeddings={max_length} as max_length")

    # データセット読み込み
    print(f"Loading dataset: {args.dataset_name}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)

    # データセット構造確認
    print(f"Dataset splits: {list(dataset.keys())}")

    # trainスプリットを使用
    if "train" in dataset:
        dataset = dataset["train"]
    else:
        all_splits = list(dataset.keys())
        dataset = dataset[all_splits[0]]

    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")

    # テキストカラム検出
    text_column = args.text_column
    if text_column is None:
        possible_text_columns = ["text", "sentence", "content", "transcription", "honkoku"]
        available_columns = list(dataset.features.keys())
        print(f"Available columns: {available_columns}")
        for col in possible_text_columns:
            if col in available_columns:
                text_column = col
                print(f"Using text column: {text_column}")
                break

        if text_column is None:
            for col in dataset.features.keys():
                if dataset.features[col].dtype == "string":
                    text_column = col
                    print(f"Using first string column as text: {text_column}")
                    break

        if text_column is None:
            raise ValueError(f"Could not detect text column. Available columns: {available_columns}")

    # テキストカラムをリネーム
    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
        print(f"Renamed column '{text_column}' to 'text'")

    # シーケンス長がmax_length以下のデータのみをフィルタリング
    # max_length - 1 を使用（特殊トークンの余地を確保）
    max_text_length = max_length - 2  # [CLS]と[SEP]の分を引く
    print(f"Filtering dataset to texts with length <= {max_text_length} characters...")
    original_size = len(dataset)
    dataset = dataset.filter(lambda x: len(x["text"]) <= max_text_length, desc="Filtering by length")
    filtered_size = len(dataset)
    print(f"Filtered dataset: {original_size} -> {filtered_size} ({filtered_size / original_size * 100:.1f}%)")

    # テストデータ抽出
    print(f"Extracting {args.test_size * 100:.0f}% of data for testing...")
    test_size = int(len(dataset) * args.test_size)
    indices = np.random.choice(len(dataset), test_size, replace=False)
    test_dataset = dataset.select(indices)
    print(f"Test dataset size: {len(test_dataset)}")

    # トークナイズ
    vocab_size = model.config.vocab_size
    unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_token_type_ids=False,
        )
        if "token_type_ids" in tokenized:
            del tokenized["token_type_ids"]

        # 無効なトークンIDをUNKに置換
        validated_input_ids = []
        for ids in tokenized["input_ids"]:
            ids_list = list(ids)
            ids_list = [unk_token_id if (tid < 0 or tid >= vocab_size) else tid for tid in ids_list]
            validated_input_ids.append(ids_list)
        tokenized["input_ids"] = validated_input_ids

        return tokenized

    print("Tokenizing dataset...")
    tokenized_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing")

    # フォーマット設定
    columns_to_set = [col for col in ["input_ids", "attention_mask"] if col in tokenized_dataset.column_names]
    tokenized_dataset.set_format(type="torch", columns=columns_to_set)
    print(f"Tokenized dataset columns: {tokenized_dataset.column_names}")

    # データコレーター
    class TestDataCollator(DataCollatorForLanguageModeling):
        def __init__(self, *args, vocab_size=None, max_length=None, unk_token_id=0, **kwargs):
            super().__init__(*args, **kwargs)
            self.vocab_size = vocab_size
            self.max_length = max_length
            self.unk_token_id = unk_token_id

        def __call__(self, features):
            # 長さ制限
            if self.max_length is not None:
                for feature in features:
                    if "input_ids" in feature:
                        input_ids = feature["input_ids"]
                        if isinstance(input_ids, torch.Tensor):
                            seq_len = input_ids.size(0) if input_ids.dim() > 0 else len(input_ids)
                        else:
                            seq_len = len(input_ids)

                        if seq_len > self.max_length:
                            feature["input_ids"] = input_ids[: self.max_length]
                            if "attention_mask" in feature:
                                feature["attention_mask"] = feature["attention_mask"][: self.max_length]

            batch = super().__call__(features)

            if "token_type_ids" in batch:
                del batch["token_type_ids"]

            # 長さ制限（バッチ後）
            if self.max_length is not None and "input_ids" in batch:
                seq_len = batch["input_ids"].size(1)
                if seq_len > self.max_length:
                    batch["input_ids"] = batch["input_ids"][:, : self.max_length]
                    if "attention_mask" in batch:
                        batch["attention_mask"] = batch["attention_mask"][:, : self.max_length]
                    if "labels" in batch:
                        batch["labels"] = batch["labels"][:, : self.max_length]

            # 無効なトークンIDをバリデーション
            if self.vocab_size is not None and "input_ids" in batch:
                input_ids = batch["input_ids"]
                invalid_mask = (input_ids < 0) | (input_ids >= self.vocab_size)
                if invalid_mask.any():
                    batch["input_ids"] = torch.where(invalid_mask, self.unk_token_id, input_ids)

            if self.vocab_size is not None and "labels" in batch:
                labels = batch["labels"]
                # labelsは-100（無視）以外で無効なIDをチェック
                valid_label_mask = (labels != -100) & ((labels < 0) | (labels >= self.vocab_size))
                if valid_label_mask.any():
                    batch["labels"] = torch.where(valid_label_mask, -100, labels)

            return batch

    data_collator = TestDataCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mask_probability,
        return_tensors="pt",
        vocab_size=model.config.vocab_size,
        max_length=max_length,
        unk_token_id=unk_token_id,
    )
    print(f"Data collator initialized with mask_probability={args.mask_probability}")

    # 評価実行
    print("\n" + "=" * 80)
    print("モデル評価開始")
    print("=" * 80)

    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized_dataset,
        data_collator=data_collator,
        batch_size=args.batch_size,
        max_length=max_length,
        device=device,
    )

    # 結果表示
    print("\n" + "=" * 80)
    print("評価結果")
    print("=" * 80)
    print(f"テストサンプル数: {results['total_samples']}")
    print(f"マスクされたトークン総数: {results['total_masked_tokens']}")
    print()
    print("【Accuracy】")
    print(f"  Top-1 Accuracy: {results['top1_accuracy']:.4f} ({results['top1_accuracy'] * 100:.2f}%)")
    print(f"  Top-3 Accuracy: {results['top3_accuracy']:.4f} ({results['top3_accuracy'] * 100:.2f}%)")
    print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f} ({results['top5_accuracy'] * 100:.2f}%)")
    print()
    print("【Precision / Recall / F1】")
    print(f"  Precision (macro): {results['precision']:.4f}")
    print(f"  Recall (macro): {results['recall']:.4f}")
    print(f"  F1 (macro): {results['f1']:.4f}")
    print()
    print("【CER (Character Error Rate)】")
    print(f"  Mean CER: {results['mean_cer']:.4f} ({results['mean_cer'] * 100:.2f}%)")
    print(f"  Std CER: {results['std_cer']:.4f}")
    print("=" * 80)

    # 復元例表示
    print("\n復元例を表示します...")
    show_examples(
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized_dataset,
        data_collator=data_collator,
        num_examples=args.num_examples,
        device=device,
    )

    # 結果をファイルに保存
    output_dir = os.path.dirname(args.checkpoint_path)
    results_file = os.path.join(output_dir, "test_results.txt")

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("言語モデル評価結果\n")
        f.write("=" * 80 + "\n")
        f.write(f"チェックポイント: {args.checkpoint_path}\n")
        f.write(f"データセット: {args.dataset_name}\n")
        f.write(f"テストサイズ: {args.test_size * 100:.0f}%\n")
        f.write(f"マスク確率: {args.mask_probability}\n")
        f.write(f"シード: {args.seed}\n")
        f.write("\n")
        f.write(f"テストサンプル数: {results['total_samples']}\n")
        f.write(f"マスクされたトークン総数: {results['total_masked_tokens']}\n")
        f.write("\n")
        f.write("【Accuracy】\n")
        f.write(f"  Top-1 Accuracy: {results['top1_accuracy']:.4f} ({results['top1_accuracy'] * 100:.2f}%)\n")
        f.write(f"  Top-3 Accuracy: {results['top3_accuracy']:.4f} ({results['top3_accuracy'] * 100:.2f}%)\n")
        f.write(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f} ({results['top5_accuracy'] * 100:.2f}%)\n")
        f.write("\n")
        f.write("【Precision / Recall / F1】\n")
        f.write(f"  Precision (macro): {results['precision']:.4f}\n")
        f.write(f"  Recall (macro): {results['recall']:.4f}\n")
        f.write(f"  F1 (macro): {results['f1']:.4f}\n")
        f.write("\n")
        f.write("【CER (Character Error Rate)】\n")
        f.write(f"  Mean CER: {results['mean_cer']:.4f} ({results['mean_cer'] * 100:.2f}%)\n")
        f.write(f"  Std CER: {results['std_cer']:.4f}\n")
        f.write("=" * 80 + "\n")

    print(f"\n結果を保存しました: {results_file}")


if __name__ == "__main__":
    main()
