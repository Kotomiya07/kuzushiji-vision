"""
翻刻テキストを使用して、モデルに文字と文字のつながりを学習させるスクリプト
"""

import argparse
import glob
import os
from datetime import datetime

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


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
        ],
        help="Directory containing the text dataset files.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./pretrain_output", help="Output directory for a_model_name and checkpoints."
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size for evaluation.")
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

    args = parser.parse_args()

    # 1. Vocab and Tokenizer
    DEFAULT_VOCAB_SIZE = args.vocab_size
    DEFAULT_MODEL_TYPE = args.model_type
    BASE_EXPERIMENT_DIR = "experiments/kuzushiji_tokenizer_one_char"
    MODEL_SPECIFIC_DIR_NAME = f"vocab{DEFAULT_VOCAB_SIZE}_{DEFAULT_MODEL_TYPE}"
    os.path.join(BASE_EXPERIMENT_DIR, MODEL_SPECIFIC_DIR_NAME)

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

    output_dir = os.path.join(args.output_dir, args.model_name.split("/")[-1], datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=24,
        torch_compile=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size if not args.auto_find_batch_size else 512,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        auto_find_batch_size=args.auto_find_batch_size,
        use_cpu=True,
    )
    if args.gradient_accumulation_steps:
        training_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    print("TrainingArguments configured.")

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
    if args.early_stopping_patience is not None:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
        )
        callbacks = [early_stopping_callback]
    else:
        callbacks = []

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        callbacks=callbacks,
    )
    print("Trainer initialized.")

    # # 8. Train
    # print("Starting training...")
    # trainer.train()
    # print("Training finished.")

    # # 9. Save model and tokenizer
    # final_model_path = os.path.join(output_dir, "final_model")
    # trainer.save_model(final_model_path)
    # tokenizer.save_pretrained(final_model_path)
    # print(f"Final model and tokenizer saved to {final_model_path}")

    # 10. Evaluate (if eval_dataset exists)
    if eval_dataset:
        print("Starting final evaluation...")
        eval_results = trainer.evaluate()
        print("Evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
    else:
        print("No evaluation dataset provided. Skipping final evaluation.")


if __name__ == "__main__":
    main()

# How to run
"""
python test_language_model.py \
    --output_dir experiments/pretrain_language_model \
    --num_train_epochs 1000 \
    --per_device_train_batch_size 1024 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --mask_probability 0.1 \
    --test_size 0.01 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --logging_steps 100 \
    --gradient_accumulation_steps 2
"""

"""
Found 66537 text files in ndl-minhon-ocrdataset/src/honkoku_oneline_v1
Found 360052 text files in ndl-minhon-ocrdataset/src/honkoku_oneline_v2
Found 84990 text files in honkoku_yatanavi/honkoku_oneline
Found 29603 text files in data/oneline
"""
