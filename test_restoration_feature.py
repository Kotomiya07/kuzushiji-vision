#!/usr/bin/env python3
"""
マスクされたトークンの復元機能をテストするスクリプト
"""

import glob
import os
import sys

from datasets import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

# train_language_model.pyから復元機能をインポート
sys.path.append(".")
from train_language_model import print_restoration_examples, restore_masked_text


def test_restoration_feature():
    """復元機能の基本的なテストを実行"""

    print("復元機能のテストを開始します...")

    # 1. トークナイザーとモデルの読み込み
    print("1. トークナイザーとモデルを読み込み中...")

    # 既存のトークナイザーを使用
    model_name = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250517_091839/final_model"   

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ トークナイザーを読み込みました: {model_name}")
    except Exception as e:
        print(f"エラー: トークナイザーの読み込みに失敗しました: {e}")
        return

    try:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
        print(f"✓ モデルを読み込みました: {model_name}")
    except Exception as e:
        print(f"エラー: モデルの読み込みに失敗しました: {e}")
        return

    # 2. テストデータの準備
    print("\n2. テストデータを準備中...")
    dataset_dirs = ["kokubunken_repo/text/eirigenji"]
    test_texts = []

    count = 0
    for dataset_dir in dataset_dirs:
        text_files_iterator = glob.iglob(os.path.join(dataset_dir, "**/*.txt"), recursive=True)
        for text_file in text_files_iterator:
            with open(text_file, encoding="utf-8") as f:
                text = f.read().strip()
            test_texts.append(text)
            count += 1
            
            if count >= 10:
                break

    # データセットを作成
    dataset_dict = {"text": test_texts}
    dataset = Dataset.from_dict(dataset_dict)

    # トークナイズ
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print(f"✓ {len(test_texts)}個のテストテキストをトークナイズしました")

    # 3. マスクされたデータの生成
    print("\n3. マスクされたデータを生成中...")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # バッチを作成
    batch = data_collator([tokenized_dataset[i] for i in range(len(test_texts))])
    print("✓ マスクされたバッチを生成しました")

    # 4. 復元機能のテスト
    print("\n4. 復元機能をテスト中...")

    try:
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        restoration_results = restore_masked_text(
            model=model, tokenizer=tokenizer, input_ids=input_ids, labels=labels, max_examples=len(test_texts)
        )

        print(f"✓ {len(restoration_results)}個の復元結果を生成しました")

        # 結果を表示
        print_restoration_examples(restoration_results, max_display=len(test_texts))

        print("\n✓ 復元機能のテストが正常に完了しました！")

    except Exception as e:
        print(f"エラー: 復元機能のテスト中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_restoration_feature()
