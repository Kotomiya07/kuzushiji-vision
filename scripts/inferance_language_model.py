# %%
import glob
import os
import random

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250517_091839/final_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

dataset_dirs = ["kokubunken_repo/text/eirigenji"]
num_samples = 10
random_sample = True


def load_texts(dataset_dirs: list[str]) -> list[str]:
    texts = []
    count = 0
    for dataset_dir in dataset_dirs:
        text_files_iterator = glob.iglob(os.path.join(dataset_dir, "**/*.txt"), recursive=True)
        for text_file in text_files_iterator:
            with open(text_file, encoding="utf-8") as f:
                text = f.read().strip()
            texts.append(text)
            count += 1

            if count >= 10000:
                break

    print(f"Loaded {len(texts)} texts")

    return texts


def random_mask_text(text: str) -> str:
    """
    文字列をランダムにマスクする
    input: str
    output: str
    """
    # テキストをトークン化してIDに変換
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]  # バッチ次元を削除

    # ランダムにマスクを適用
    masked_input_ids = []
    mask_token_id = tokenizer.mask_token_id

    # 特殊トークンを除外してマスク対象を選ぶ
    special_tokens = tokenizer.all_special_ids
    non_special_token_indices = [i for i, token_id in enumerate(input_ids) if token_id not in special_tokens]

    num_tokens_to_mask = max(1, int(len(non_special_token_indices) * 0.15)) # 最低1つはマスク
    indices_to_mask = random.sample(non_special_token_indices, min(num_tokens_to_mask, len(non_special_token_indices)))


    for i, token_id in enumerate(input_ids):
        if i in indices_to_mask:
            masked_input_ids.append(mask_token_id)
        else:
            masked_input_ids.append(token_id)

    # デコードして文字列に戻す
    return tokenizer.decode(masked_input_ids, skip_special_tokens=False)


if random_sample:
    texts = random.sample(load_texts(dataset_dirs), num_samples)
else:
    texts = load_texts(dataset_dirs)[:num_samples]

for text in texts:
    masked_text = random_mask_text(text)
    inputs = tokenizer(masked_text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    
    # マスクされたトークンのすべてのインデックスを取得
    mask_token_indices = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    if len(mask_token_indices) > 0:
        # 復元された文を構築するための入力IDのコピー
        restored_input_ids = inputs.input_ids[0].clone()

        # 最初のマスクトークンの予測結果（Top 5）を表示用に保存
        top_5_predictions_for_first_mask = []
        if len(mask_token_indices) > 0:
            first_mask_token_logits = logits[0, mask_token_indices[0], :]
            top_5_tokens = torch.topk(first_mask_token_logits, 5, dim=-1)
            for token_id, score in zip(top_5_tokens.indices, top_5_tokens.values, strict=False):
                predicted_token = tokenizer.decode([token_id.item()])
                top_5_predictions_for_first_mask.append((predicted_token, score.item()))

        # すべてのマスクされたトークンをTop-1予測で置き換えて文を復元
        for mask_idx in mask_token_indices:
            current_mask_token_logits = logits[0, mask_idx, :]
            top_1_token_id = torch.argmax(current_mask_token_logits, dim=-1).item()
            restored_input_ids[mask_idx] = top_1_token_id

        # 復元された文をデコード（特殊トークンはスキップ）
        restored_sentence = tokenizer.decode(restored_input_ids, skip_special_tokens=True)

        print(f"Original text: {text[:100]}...")
        print(f"Masked text: 　{masked_text}")
        print(f"Restored sentence (Top-1): {restored_sentence}") # 復元された文を表示

        print("Top 5 predictions (for the first mask token):")
        for i, (token, score) in enumerate(top_5_predictions_for_first_mask):
            print(f"  {i + 1}. {token}: {score:.2f}")
        print("-" * 50)
    else:
        print(f"No mask tokens found in: {masked_text}")
        print("-" * 50)

# %%
