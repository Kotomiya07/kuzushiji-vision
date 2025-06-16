#!/usr/bin/env python3
"""
拡張されたDeBERTaトークナイザーのテストスクリプト
"""

import json

from transformers import AutoTokenizer


def test_extended_tokenizer():
    """拡張されたトークナイザーをテスト"""
    print("=== 拡張DeBERTaトークナイザーのテスト ===\n")

    # 元のトークナイザーをロード
    original_tokenizer = AutoTokenizer.from_pretrained("globis-university/deberta-v3-japanese-xsmall")

    # 拡張されたトークナイザーをロード
    try:
        extended_tokenizer = AutoTokenizer.from_pretrained("experiments/deberta_v3_japanese_extended_tokenizer")
    except Exception as e:
        print(f"拡張トークナイザーのロードエラー: {e}")
        return

    # 統計情報を読み込み
    with open("experiments/deberta_v3_japanese_extended_tokenizer/extension_stats.json", encoding="utf-8") as f:
        stats = json.load(f)

    print("=== 拡張統計情報 ===")
    print(f"元の語彙サイズ: {stats['original_vocab_size']:,}")
    print(f"拡張後語彙サイズ: {stats['extended_vocab_size']:,}")
    print(f"追加された未知文字数: {stats['unknown_characters_added']:,}")
    print(f"処理された文数: {stats['total_sentences_processed']:,}")
    print(f"語彙増加数: {stats['vocab_increase']:,}")
    print()

    # 実際に未知文字を含むテスト文
    test_sentences = [
        "て阿蘇噴火脈の蹝を垂るゝのみ美濃尾張近江の三国には従来の調査に拠れば一箇の",
        "夏莱菔は真盛り、二",
        "此レヲシテ豊饒ナラシメ又早晹ノ雰囲気中ニ配分シテ之",
        "バー」ノ若干湖及𤇆煙ヲ発出スル尖円ノ小群山ヲ見ル",
        "一般的な文章でUNKが発生しないことを確認する",
    ]

    print("=== 文章レベルテスト ===")
    total_original_unk = 0
    total_extended_unk = 0

    for i, sentence in enumerate(test_sentences):
        print(f"\nテスト文 {i + 1}: {sentence}")

        # 元のトークナイザー
        original_tokens = original_tokenizer.tokenize(sentence)
        original_unk_count = original_tokens.count("[UNK]")
        total_original_unk += original_unk_count

        # 拡張されたトークナイザー
        extended_tokens = extended_tokenizer.tokenize(sentence)
        extended_unk_count = extended_tokens.count("[UNK]")
        total_extended_unk += extended_unk_count

        print(f"  元のトークナイザー: {original_tokens}")
        print(f"  UNK数: {original_unk_count}")
        print(f"  拡張トークナイザー: {extended_tokens}")
        print(f"  UNK数: {extended_unk_count}")
        print(f"  UNK削減: {original_unk_count - extended_unk_count}")

        # 詳細分析
        if original_unk_count > 0 or extended_unk_count > 0:
            print("  詳細分析:")
            max_len = max(len(original_tokens), len(extended_tokens))
            for j in range(max_len):
                orig_token = original_tokens[j] if j < len(original_tokens) else ""
                ext_token = extended_tokens[j] if j < len(extended_tokens) else ""

                if orig_token == "[UNK]" and ext_token != "[UNK]":
                    print(f"    位置 {j}: '[UNK]' → '{ext_token}' (改善)")
                elif orig_token != "[UNK]" and ext_token == "[UNK]":
                    print(f"    位置 {j}: '{orig_token}' → '[UNK]' (悪化)")

    # 個別文字のテスト
    print("\n=== 個別文字テスト ===")
    unknown_char_samples = ["蹝", "菔", "晹", "𤇆", "㐫", "℥", "◓", "㑹", "㔫", "㕝"]

    for char in unknown_char_samples:
        if char in stats["unknown_characters"]:
            # 単体文字のテスト
            orig_result = original_tokenizer.tokenize(char)
            ext_result = extended_tokenizer.tokenize(char)

            print(f"文字 '{char}' (U+{ord(char):04X}):")
            print(f"  元: {orig_result}")
            print(f"  拡張: {ext_result}")

            # 文中でのテスト
            test_word = f"これは{char}です"
            orig_word = original_tokenizer.tokenize(test_word)
            ext_word = extended_tokenizer.tokenize(test_word)
            orig_unk_in_word = orig_word.count("[UNK]")
            ext_unk_in_word = ext_word.count("[UNK]")

            print(f"  文中 '{test_word}':")
            print(f"    元: {orig_word} (UNK: {orig_unk_in_word})")
            print(f"    拡張: {ext_word} (UNK: {ext_unk_in_word})")
            print(f"    UNK削減: {orig_unk_in_word - ext_unk_in_word}")
            print()

    # 全体の統計
    print("=== 全体結果 ===")
    print(f"テスト文でのUNK削減: {total_original_unk} → {total_extended_unk}")
    print(f"UNK削減数: {total_original_unk - total_extended_unk}")
    if total_original_unk > 0:
        reduction_rate = (total_original_unk - total_extended_unk) / total_original_unk * 100
        print(f"UNK削減率: {reduction_rate:.1f}%")


if __name__ == "__main__":
    test_extended_tokenizer()
