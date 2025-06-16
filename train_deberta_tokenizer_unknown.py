#!/usr/bin/env python3
"""
DeBERTa v3 Japanese XSmall Tokenizer Unknown Characters Learning

このスクリプトは、globis-university/deberta-v3-japanese-xsmall のトークナイザーを使用して、
data/honkoku/honkoku.txt の文章を1文ずつ処理し、unknown（[UNK]）になった文字だけを
トークナイザーの辞書に追加してトークナイザーを学習します。
"""

import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer


class DeBERTaTokenizerExtender:
    """DeBERTa トークナイザーを拡張するクラス"""

    def __init__(self, model_name: str = "globis-university/deberta-v3-japanese-xsmall"):
        """
        初期化

        Args:
            model_name: 元のトークナイザーのモデル名
        """
        self.model_name = model_name
        self.original_tokenizer = None
        self.extended_tokenizer = None
        self.unknown_chars = set()
        self.sentences = []

        # 設定項目
        self.input_file = "data/honkoku/honkoku.txt"
        self.output_dir = "experiments/deberta_v3_japanese_extended_tokenizer"
        self.max_vocab_size = 40000  # 元の32000から拡張
        self.min_frequency = 1

        # 特殊トークン（DeBERTa v3の形式に合わせる）
        self.special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

    def load_original_tokenizer(self) -> None:
        """元のDeBERTa トークナイザーをロード"""
        print(f"元のトークナイザーをロード中: {self.model_name}")
        try:
            self.original_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print(f"元の語彙サイズ: {len(self.original_tokenizer)}")
            print(f"UNKトークン: {self.original_tokenizer.unk_token}")
        except Exception as e:
            print(f"トークナイザーのロードに失敗: {e}")
            raise

    def load_text_file(self) -> None:
        """テキストファイルを読み込み、文単位で分割"""
        print(f"テキストファイルを読み込み中: {self.input_file}")

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"ファイルが見つかりません: {self.input_file}")

        try:
            with open(self.input_file, encoding="utf-8") as f:
                content = f.read()

            # 文単位での分割（改行文字で分割し、空行を除去）
            raw_sentences = content.split("\n")
            self.sentences = [s.strip() for s in raw_sentences if s.strip()]

            print(f"読み込み完了: {len(self.sentences)}文")

        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
            raise

    def detect_unknown_characters(self) -> None:
        """各文を処理してUNK文字を検出"""
        print("未知文字を検出中...")

        unknown_chars = set()
        unk_token_id = self.original_tokenizer.unk_token_id
        all_chars = set()
        unk_sentences = []

        print(f"UNKトークンID: {unk_token_id}")

        for i, sentence in enumerate(tqdm(self.sentences, desc="文を処理中")):
            if not sentence:
                continue

            # 全ての文字を記録
            for char in sentence:
                all_chars.add(char)

            # トークナイゼーション実行
            encoding = self.original_tokenizer(sentence, add_special_tokens=False, return_tensors="pt")
            token_ids = encoding.input_ids[0].tolist()
            tokens = self.original_tokenizer.tokenize(sentence)

            # UNKトークンが含まれているかチェック
            if unk_token_id in token_ids:
                unk_sentences.append((i, sentence))

                # UNKトークンの位置を特定
                unk_positions = [idx for idx, tid in enumerate(token_ids) if tid == unk_token_id]

                # UNKトークンに対応する文字を特定
                for pos in unk_positions:
                    if pos < len(tokens):
                        unk_token = tokens[pos]
                        # UNKトークンが実際の文字である場合、それを未知文字として記録
                        if unk_token != "[UNK]" and len(unk_token) == 1:
                            unknown_chars.add(unk_token)

                # 詳細表示は最初の5例のみ
                if len(unk_sentences) <= 5:
                    print(f"\nUNK含む文 {i}: {sentence[:100]}...")
                    print(f"  トークン: {tokens[:15]}...")
                    print(f"  UNK位置: {unk_positions}")
                    self._analyze_unk_tokens(sentence, tokens, unk_positions)

        # 代替方法: 稀な文字を個別にテスト
        print("\n稀な文字の個別テスト...")
        char_freq = {}
        for sentence in self.sentences:
            for char in sentence:
                char_freq[char] = char_freq.get(char, 0) + 1

        # 出現頻度の低い文字をテスト（頻度1-10の文字）
        rare_chars = [char for char, freq in char_freq.items() if 1 <= freq <= 10]
        print(f"稀な文字（出現1-10回）: {len(rare_chars)}個")

        rare_unk_chars = set()
        for char in rare_chars:
            # 文字単体でテスト
            char_encoding = self.original_tokenizer(char, add_special_tokens=False, return_tensors="pt")
            char_token_ids = char_encoding.input_ids[0].tolist()
            if len(char_token_ids) == 1 and char_token_ids[0] == unk_token_id:
                rare_unk_chars.add(char)
                if len(rare_unk_chars) <= 20:  # 最初の20個だけ表示
                    print(f"  単体でUNKになる文字: '{char}' (U+{ord(char):04X}, 頻度: {char_freq[char]})")

        self.unknown_chars = unknown_chars.union(rare_unk_chars)

        print("\n=== 分析結果 ===")
        print(f"処理した文数: {len(self.sentences)}")
        print(f"UNKを含む文数: {len(unk_sentences)}")
        print(f"全文字種数: {len(all_chars)}")
        print(f"検出された未知文字数: {len(self.unknown_chars)}")

        if self.unknown_chars:
            print(f"未知文字: {sorted(self.unknown_chars)}")
            print(f"未知文字（Unicode）: {[f'U+{ord(c):04X}' for c in sorted(self.unknown_chars)]}")

        # 文字の分布を表示
        print("\n最も頻度の高い文字トップ10:")
        for char, freq in sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  '{char}': {freq}回")

        print(f"\n1回のみ出現する文字数: {len([c for c in char_freq if char_freq[c] == 1])}")

        # 未知文字の使用例を表示
        if self.unknown_chars:
            print("\n未知文字の使用例:")
            for char in sorted(self.unknown_chars)[:10]:
                examples = []
                for sentence in self.sentences:
                    if char in sentence:
                        examples.append(sentence[:50] + "...")
                        if len(examples) >= 2:
                            break
                print(f"  '{char}': {examples}")

    def _analyze_unk_tokens(self, sentence: str, tokens: list[str], unk_positions: list[int]) -> None:
        """UNKトークンの詳細分析"""
        print("    UNKトークンの詳細分析:")

        for pos in unk_positions:
            if pos < len(tokens):
                print(f"      位置 {pos}: トークン '{tokens[pos]}'")

                # 前後のトークンも表示
                context_start = max(0, pos - 2)
                context_end = min(len(tokens), pos + 3)
                context = tokens[context_start:context_end]
                print(f"      文脈: {context}")

        # 文字列の復元を試行してUNK箇所を特定
        try:
            # トークンを文字列に戻す
            decoded = self.original_tokenizer.convert_tokens_to_string(tokens)
            print(f"      復元文字列: {decoded[:100]}...")

        except Exception as e:
            print(f"      復元エラー: {e}")

    def _find_differences(self, original: str, decoded: str) -> None:
        """元の文字列と復元文字列の違いを特定"""
        print("        差異分析:")
        min_len = min(len(original), len(decoded))

        # 文字単位で比較
        for i in range(min_len):
            if original[i] != decoded[i]:
                print(f"          位置 {i}: '{original[i]}' -> '{decoded[i]}'")
                # この文字がUNKになる可能性
                char_test = self.original_tokenizer(original[i], add_special_tokens=False)
                print(f"          '{original[i]}'のトークン化: {char_test.tokens}")
                break

        if len(original) != len(decoded):
            print(f"        長さが異なります: {len(original)} vs {len(decoded)}")

    def create_extended_vocabulary(self) -> dict[str, int]:
        """拡張された語彙辞書を作成"""
        print("拡張語彙を作成中...")

        # 元の語彙を取得
        original_vocab = self.original_tokenizer.get_vocab()
        print(f"元の語彙サイズ: {len(original_vocab)}")

        # 新しい語彙辞書を作成
        extended_vocab = original_vocab.copy()

        # 未知文字を追加
        next_id = max(original_vocab.values()) + 1

        for char in sorted(self.unknown_chars):
            if char not in extended_vocab:
                extended_vocab[char] = next_id
                next_id += 1

        print(f"拡張後の語彙サイズ: {len(extended_vocab)}")
        return extended_vocab

    def train_extended_tokenizer(self) -> None:
        """拡張された語彙で新しいトークナイザーを学習"""
        print("拡張トークナイザーを学習中...")

        # 元のトークナイザーの設定を取得
        original_tokenizer = self.original_tokenizer

        # 新しい語彙辞書を作成
        original_vocab = original_tokenizer.get_vocab()
        print(f"元の語彙サイズ: {len(original_vocab)}")

        # 未知文字を語彙に追加
        new_vocab = original_vocab.copy()
        next_id = max(original_vocab.values()) + 1

        for char in sorted(self.unknown_chars):
            if char not in new_vocab:
                new_vocab[char] = next_id
                next_id += 1

        print(f"拡張後の語彙サイズ: {len(new_vocab)}")

        # 新しい語彙で更新

        # SentencePieceモデルではなく、直接語彙を設定する方法を試す
        # 元のトークナイザーの構造をそのまま使用し、語彙のみ拡張
        try:
            # 元のトークナイザーをコピー
            import copy

            new_tokenizer = copy.deepcopy(original_tokenizer)

            # 語彙を拡張する（HuggingFaceトークナイザーの場合）
            for char in sorted(self.unknown_chars):
                if char not in new_tokenizer.get_vocab():
                    new_tokenizer.add_tokens([char])

            # 拡張されたトークナイザーを保存
            self.extended_tokenizer = new_tokenizer
            print(f"語彙拡張完了: {len(original_vocab)} → {len(new_tokenizer.get_vocab())}")

        except Exception as e:
            print(f"トークナイザー拡張エラー: {e}")
            print("代替方法を試行中...")

            # 代替方法: 元のトークナイザーに直接トークンを追加
            new_tokens = []

            for char in sorted(self.unknown_chars):
                if char not in original_tokenizer.get_vocab():
                    new_tokens.append(char)

            if new_tokens:
                print(f"追加するトークン数: {len(new_tokens)}")
                num_added = original_tokenizer.add_tokens(new_tokens)
                print(f"実際に追加されたトークン数: {num_added}")

            self.extended_tokenizer = original_tokenizer
            print("語彙拡張が完了しました")

    def save_extended_tokenizer(self) -> None:
        """拡張トークナイザーを保存"""
        print(f"トークナイザーを保存中: {self.output_dir}")

        # 出力ディレクトリを作成
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            # HuggingFace形式で保存
            self.extended_tokenizer.save_pretrained(self.output_dir)

            # 統計情報を保存
            original_vocab_size = len(self.original_tokenizer.get_vocab())
            extended_vocab_size = len(self.extended_tokenizer.get_vocab())

            stats = {
                "original_model": self.model_name,
                "original_vocab_size": original_vocab_size,
                "extended_vocab_size": extended_vocab_size,
                "unknown_characters_added": len(self.unknown_chars),
                "unknown_characters": sorted(self.unknown_chars),
                "total_sentences_processed": len(self.sentences),
                "vocab_increase": extended_vocab_size - original_vocab_size,
            }

            with open(os.path.join(self.output_dir, "extension_stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            print(f"保存完了: {self.output_dir}")
            print(f"語彙サイズ: {original_vocab_size} → {extended_vocab_size}")

        except Exception as e:
            print(f"保存エラー: {e}")
            raise

    def test_tokenizer(self) -> None:
        """新しいトークナイザーをテスト"""
        print("\nトークナイザーをテスト中...")

        # 保存されたトークナイザーをロード
        try:
            new_tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        except Exception as e:
            print(f"新しいトークナイザーのロードに失敗: {e}")
            return

        # 実際に未知文字を含むテスト文を使用
        test_sentences = [
            "て阿蘇噴火脈の蹝を垂るゝのみ美濃尾張近江の三国には従来の調査に拠れば一箇の",
            "夏莱菔は真盛り、二",
            "此レヲシテ豊饒ナラシメ又早晹ノ雰囲気中ニ配分シテ之",
            "バー」ノ若干湖及𤇆煙ヲ発出スル尖円ノ小群山ヲ見ル",
            "一般的な文章でUNKが発生しないことを確認する",
        ]

        print("\n=== テスト結果 ===")
        for i, sentence in enumerate(test_sentences):
            print(f"\nテスト文 {i + 1}: {sentence}")

            # 元のトークナイザー
            original_tokens = self.original_tokenizer.tokenize(sentence)
            original_unk_count = original_tokens.count("[UNK]")

            # 新しいトークナイザー
            new_tokens = new_tokenizer.tokenize(sentence)
            new_unk_count = new_tokens.count("[UNK]")

            print(f"  元のトークナイザー: {original_tokens}")
            print(f"  UNK数: {original_unk_count}")
            print(f"  新しいトークナイザー: {new_tokens}")
            print(f"  UNK数: {new_unk_count}")
            print(f"  改善: {original_unk_count - new_unk_count} UNK削減")

            # 未知文字が含まれていた場合の詳細分析
            if original_unk_count > 0:
                print("  詳細分析:")
                for j, (orig_token, new_token) in enumerate(zip(original_tokens, new_tokens, strict=False)):
                    if orig_token == "[UNK]" and new_token != "[UNK]":
                        print(f"    位置 {j}: '[UNK]' → '{new_token}' (改善)")
                    elif orig_token != "[UNK]" and new_token == "[UNK]":
                        print(f"    位置 {j}: '{orig_token}' → '[UNK]' (悪化)")

        # 追加された未知文字のテスト
        print("\n=== 追加された未知文字のテスト ===")
        unknown_char_samples = ["蹝", "菔", "晹", "𤇆", "㐫", "℥"]

        for char in unknown_char_samples:
            if char in self.unknown_chars:
                # 元のトークナイザー
                orig_result = self.original_tokenizer.tokenize(char)
                # 新しいトークナイザー
                new_result = new_tokenizer.tokenize(char)

                print(f"文字 '{char}': 元={orig_result} → 新={new_result}")

                # 単語として含まれる場合もテスト
                test_word = f"これは{char}です"
                orig_word = self.original_tokenizer.tokenize(test_word)
                new_word = new_tokenizer.tokenize(test_word)
                print(f"  文中 '{test_word}': 元={orig_word} → 新={new_word}")
                print(f"  UNK削減: {orig_word.count('[UNK]') - new_word.count('[UNK]')}")
                print()

    def run(self) -> None:
        """メインの実行プロセス"""
        print("DeBERTa v3 Japanese XSmall トークナイザー拡張を開始...")

        try:
            # 1. 元のトークナイザーをロード
            self.load_original_tokenizer()

            # 2. テキストファイルを読み込み
            self.load_text_file()

            # 3. 未知文字を検出
            self.detect_unknown_characters()

            if not self.unknown_chars:
                print("未知文字が見つかりませんでした。処理を終了します。")
                return

            # 4. 拡張トークナイザーを学習
            self.train_extended_tokenizer()

            # 5. トークナイザーを保存
            self.save_extended_tokenizer()

            # 6. テスト実行
            self.test_tokenizer()

            print("\n処理が正常に完了しました！")

        except Exception as e:
            print(f"エラーが発生しました: {e}")
            raise


def main():
    """メイン関数"""
    extender = DeBERTaTokenizerExtender()
    extender.run()


if __name__ == "__main__":
    main()
