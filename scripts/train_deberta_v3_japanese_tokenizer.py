#!/usr/bin/env python3
"""
globis-university/deberta-v3-japanese-xsmallモデルと同じ設定のtokenizerを学習するスクリプト

このスクリプトは、data/honkoku以下のテキストデータを使用して、
globis-university/deberta-v3-japanese-xsmallモデルと同じ設定の
DeBERTaV2TokenizerFastを学習します。
"""

import argparse
import glob
import os
import tempfile
from typing import Any

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import DebertaV2TokenizerFast


def get_reference_tokenizer_config() -> dict[str, Any]:
    """
    globis-university/deberta-v3-japanese-xsmallの設定を取得

    Returns:
        Dict[str, Any]: トークナイザーの設定辞書
    """
    print("Loading reference tokenizer: globis-university/deberta-v3-japanese-xsmall")
    reference_tokenizer = DebertaV2TokenizerFast.from_pretrained("globis-university/deberta-v3-japanese-xsmall")

    config = {
        "vocab_size": reference_tokenizer.vocab_size,
        "model_max_length": 512,  # 現実的な値に調整
        "pad_token": reference_tokenizer.pad_token,
        "unk_token": reference_tokenizer.unk_token,
        "cls_token": reference_tokenizer.cls_token,
        "sep_token": reference_tokenizer.sep_token,
        "mask_token": reference_tokenizer.mask_token,
        "pad_token_id": reference_tokenizer.pad_token_id,
        "unk_token_id": reference_tokenizer.unk_token_id,
        "cls_token_id": reference_tokenizer.cls_token_id,
        "sep_token_id": reference_tokenizer.sep_token_id,
        "mask_token_id": reference_tokenizer.mask_token_id,
        "do_lower_case": False,
        "split_by_punct": False,
        "clean_up_tokenization_spaces": True,
    }

    print("Reference config extracted:")
    print(f"  Vocab size: {config['vocab_size']}")
    print(
        f"  Special tokens: {[config['pad_token'], config['unk_token'], config['cls_token'], config['sep_token'], config['mask_token']]}"
    )

    return config


def get_all_text_files(data_dirs: list[str]) -> list[str]:
    """
    指定されたディレクトリ内のすべての.txtファイルへのパスのリストを取得

    Args:
        data_dirs: 検索対象のディレクトリリスト

    Returns:
        List[str]: ファイルパスのリスト
    """
    all_files = []
    for data_dir in data_dirs:
        if os.path.isfile(data_dir):
            # ファイルが直接指定された場合
            all_files.append(data_dir)
        else:
            # ディレクトリの場合、再帰的に.txtファイルを検索
            files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
            all_files.extend(files)

    if not all_files:
        raise FileNotFoundError(f"指定されたディレクトリ/ファイルにテキストファイルが見つかりませんでした: {data_dirs}")

    print(f"Found {len(all_files)} text files:")
    for file in all_files[:5]:  # 最初の5ファイルのみ表示
        print(f"  {file}")
    if len(all_files) > 5:
        print(f"  ... and {len(all_files) - 5} more files")

    return all_files


def concatenate_files(file_list: list[str], output_dir: str) -> str:
    """
    指定されたファイルリストの内容を1つの一時ファイルに結合

    Args:
        file_list: 結合するファイルのリスト
        output_dir: 出力ディレクトリ

    Returns:
        str: 結合されたファイルのパス
    """
    try:
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=output_dir,
            prefix="deberta_v3_train_data_",
            suffix=".txt",
        )
        print(f"Creating temporary training file: {temp_file.name}")

        total_lines = 0
        for filepath in file_list:
            try:
                with open(filepath, encoding="utf-8") as infile:
                    for line in infile:
                        line = line.strip()
                        if line:  # 空行をスキップ
                            temp_file.write(line + "\n")
                            total_lines += 1
            except Exception as e:
                print(f"Warning: Error reading file {filepath}: {e}")

        temp_file.close()
        print(f"Training data prepared: {total_lines} lines in {temp_file.name}")
        return temp_file.name

    except Exception as e:
        print(f"Error creating training file: {e}")
        if "temp_file" in locals() and temp_file.name and os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise


def create_deberta_v3_tokenizer(
    input_file_path: str, model_output_dir: str, model_filename_stem: str, config: dict[str, Any]
) -> None:
    """
    DeBERTa V3 Japaneseと同じ設定でtokenizerを学習

    Args:
        input_file_path: 学習データファイルパス
        model_output_dir: 出力ディレクトリ
        model_filename_stem: ファイル名の接頭辞
        config: トークナイザー設定
    """
    output_path = os.path.join(model_output_dir, f"{model_filename_stem}.json")
    print("Training tokenizer with DeBERTa V3 Japanese settings...")
    print(f"Output path: {output_path}")

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
        print(f"Created directory: {model_output_dir}")

    # 1. Normalizer設定 (DeBERTa V3 Japaneseと同じ)
    # NFKC正規化 + Strip + Precompiled + 連続空白の置換
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.Strip(left=True, right=True),
            # Precompiledは省略（学習時には不要）
            normalizers.Replace(pattern=r" {2,}", content=" "),
        ]
    )

    # 2. Pre-tokenizer設定 (Metaspace with ▁)
    pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)])

    # 3. Unigram モデルで初期化
    tokenizer_model = models.Unigram()
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer

    # 4. Decoder設定
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)

    # 5. 特殊トークンの設定
    special_tokens = [
        config["pad_token"],  # [PAD]
        config["cls_token"],  # [CLS]
        config["sep_token"],  # [SEP]
        config["unk_token"],  # [UNK]
        config["mask_token"],  # [MASK]
    ]

    print("Training parameters:")
    print(f"  Input file: {input_file_path}")
    print(f"  Vocab size: {config['vocab_size']}")
    print("  Model type: Unigram")
    print(f"  Special tokens: {special_tokens}")
    print(f"  Normalizer: {tokenizer.normalizer}")
    print(f"  Pre-tokenizer: {tokenizer.pre_tokenizer}")

    # 6. Trainer設定とトレーニング
    trainer = trainers.UnigramTrainer(
        vocab_size=config["vocab_size"],
        unk_token=config["unk_token"],
        special_tokens=special_tokens,
        show_progress=True,
        shrinking_factor=0.75,
        max_piece_length=16,
        n_sub_iterations=2,
    )

    try:
        # トレーニング実行
        tokenizer.train([input_file_path], trainer=trainer)
        print("Tokenizer training completed successfully!")

        # トークナイザー保存
        tokenizer.save(output_path)
        print(f"Tokenizer saved to: {output_path}")

        # 語彙ファイル保存
        vocab_output_path = os.path.join(model_output_dir, f"{model_filename_stem}.vocab")
        vocab_with_scores = tokenizer.get_vocab(with_added_tokens=True)
        with open(vocab_output_path, "w", encoding="utf-8") as f:
            for token, token_id in sorted(vocab_with_scores.items(), key=lambda item: item[1]):
                f.write(f"{token}\t{token_id}\n")
        print(f"Vocabulary saved to: {vocab_output_path}")

        # DeBERTaV2TokenizerFast互換形式で保存
        print("\nCreating DeBERTaV2TokenizerFast compatible tokenizer...")
        deberta_tokenizer = DebertaV2TokenizerFast(
            tokenizer_object=tokenizer,
            unk_token=config["unk_token"],
            sep_token=config["sep_token"],
            pad_token=config["pad_token"],
            cls_token=config["cls_token"],
            mask_token=config["mask_token"],
            do_lower_case=config["do_lower_case"],
            split_by_punct=config["split_by_punct"],
            clean_up_tokenization_spaces=config["clean_up_tokenization_spaces"],
        )

        # HuggingFace形式で保存
        hf_output_dir = os.path.join(model_output_dir, f"{model_filename_stem}_hf")
        deberta_tokenizer.save_pretrained(hf_output_dir)
        print(f"HuggingFace compatible tokenizer saved to: {hf_output_dir}")

        # テスト
        test_text = "これは日本語のテストです。"
        tokens = deberta_tokenizer.tokenize(test_text)
        print(f"\nTest tokenization of '{test_text}': {tokens}")

    except Exception as e:
        print(f"Error during tokenizer training: {e}")
        raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="DeBERTa V3 Japanese tokenizer training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dirs", nargs="+", default=["data/honkoku"], help="Training data directories or files")
    parser.add_argument("--output_dir", default="experiments/tokenizers", help="Output directory for trained tokenizer")
    parser.add_argument("--model_name", default="deberta_v3_japanese_honkoku", help="Model name for output files")
    parser.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size (default: same as reference model)")
    parser.add_argument("--cleanup_temp", action="store_true", help="Remove temporary files after training")

    args = parser.parse_args()

    print("=== DeBERTa V3 Japanese Tokenizer Training ===")
    print(f"Data directories: {args.data_dirs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model name: {args.model_name}")

    try:
        # 1. 参照モデルの設定取得
        config = get_reference_tokenizer_config()

        # 語彙サイズの上書き
        if args.vocab_size:
            config["vocab_size"] = args.vocab_size
            print(f"Vocabulary size overridden to: {args.vocab_size}")

        # 2. テキストファイル収集
        text_files = get_all_text_files(args.data_dirs)

        # 3. 出力ディレクトリ作成
        os.makedirs(args.output_dir, exist_ok=True)

        # 4. ファイル結合（必要に応じて）
        if len(text_files) == 1:
            input_file = text_files[0]
            print(f"Using single input file: {input_file}")
        else:
            input_file = concatenate_files(text_files, args.output_dir)

        # 5. トークナイザー学習
        create_deberta_v3_tokenizer(
            input_file_path=input_file, model_output_dir=args.output_dir, model_filename_stem=args.model_name, config=config
        )

        # 6. 一時ファイル削除
        if args.cleanup_temp and len(text_files) > 1 and os.path.exists(input_file):
            os.remove(input_file)
            print(f"Temporary file removed: {input_file}")

        print("\n=== Training Completed Successfully! ===")
        print(f"Tokenizer files saved in: {args.output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

"""
# 基本的な使用方法
python scripts/train_deberta_v3_japanese_tokenizer.py --data_dirs data/honkoku/honkoku.txt --output_dir experiments/kuzushiji_tokenizer_deberta --model_name deberta_v3_japanese_honkoku --cleanup_temp

# 学習したトークナイザーの使用例
python examples/use_trained_tokenizer.py
"""
