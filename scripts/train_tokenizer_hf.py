import argparse
import glob
import os
import tempfile

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import DebertaV2TokenizerFast


# --- Functions ---
def get_all_text_files(data_dirs):
    """指定されたディレクトリ内のすべての.txtファイルへのパスのリストを取得します。"""
    all_files = []
    for data_dir in data_dirs:
        files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
        all_files.extend(files)
    if not all_files:
        raise FileNotFoundError(f"指定されたディレクトリにテキストファイルが見つかりませんでした: {data_dirs}")
    return all_files


def concatenate_files(file_list, output_dir):
    """指定されたファイルリストの内容を1つの一時ファイルに結合します。"""
    try:
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=output_dir,
            prefix="concatenated_train_data_",
            suffix=".txt",
        )
        print(f"一時ファイルを作成中: {temp_file.name}")
        for filepath in file_list:
            try:
                with open(filepath, encoding="utf-8") as infile:
                    for line in infile:
                        temp_file.write(line)
                    temp_file.write("\n")  # 各ファイルの内容を書き込んだ後に改行を追加
            except Exception as e:
                print(f"ファイル {filepath} の読み込み中にエラー: {e}")
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"一時ファイルの作成中にエラーが発生しました: {e}")
        if "temp_file" in locals() and temp_file.name and os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise


def train_hf_tokenizer(
    input_file_path,  # SentencePieceのinput_fileに相当
    model_output_dir,  # 保存ディレクトリ
    model_filename_stem,  # 保存ファイル名の幹 (例: "kuzushiji_tokenizer")
    vocab_size,
    model_type,  # "bpe" or "unigram"
    normalization_rule_name,  # "nfkc_cf"など
    unk_token_surface,  # UNKトークンの表現
    special_tokens_list=None,  # 特殊トークンのリスト (例: ["[UNK]", "[PAD]"])
):
    """Hugging Faceトークナイザーをトレーニングします。"""
    output_path = os.path.join(model_output_dir, f"{model_filename_stem}.json")
    print(f"モデルを {output_path} に保存します...")
    print(f"出力ディレクトリ: {model_output_dir}")

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
        print(f"ディレクトリを作成しました: {model_output_dir}")

    # 1. Normalizer の設定
    normalizer_list = []
    if "nfkc" in normalization_rule_name.lower():
        normalizer_list.append(normalizers.NFKC())
    if "cf" in normalization_rule_name.lower():  # Case Folding
        normalizer_list.append(normalizers.Lowercase())
    # SentencePieceのnfkc_cfは追加の空白処理も含む場合があるが、ここではNFKCとLowercaseのみ
    if not normalizer_list:  # デフォルトとしてNFC正規化と小文字化を追加することも検討できる
        print(
            f"警告: normalization_rule_name '{normalization_rule_name}' に基づくノーマライザが設定されませんでした。BertNormalizerを使用します。"
        )
        # BertNormalizerはNFD, Lowercase, StripAccentsを含む。
        # 必要に応じて、より単純なものやカスタムシーケンスに変更してください。
        current_normalizer = normalizers.BertNormalizer(lowercase=False, handle_chinese_chars=False, strip_accents=True)
    else:
        normalizer_list.append(normalizers.StripAccents())  # アクセント除去を追加
        normalizer_list.append(normalizers.Replace(r"\s+", " "))  # 連続する空白を1つに (文字列パターンを直接使用)
        current_normalizer = normalizers.Sequence(normalizer_list)

    # 2. Pre-tokenizer の設定
    # SentencePieceのBPEはMetaspaceに似た動作をします。
    # Metaspaceはスペースを特殊文字(デフォルトは ' ')に置き換え、その文字で分割します。
    # add_prefix_space=Trueは、各単語の前にこのMetaspace文字を追加します。
    # これは、単語の最初のトークンとそれ以降のトークンを区別するのに役立ちます。
    # (例: "Hello world" -> " Hello", " world")
    # `\u2581` は SentencePiece でよく使われる Metaspace 文字です。
    current_pre_tokenizer = pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="always")

    # 3. Model と Tokenizer の初期化
    if model_type.lower() == "bpe":
        tokenizer_model = models.BPE(unk_token=str(unk_token_surface))  # unk_tokenは文字列型
        tokenizer = Tokenizer(tokenizer_model)
        tokenizer.pre_tokenizer = current_pre_tokenizer
        # BPE 用デコーダー (Metaspaceの逆処理)
        tokenizer.decoder = decoders.Metaspace(replacement="\u2581", prepend_scheme="always")

    elif model_type.lower() == "unigram":
        # Unigramモデルでは、unk_tokenはtrainerではなくモデルの初期化時に渡すことが推奨される場合がある
        # しかし、UnigramTrainerにもunk_tokenパラメータがある。整合性を取る。
        # models.Unigram() は list of (token, score) で初期化もできるが、ここでは空で。
        tokenizer_model = models.Unigram()  # scores, unk_idなしで初期化
        tokenizer = Tokenizer(tokenizer_model)
        # Unigramの場合も、Metaspaceで初期分割を行うことが一般的
        tokenizer.pre_tokenizer = current_pre_tokenizer
        # Unigram 用デコーダー (多くの場合BPEと同様のMetaspaceで対応可能)
        tokenizer.decoder = decoders.Metaspace(replacement="\u2581", prepend_scheme="always")
    else:
        raise ValueError(f"サポートされていないモデルタイプ: {model_type}")

    tokenizer.normalizer = current_normalizer

    # 4. Trainer の設定とトレーニング
    # 特殊トークンは unk_token を含め、ユーザーが指定するリストで上書きまたは追加
    actual_special_tokens = [str(unk_token_surface)]
    if special_tokens_list:
        for st in special_tokens_list:
            if st not in actual_special_tokens:
                actual_special_tokens.append(st)

    print("トレーニングパラメータ:")
    print(f"  Input file: {input_file_path}")
    print(f"  Output path: {output_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model type: {model_type}")
    print(f"  Normalization rule: {normalization_rule_name} (Interpreted as: {tokenizer.normalizer})")
    print(f"  Pre-tokenizer: {tokenizer.pre_tokenizer}")
    print(f"  UNK token: {unk_token_surface}")
    print(f"  Special tokens for trainer: {actual_special_tokens}")

    if model_type.lower() == "bpe":
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,  # SentencePieceのデフォルトに近い値 (調整可能)
            show_progress=True,
            special_tokens=actual_special_tokens,  # UNKトークンや他の特殊トークン
            # initial_alphabet=pre_tokenizers.ByteLevel.alphabet() # 全てのバイトを初期候補とする場合
            # 日本語など文字種が多い場合は注意
        )
    elif model_type.lower() == "unigram":
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            unk_token=str(unk_token_surface),  # UnigramTrainerにもunk_tokenがある
            special_tokens=actual_special_tokens,
            show_progress=True,
            # SentencePieceのUnigramTrainerのパラメータに対応するものを設定
            # shrinking_factor=0.75,
            # max_piece_length=16,
            # n_sub_iterations=2,
        )
    else:  # 上で捕捉済みだが念のため
        raise ValueError(f"サポートされていないモデルタイプ: {model_type}")

    try:
        # ファイルリストとして渡す
        tokenizer.train([input_file_path], trainer=trainer)
        print("Hugging Face トークナイザーのトレーニングが正常に完了しました。")

        # トレーニング後、語彙に特殊トークンが正しく追加されているか確認し、
        # 必要であれば tokenizer.add_special_tokens() で追加/更新
        # BpeTrainer や UnigramTrainer に special_tokens を渡すと、
        # それらが語彙に含まれるように処理される。

        tokenizer.save(output_path)
        print(f"トークナイザーは {output_path} に保存されました。")

        # オプション: .vocab ファイルをSentencePiece風に保存する場合
        vocab_output_path = os.path.join(model_output_dir, f"{model_filename_stem}.vocab")
        vocab_with_scores = tokenizer.get_vocab(with_added_tokens=True)  # スコアはUnigramの場合のみ意味がある
        with open(vocab_output_path, "w", encoding="utf-8") as f:
            for token, token_id in sorted(vocab_with_scores.items(), key=lambda item: item[1]):
                # Unigramモデルの場合、スコアも保存できるが、BPEの場合はスコアがない
                # SentencePieceの.vocab形式に合わせる (token <tab> id or score)
                # ここでは単純にトークンのみ、またはIDも。
                # SentencePiece .vocab は通常 token <tab> log_probability (Unigram) or score (BPE)
                # Hugging Face get_vocab は id を返すので、ここでは token <tab> id にする
                f.write(f"{token}\t{token_id}\n")
        print(f"語彙ファイル (Hugging Face形式) は {vocab_output_path} に保存されました。")

    except RuntimeError as e:
        print(f"Hugging Face トークナイザーのトレーニング中にランタイムエラーが発生しました: {e}")
        raise
    except Exception as e:
        print(f"Hugging Face トークナイザーのトレーニング中に予期せぬエラーが発生しました: {e}")
        raise


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face トークナイザーのトレーニング")
    parser.add_argument("--vocab_size", type=int, default=32000, help="語彙サイズ")
    parser.add_argument(
        "--model_type", type=str, default="bpe", choices=["bpe", "unigram"], help="モデルタイプ (bpe, unigram)"
    )
    args = parser.parse_args()

    # --- Configuration ---
    DATA_DIRS = [
        "ndl-minhon-ocrdataset/src/honkoku_oneline_v1",
        "ndl-minhon-ocrdataset/src/honkoku_oneline_v2",
        "honkoku_yatanavi/honkoku_oneline",
        "data/oneline",
    ]

    VOCAB_SIZE = args.vocab_size
    MODEL_TYPE = args.model_type

    BASE_EXPERIMENT_DIR = "experiments/kuzushiji_tokenizer_hf"  # 出力先ディレクトリ変更
    MODEL_SPECIFIC_DIR_NAME = f"vocab{VOCAB_SIZE}_{MODEL_TYPE}"
    MODEL_OUTPUT_DIR = os.path.join(BASE_EXPERIMENT_DIR, MODEL_SPECIFIC_DIR_NAME)
    MODEL_FILENAME_STEM = "tokenizer"  # .jsonや.vocabの前の部分

    # SentencePieceパラメータ (Hugging Faceにマッピングまたは参考情報として)
    NORMALIZATION_RULE_NAME = "nfkc"  # NFKC正規化 + Case Folding
    UNK_SURFACE = "[UNK]"

    # Hugging Face 用の特殊トークンリスト (UNKはtrainer/modelに渡すので重複しないように)
    # ここで定義するものは、UNK_SURFACE 以外に追加したいもの
    HF_SPECIAL_TOKENS = [
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "[MASK]",
    ]

    print("Hugging Face トークナイザーのトレーニングを開始します...")
    print(f"設定: VOCAB_SIZE={VOCAB_SIZE}, MODEL_TYPE='{MODEL_TYPE}'")
    print(f"モデルは次の場所に保存されます: {os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME_STEM + '.json')}")
    temp_training_file = None
    try:
        text_files = get_all_text_files(DATA_DIRS)
        print(f"{len(text_files)} 個のテキストファイルが見つかりました。")

        if not os.path.exists(MODEL_OUTPUT_DIR):
            os.makedirs(MODEL_OUTPUT_DIR)
            print(f"出力ディレクトリを作成しました: {MODEL_OUTPUT_DIR}")

        temp_training_file = concatenate_files(text_files, MODEL_OUTPUT_DIR)
        print(f"学習データを一時ファイル {temp_training_file} に結合しました。")

        train_hf_tokenizer(
            input_file_path=temp_training_file,
            model_output_dir=MODEL_OUTPUT_DIR,
            model_filename_stem=MODEL_FILENAME_STEM,
            vocab_size=VOCAB_SIZE,
            model_type=MODEL_TYPE,
            normalization_rule_name=NORMALIZATION_RULE_NAME,
            unk_token_surface=UNK_SURFACE,
            special_tokens_list=HF_SPECIAL_TOKENS,
        )
        print("トークナイザーのトレーニングが完了しました。")

        tokenizer = DebertaV2TokenizerFast(
            tokenizer_file=os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME_STEM + ".json"),
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            bos_token="[CLS]",
            eos_token="[SEP]",
            vocab_file=os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME_STEM + ".vocab"),
            do_lower_case=False,
        )
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("データディレクトリのパスが正しいか、ファイルが存在するか確認してください。")
    except RuntimeError as e:
        print(f"ランタイムエラー: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
    finally:
        if temp_training_file and os.path.exists(temp_training_file):
            try:
                os.remove(temp_training_file)
                print(f"一時ファイル {temp_training_file} を削除しました。")
            except OSError as e:
                print(f"一時ファイル {temp_training_file} の削除に失敗しました: {e}")
