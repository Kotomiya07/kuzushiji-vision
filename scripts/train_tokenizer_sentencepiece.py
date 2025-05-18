import glob
import os
import tempfile
import argparse

import sentencepiece as spm

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
        # 一時ファイルを安全に作成
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,  # train_sentencepiece_tokenizerに渡すため、ここでは削除しない
            dir=output_dir,  # MODEL_OUTPUT_DIRに一時ファイルを作成
            prefix="concatenated_train_data_",
            suffix=".txt",
        )
        print(f"一時ファイルを作成中: {temp_file.name}")
        for filepath in file_list:
            try:
                with open(filepath, encoding="utf-8") as infile:
                    for line in infile:
                        temp_file.write(line)
                    # 各ファイルの内容を書き込んだ後に改行を追加
                    temp_file.write("\n")
            except Exception as e:
                print(f"ファイル {filepath} の読み込み中にエラー: {e}")
                # エラーが発生しても処理を続行するか、ここで例外を発生させるか選択
                # 今回は警告を出して続行
        temp_file.close()  # ファイルを閉じて書き込みを確定
        return temp_file.name
    except Exception as e:
        print(f"一時ファイルの作成中にエラーが発生しました: {e}")
        if "temp_file" in locals() and temp_file.name and os.path.exists(temp_file.name):
            os.remove(temp_file.name)  # エラー発生時は作成途中の一時ファイルを削除
        raise


def train_sentencepiece_tokenizer(
    input_file, model_prefix, vocab_size, character_coverage, model_type, normalization_rule_name
):
    """SentencePieceトークナイザーをトレーニングします (sentencepieceライブラリを直接使用)。"""
    # MODEL_PREFIX に基づいて出力ディレクトリが決定されるため、model_prefix を使用
    output_dir = os.path.dirname(model_prefix)
    print(f"モデルを {model_prefix}.model と {model_prefix}.vocab に保存します...")
    print(f"出力ディレクトリ: {output_dir}")  # 確認用に追加

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"ディレクトリを作成しました: {output_dir}")

    # パラメータを文字列として結合するのではなく、キーワード引数として渡す
    train_params = f" \
        --input={input_file} \
        --model_prefix={model_prefix} \
        --vocab_size={vocab_size} \
        --character_coverage={character_coverage} \
        --model_type={model_type} \
        --normalization_rule_name={normalization_rule_name} \
        --input_sentence_size={INPUT_SENTENCE_SIZE} \
        --shuffle_input_sentence={str(SHUFFLE_INPUT_SENTENCE).lower()} \
        --hard_vocab_limit={str(HARD_VOCAB_LIMIT).lower()} \
        --unk_surface={UNK_SURFACE}"
    # model_type は小文字である必要がある場合があるため、明示的に .lower() は避ける
    # shuffle_input_sentence と hard_vocab_limit は boolean だが、spm.SentencePieceTrainer.train() に渡す際は
    # 文字列化された "true"/"false" か、直接booleanで渡せるか確認が必要。
    # sentencepieceのドキュメントや使用例を参照すると、多くの場合、直接的なPythonの型で渡せる。

    print("SentencePieceトレーニングを開始します。パラメータ:")
    print(f"  Input file: {input_file}")
    print(f"  Model prefix: {model_prefix}")  # このmodel_prefixが完全なパスになっていることを確認
    print(f"  Vocab size: {vocab_size}")
    print(f"  Character coverage: {character_coverage}")
    print(f"  Model type: {model_type}")
    print(f"  Normalization rule: {normalization_rule_name}")
    print(f"  Input sentence size: {INPUT_SENTENCE_SIZE}")
    print(f"  Shuffle input: {SHUFFLE_INPUT_SENTENCE}")
    print(f"  Hard vocab limit: {HARD_VOCAB_LIMIT}")
    print(f"  UNK surface: {UNK_SURFACE}")

    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            normalization_rule_name=normalization_rule_name,
            input_sentence_size=INPUT_SENTENCE_SIZE,
            shuffle_input_sentence=SHUFFLE_INPUT_SENTENCE,
            hard_vocab_limit=HARD_VOCAB_LIMIT,
            unk_surface=UNK_SURFACE,
            # sentencepiece ライブラリ v0.1.96 時点では unk_surface は直接指定可能
            # それ以前のバージョンや spm_train コマンドラインでは --unk_piece と --unk_id も関連
            # ここでは unk_surface のみ指定
        )
        print("SentencePieceトークナイザーのトレーニングが正常に完了しました。")
        print(f"モデルは {model_prefix}.model に保存されました")
        print(f"語彙は {model_prefix}.vocab に保存されました")

    except RuntimeError as e:
        print(f"SentencePieceトークナイザーのトレーニング中にランタイムエラーが発生しました: {e}")
        raise
    except Exception as e:
        print(f"SentencePieceトークナイザーのトレーニング中に予期せぬエラーが発生しました: {e}")
        raise


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentencePieceトークナイザーのトレーニング")
    parser.add_argument("--vocab_size", type=int, default=32000, help="語彙サイズ")
    parser.add_argument("--model_type", type=str, default="bpe", help="モデルタイプ")
    args = parser.parse_args()

    # --- Configuration ---
    DATA_DIRS = [
        "ndl-minhon-ocrdataset/src/honkoku_oneline_v1",
        "ndl-minhon-ocrdataset/src/honkoku_oneline_v2",
        "honkoku_yatanavi/honkoku_oneline",
    ]

    VOCAB_SIZE = args.vocab_size
    MODEL_TYPE = args.model_type # "bpe", "unigram", "char", or "word"

    # 動的な出力ディレクトリとモデルプレフィックス
    BASE_EXPERIMENT_DIR = "experiments/kuzushiji_tokenizer"  # ベースの実験ディレクトリ
    MODEL_SPECIFIC_DIR_NAME = f"vocab{VOCAB_SIZE}_{MODEL_TYPE}"
    MODEL_OUTPUT_DIR = os.path.join(BASE_EXPERIMENT_DIR, MODEL_SPECIFIC_DIR_NAME)
    MODEL_PREFIX = os.path.join(MODEL_OUTPUT_DIR, "kuzushiji_tokenizer")  # モデルファイル名自体は固定

    CHARACTER_COVERAGE = 0.9999
    NORMALIZATION_RULE_NAME = "nfkc_cf"
    INPUT_SENTENCE_SIZE = 12000000
    SHUFFLE_INPUT_SENTENCE = True
    HARD_VOCAB_LIMIT = False
    UNK_SURFACE = "\u2581UNK"

    print("SentencePieceトークナイザーのトレーニングを開始します...")
    print(f"設定: VOCAB_SIZE={VOCAB_SIZE}, MODEL_TYPE='{MODEL_TYPE}'")  # 設定値を表示
    print(f"モデルは次の場所に保存されます: {MODEL_PREFIX}.model / .vocab")  # 保存先を表示
    temp_training_file = None
    try:
        text_files = get_all_text_files(DATA_DIRS)
        print(f"{len(text_files)} 個のテキストファイルが見つかりました。")
        # print("最初の数ファイル:", text_files[:5]) # デバッグ用

        # MODEL_OUTPUT_DIR がなければ作成 (concatenate_filesで利用するため)
        if not os.path.exists(MODEL_OUTPUT_DIR):
            os.makedirs(MODEL_OUTPUT_DIR)
            print(f"出力ディレクトリを作成しました: {MODEL_OUTPUT_DIR}")

        temp_training_file = concatenate_files(text_files, MODEL_OUTPUT_DIR)
        print(f"学習データを一時ファイル {temp_training_file} に結合しました。")

        train_sentencepiece_tokenizer(
            temp_training_file,  # 結合されたファイルパスを渡す
            MODEL_PREFIX,
            VOCAB_SIZE,
            CHARACTER_COVERAGE,
            MODEL_TYPE,
            NORMALIZATION_RULE_NAME,
        )
        print("トークナイザーのトレーニングが完了しました。")
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
