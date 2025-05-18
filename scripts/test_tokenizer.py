# # %%
# import argparse
# import os

# import sentencepiece as sp

# # コマンドライン引数のパース
# parser = argparse.ArgumentParser(description="SentencePieceトークナイザーのテスト")
# parser.add_argument("--text", type=str, default="御使に竹取出あひてなく事限なし", help="テストするテキスト")

# args = parser.parse_args()

# # experiments/kuzushiji_tokenizerにあるモデルをすべてテスト
# # モデルパスの例： experiments/kuzushiji_tokenizer/vocab8000_bpe/kuzushiji_tokenizer.model
# models_dir = "experiments/kuzushiji_tokenizer"

# print("sentencepieceのモデルをテストします")
# for model_name in os.listdir(models_dir):
#     for model_path in os.listdir(os.path.join(models_dir, model_name)):
#         if model_path.endswith(".model"):
#             sp_processor = sp.SentencePieceProcessor()
#             sp_processor.load(os.path.join(models_dir, model_name, model_path))

#             # テキスト
#             text = args.text

#             # encode
#             encode = sp_processor.encode_as_ids(text)

#             # decode
#             decode_sentence = []
#             for encode_subword in encode:
#                 decode_subword = sp_processor.decode(encode_subword)
#                 decode_sentence.append(decode_subword)
#             # モデルパスからタイプと語彙サイズを取得
#             model_type = model_name.split("_")[1]
#             vocab_size = model_name.split("_")[0].split("vocab")[1]
#             print(f"モデル: {model_name}, タイプ: {model_type}, 語彙サイズ: {vocab_size}")
#             print(f"トークン化結果: {decode_sentence}\n")
# %%
import os
import argparse
from tokenizers import Tokenizer

def test_hf_tokenizer(tokenizer_path, sample_texts):
    """
    学習済みのHugging Faceトークナイザーをテストします。

    Args:
        tokenizer_path (str): 学習済みトークナイザーの .json ファイルへのパス。
        sample_texts (list): エンコード・デコードをテストするための文字列のリスト。
    """
    if not os.path.exists(tokenizer_path):
        print(f"エラー: トークナイザーファイルが見つかりません: {tokenizer_path}")
        return

    # 1. トークナイザーのロード
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        #print(f"トークナイザーを正常にロードしました: {tokenizer_path}\n")
    except Exception as e:
        print(f"トークナイザーのロード中にエラーが発生しました: {e}")
        return

    # 2. トークナイザー情報の表示
    print("--- トークナイザー情報 ---")
    print(f"語彙サイズ: {tokenizer.get_vocab_size()}")
    print(f"UNKトークン: ID={tokenizer.token_to_id(tokenizer.model.unk_token)}, Token='{tokenizer.model.unk_token}'")
    
    # Normalizer, PreTokenizer, Decoder の情報を表示 (デバッグ用)
    # print(f"Normalizer: {tokenizer.normalizer}")
    # print(f"PreTokenizer: {tokenizer.pre_tokenizer}")
    # print(f"Decoder: {tokenizer.decoder}")
    
    # 特殊トークンの確認 (もし明示的に追加されていれば)
    # added_tokens = tokenizer.get_added_vocab() # add_special_tokensなどで追加されたもの
    # if added_tokens:
    #     print("追加された(特殊)トークン:")
    #     for token, token_id in added_tokens.items():
    #         if token_id >= tokenizer.get_vocab_size(with_added_tokens=False): # 本体語彙以外
    #             print(f"  Token: '{token}', ID: {token_id}")
    print("-" * 25)
    print("\n")


    # 3. サンプルテキストのエンコードとデコード
    for i, text in enumerate(sample_texts):
        print(f"--- サンプルテキスト {i+1} ---")
        print(f"原文: '{text}'")

        # エンコード
        try:
            encoding = tokenizer.encode(text)
            print(f"エンコード結果:")
            #print(f"  IDs    : {encoding.ids}")
            print(f"  Tokens : {encoding.tokens}")
            #print(f"  Offsets: {encoding.offsets}") # 各トークンの元テキスト中の位置

            # デコード
            try:
                decoded_text = tokenizer.decode(encoding.ids)
                print(f"デコード結果: '{decoded_text}'")
            except Exception as e:
                print(f"デコード中にエラー: {e}")

        except Exception as e:
            print(f"エンコード中にエラー: {e}")
        
        print("-" * 25 + "\n")


if __name__ == "__main__":
    # 学習スクリプトの出力設定に合わせてパスを構成
    # この部分は、実際の学習済みトークナイザーの場所に合わせて調整してください。
    parser = argparse.ArgumentParser(description="学習済みHugging Faceトークナイザーのテスト")
    parser.add_argument(
        "--tokenizer_file", 
        type=str,
        help="テストするトークナイザーの .json ファイルへのパス。"
    )
    parser.add_argument("--vocab_size", type=int, default=32000, help="学習時の語彙サイズ (パスの推測用)")
    parser.add_argument("--model_type", type=str, default="bpe", choices=["bpe", "unigram"], help="学習時のモデルタイプ (パスの推測用)")

    args = parser.parse_args()

    if args.tokenizer_file:
        TOKENIZER_FILE_PATH = args.tokenizer_file
    else:
        # 学習スクリプトのデフォルト設定からパスを推測
        #print("tokenizer_fileが指定されていないため、学習スクリプトのデフォルト設定からパスを推測します。")
        DEFAULT_VOCAB_SIZE = args.vocab_size
        DEFAULT_MODEL_TYPE = args.model_type
        BASE_EXPERIMENT_DIR = "experiments/kuzushiji_tokenizer_hf"
        MODEL_SPECIFIC_DIR_NAME = f"vocab{DEFAULT_VOCAB_SIZE}_{DEFAULT_MODEL_TYPE}"
        MODEL_FILENAME_STEM = "kuzushiji_tokenizer"
        TOKENIZER_FILE_PATH = os.path.join(
            BASE_EXPERIMENT_DIR, MODEL_SPECIFIC_DIR_NAME, f"{MODEL_FILENAME_STEM}.json"
        )
        #print(f"推測されたトークナイザーパス: {TOKENIZER_FILE_PATH}")


    # テスト用のサンプルテキスト 
    SAMPLE_TEXTS = [
        "御使に竹取出あひてなく事限なし",
        # "いろはにほへとちりぬるを",
        # "さるほどに、エウラウパのうちヒリジヤの国のトロヤといふ所に、アモウニヤといふ里あり。",
        # "去程にえうらうはのうちひりしやの国のとろやと云所にあもうにやといふ里ありそのさとにいそほといふ人ありけり"
    ]

    test_hf_tokenizer(TOKENIZER_FILE_PATH, SAMPLE_TEXTS)
# %%
