import glob
import os

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, Metaspace
from tokenizers.models import Unigram
from tokenizers.trainers import WordLevelTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# --- 設定項目 ---
# ご自身の環境に合わせて変更してください。

# 学習に使用するテキストファイルが格納されているディレクトリ。
# このディレクトリ内のUTF-8エンコードされた .txt ファイルが学習に使われます。
# 古典籍のテキストファイルをこのディレクトリに配置してください。
TEXT_FILES_DIR = "data/honkoku"  # 例: "data/kotenseki_texts"

# 学習済みトークナイザーの保存先ディレクトリ名
OUTPUT_TOKENIZER_DIR = "experiments/kuzushiji_tokenizer_bigram"

# 語彙サイズの上限。実際のbigram種がこれより少なければ、そのbigram種数が語彙サイズになります。
VOCAB_SIZE = 50000  # bigramの組み合わせに応じて調整してください。1文字より多くなることが想定されます。

# 語彙に含めるための最小出現頻度。これより低い頻度のbigramは [UNK] トークンで扱われます。
MIN_FREQUENCY = 2  # bigram単位なので、1文字単位より高めに設定することを推奨します。

# 特殊トークン。モデルの要件に合わせて調整してください。
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

# --- スクリプト本体 ---


def train_bigram_tokenizer():
    """
    bigram（2文字単位）のトークナイザーを学習し、ファイルに保存します。
    """

    # 学習用テキストファイルのリストを取得
    if not os.path.isdir(TEXT_FILES_DIR):
        print(f"Error: Directory '{TEXT_FILES_DIR}' not found.")
        print("Please create the directory and place your UTF-8 encoded classical Japanese text files (e.g., .txt) in it.")
        return -1
    else:
        text_files = glob.glob(os.path.join(TEXT_FILES_DIR, "*.txt"))

    if not text_files:
        print(f"Error: No .txt files found in '{TEXT_FILES_DIR}'.")
        if "dummy_classical_text.txt" in locals() and os.path.exists("dummy_classical_text.txt"):
            print("Using only the dummy file: dummy_classical_text.txt")
        else:
            print("Please add text files to the directory or check the path.")
            return

    # 1. トークナイザーの初期化
    # WordLevelモデルを使用。語彙はTrainerが学習データから構築します。
    # [UNK] トークンは必須です。
    bigram_tokenizer = Tokenizer(Unigram(unk_token="[UNK]"))

    # 2. Normalizer の設定
    # 古典籍の場合、Unicode正規化 (例: NFKC, NFC) が有効な場合があります。
    # NFKCは互換文字を標準的な文字に変換します（例: ㍿ → 株式会社）。
    # 古典籍の特性（異体字、旧字体など）に応じて、追加の正規化処理を検討・実装することも可能です。
    # (例: normalizers.Replace(Regex("旧字体"), "新字体"))
    # ここではNFKCを基本とし、必要に応じてNFCなども検討してください。
    bigram_tokenizer.normalizer = normalizers.Sequence(
        [
            # normalizers.NFD(),  # 分解 (例: "が" -> "か", "゛")
            # normalizers.StripAccents(), # アクセント記号除去（古典日本語にはほぼ不要）
            # normalizers.NFC(),  # 再結合 (例: "か", "゛" -> "が")
            normalizers.NFKC(),  # 互換文字の正規化 (適用するかはデータと目的に応じて判断)
            normalizers.Strip(left=True, right=True),
            normalizers.Replace(Regex(r"\n"), ""),  # 改行を削除
            normalizers.Replace(Regex(r"[\s]{2,}"), " "),  # 連続するスペースを1つにする
        ]
    )

    # 3. Pre-tokenizer の設定
    bigram_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            
        ]
    )

    bigram_tokenizer.decoder = decoders.Sequence(
        [
            decoders.Metaspace(replacement="_", add_prefix_space=True),
        ]
    )

    # 4. Trainer の設定
    # WordLevelTrainerは、pre_tokenizerによって分割された単位で語彙を学習します。
    trainer = WordLevelTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=MIN_FREQUENCY, show_progress=True, special_tokens=SPECIAL_TOKENS
    )

    # 5. トークナイザーの学習
    # text_files のリストにあるファイルを読み込んで学習します。
    # ファイルはUTF-8でエンコードされていることを想定しています。
    print(f"Starting bigram tokenizer training using files from: {os.path.abspath(TEXT_FILES_DIR)}")
    print(f"Found {len(text_files)} text file(s) for training.")
    if not text_files:  # 追加のチェック
        print("No files to train on. Exiting.")
        return

    try:
        bigram_tokenizer.train(files=text_files, trainer=trainer)
        print("Bigram tokenizer training completed.")
    except Exception as e:
        print(f"Error during tokenizer training: {e}")
        print("Please ensure that the text files exist, are UTF-8 encoded, are not empty, and the path is correct.")
        return

    # 6. (オプション) Decoder の設定
    # WordLevelモデルで、かつpre_tokenizerがIsolatedでbigram単位に分割した場合、
    # デコードは基本的にトークンの結合になります。
    # デフォルトのデコーダで多くの場合問題ありません。
    # Hugging Face `tokenizers` ライブラリの `Tokenizer.decode()` は、
    # 通常、トークン間の不要なスペースの処理や特殊トークンのスキップなどを行います。
    # bigram単位の場合は、単純な文字列結合で十分なことが多いです。
    # 明示的なデコーダ設定は必須ではありませんが、
    # `decoders.Sequence([])` や `decoders.Strip()` などで調整することも可能です。
    # ここではデフォルトのデコード挙動に任せます。

    # 7. トークナイザーの保存
    try:
        os.makedirs(OUTPUT_TOKENIZER_DIR, exist_ok=True)
        bigram_tokenizer.save(os.path.join(OUTPUT_TOKENIZER_DIR, "tokenizer.json"))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(OUTPUT_TOKENIZER_DIR, "tokenizer.json"),
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            bos_token="[CLS]",
            eos_token="[SEP]",
            vocab_file=os.path.join(OUTPUT_TOKENIZER_DIR, "vocab.json"),
            do_lower_case=False,
        )
        tokenizer.save_pretrained(OUTPUT_TOKENIZER_DIR)
        print(f"Bigram tokenizer saved to {os.path.abspath(OUTPUT_TOKENIZER_DIR)}")
        print(f"Vocabulary size: {bigram_tokenizer.get_vocab_size(with_added_tokens=True)}")  # 特殊トークンも含むサイズ
    except Exception as e:
        print(f"Error saving tokenizer: {e}")


def test_tokenizer(tokenizer_path, sample_texts):
    """
    保存されたbigramトークナイザーをロードしてテストします。
    """
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer file not found: {tokenizer_path}")
        return

    print(f"\n--- Testing bigram tokenizer: {tokenizer_path} ---")
    loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    for i, text in enumerate(sample_texts):
        print(f'\nSample text {i + 1}: "{text}"')
        try:
            encoded = loaded_tokenizer(text, add_special_tokens=True)
            print(f"  Tokens: {encoded.tokens}")
            print(f"  IDs: {encoded.input_ids}")
            # print(f"  Attention Mask: {encoded.attention_mask}") # 必要に応じて

            # decode
            print(loaded_tokenizer.convert_ids_to_tokens(encoded.input_ids))
            decoded = loaded_tokenizer.decode(encoded.input_ids, skip_special_tokens=False)
            print(f"  Decoded: {decoded}")

            encoded = loaded_tokenizer.encode(text, add_special_tokens=True)

            # decode
            print(loaded_tokenizer.convert_ids_to_tokens(encoded))
            decoded = loaded_tokenizer.decode(encoded, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            print(f"  Decoded: {decoded}")

        except Exception as e:
            print(f"  Error during tokenization/decoding: {e}")


if __name__ == "__main__":
    # bigramトークナイザーの学習を実行
    train_bigram_tokenizer()

    # 学習したトークナイザーのテスト（ファイルが実際に作成された場合のみ）
    if os.path.exists(os.path.join(OUTPUT_TOKENIZER_DIR, "tokenizer.json")):
        sample_classical_texts = [
            "いろは歌",  # 4文字 -> "いろ", "は歌"
            "竹取の翁なり。",  # 7文字 -> "竹取", "の翁", "なり", "。"
            "祇園精舎の鐘の声、諸行無常の響きあり。",  # bigram分割のテスト
            "源氏物語",  # 4文字 -> "源氏", "物語"
            "あいうえお",  # 5文字 -> "あい", "うえ", "お"
            "アイウエオ",  # 5文字 -> "アイ", "ウエ", "オ"
            "[CLS] テスト [SEP] [PAD]",  # 特殊トークンを含む例
            " ",  # 半角スペース（1文字）
            "　",  # 全角スペース（1文字）
            "",  # 空文字列
            "山路を登りながら、こう考えた。\n智に働けば角が立つ。",  # 改行を含む例とbigramの動作確認
            "a",  # 1文字のテスト
            "ab",  # 2文字のテスト
            "abc",  # 3文字のテスト（"ab", "c"に分割される）
        ]
        test_tokenizer(OUTPUT_TOKENIZER_DIR, sample_classical_texts)
    else:
        print(
            f"\nTokenizer file '{os.path.join(OUTPUT_TOKENIZER_DIR, 'tokenizer.json')}' was not created. Skipping test phase."
        )
