import os
import glob
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, Regex
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast, AutoTokenizer

# --- 設定項目 ---
# ご自身の環境に合わせて変更してください。

# 学習に使用するテキストファイルが格納されているディレクトリ。
# このディレクトリ内のUTF-8エンコードされた .txt ファイルが学習に使われます。
# 古典籍のテキストファイルをこのディレクトリに配置してください。
TEXT_FILES_DIR = "data/honkoku"  # 例: "data/kotenseki_texts"

# 学習済みトークナイザーの保存先ファイル名
OUTPUT_TOKENIZER_DIR = "experiments/kuzushiji_tokenizer_one_char"

# 語彙サイズの上限。実際の文字種がこれより少なければ、その文字種数が語彙サイズになります。
VOCAB_SIZE = 20000  # 古典籍の文字種に応じて調整してください。

# 語彙に含めるための最小出現頻度。これより低い頻度の文字は [UNK] トークンで扱われます。
MIN_FREQUENCY = 1  # 1文字単位なので、基本的に全ての文字を含めたい場合は1や2を設定します。

# 特殊トークン。モデルの要件に合わせて調整してください。
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

# --- スクリプト本体 ---

def train_char_tokenizer():
    """
    1文字単位のトークナイザーを学習し、ファイルに保存します。
    """

    # 学習用テキストファイルのリストを取得
    if not os.path.isdir(TEXT_FILES_DIR):
        print(f"Error: Directory '{TEXT_FILES_DIR}' not found.")
        print("Please create the directory and place your UTF-8 encoded classical Japanese text files (e.g., .txt) in it.")
        # テスト用にダミーファイルを作成する処理（オプション）
        if not os.path.exists("dummy_classical_text.txt"):
            print("Creating a dummy text file for testing: dummy_classical_text.txt")
            with open("dummy_classical_text.txt", "w", encoding="utf-8") as f:
                f.write("いろはにほへと ちりぬるを\n")
                f.write("わかよたれそ つねならむ\n")
                f.write("うゐのおくやま けふこえて\n")
                f.write("あさきゆめみし ゑひもせすん\n")
            text_files = ["dummy_classical_text.txt"]
        else:
            return
    else:
        text_files = glob.glob(os.path.join(TEXT_FILES_DIR, "*.txt"))

    if not text_files:
        print(f"Error: No .txt files found in '{TEXT_FILES_DIR}'.")
        if "dummy_classical_text.txt" in locals() and os.path.exists("dummy_classical_text.txt"):
             print(f"Using only the dummy file: dummy_classical_text.txt")
        else:
            print("Please add text files to the directory or check the path.")
            return


    # 1. トークナイザーの初期化
    # WordLevelモデルを使用。語彙はTrainerが学習データから構築します。
    # [UNK] トークンは必須です。
    char_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

    # 2. Normalizer の設定
    # 古典籍の場合、Unicode正規化 (例: NFKC, NFC) が有効な場合があります。
    # NFKCは互換文字を標準的な文字に変換します（例: ㍿ → 株式会社）。
    # 古典籍の特性（異体字、旧字体など）に応じて、追加の正規化処理を検討・実装することも可能です。
    # (例: normalizers.Replace(Regex("旧字体"), "新字体"))
    # ここではNFCを基本とし、必要に応じてNFKCなども検討してください。
    char_tokenizer.normalizer = normalizers.Sequence([
        #normalizers.NFD(),  # 分解 (例: "が" -> "か", "゛")
        #normalizers.StripAccents(), # アクセント記号除去（古典日本語にはほぼ不要）
        #normalizers.NFC(),  # 再結合 (例: "か", "゛" -> "が")
        normalizers.NFKC(), # 互換文字の正規化 (適用するかはデータと目的に応じて判断)
    ])

    # 3. Pre-tokenizer の設定
    # テキストを1文字ずつに分割します。
    # Regex(r"[\s\S]") は改行を含む任意の1文字にマッチします。
    # behavior="Isolated" により、マッチした各文字が独立したトークンになります。
    char_tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")

    char_tokenizer.decoder = decoders.Sequence([])

    # 4. Trainer の設定
    # WordLevelTrainerは、pre_tokenizerによって分割された単位で語彙を学習します。
    trainer = WordLevelTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS
    )

    # 5. トークナイザーの学習
    # text_files のリストにあるファイルを読み込んで学習します。
    # ファイルはUTF-8でエンコードされていることを想定しています。
    print(f"Starting tokenizer training using files from: {os.path.abspath(TEXT_FILES_DIR)}")
    print(f"Found {len(text_files)} text file(s) for training.")
    if not text_files: # 追加のチェック
        print("No files to train on. Exiting.")
        return

    try:
        char_tokenizer.train(files=text_files, trainer=trainer)
        print("Tokenizer training completed.")
    except Exception as e:
        print(f"Error during tokenizer training: {e}")
        print("Please ensure that the text files exist, are UTF-8 encoded, are not empty, and the path is correct.")
        return

    # 6. (オプション) Decoder の設定
    # WordLevelモデルで、かつpre_tokenizerがIsolatedで文字単位に分割した場合、
    # デコードは基本的にトークンの結合になります。
    # デフォルトのデコーダで多くの場合問題ありません。
    # Hugging Face `tokenizers` ライブラリの `Tokenizer.decode()` は、
    # 通常、トークン間の不要なスペースの処理や特殊トークンのスキップなどを行います。
    # 1文字単位の場合は、単純な文字列結合で十分なことが多いです。
    # 明示的なデコーダ設定は必須ではありませんが、
    # `decoders.Sequence([])` や `decoders.Strip()` などで調整することも可能です。
    # ここではデフォルトのデコード挙動に任せます。

    # 7. トークナイザーの保存
    try:
        os.makedirs(OUTPUT_TOKENIZER_DIR, exist_ok=True)
        char_tokenizer.save(os.path.join(OUTPUT_TOKENIZER_DIR, 'tokenizer.json'))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(OUTPUT_TOKENIZER_DIR, 'tokenizer.json'),
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            bos_token="[CLS]",
            eos_token="[SEP]",
            vocab_file=os.path.join(OUTPUT_TOKENIZER_DIR, 'vocab.json'),
            do_lower_case=False,
        )
        tokenizer.save_pretrained(OUTPUT_TOKENIZER_DIR)
        print(f"Tokenizer saved to {os.path.abspath(OUTPUT_TOKENIZER_DIR)}")
        print(f"Vocabulary size: {char_tokenizer.get_vocab_size(with_added_tokens=True)}") # 特殊トークンも含むサイズ
    except Exception as e:
        print(f"Error saving tokenizer: {e}")


def test_tokenizer(tokenizer_path, sample_texts):
    """
    保存されたトークナイザーをロードしてテストします。
    """
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer file not found: {tokenizer_path}")
        return

    print(f"\n--- Testing tokenizer: {tokenizer_path} ---")
    loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    for i, text in enumerate(sample_texts):
        print(f"\nSample text {i+1}: \"{text}\"")
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
    # トークナイザーの学習を実行
    train_char_tokenizer()

    # 学習したトークナイザーのテスト（ファイルが実際に作成された場合のみ）
    if os.path.exists(os.path.join(OUTPUT_TOKENIZER_DIR, 'tokenizer.json')):
        sample_classical_texts = [
            "いろは歌",
            "竹取の翁なり。",
            "祇園精舎の鐘の声、諸行無常の響きあり。",
            "源氏物語",
            "あいうえお",
            "アイウエオ",
            "[CLS] テスト [SEP] [PAD]", # 特殊トークンを含む例
            " ", # 半角スペース
            "　", # 全角スペース
            "", # 空文字列
            "山路を登りながら、こう考えた。\n智に働けば角が立つ。" # 改行を含む例
        ]
        test_tokenizer(OUTPUT_TOKENIZER_DIR, sample_classical_texts)
    else:
        print(f"\nTokenizer file '{os.path.join(OUTPUT_TOKENIZER_DIR, 'tokenizer.json')}' was not created. Skipping test phase.")

    # ダミーファイルを削除 (テスト用)
    if os.path.exists("dummy_classical_text.txt") and "dummy_classical_text.txt" in (text_files if 'text_files' in locals() else []):
        try:
            os.remove("dummy_classical_text.txt")
            print("\nRemoved dummy_classical_text.txt")
        except OSError as e:
            print(f"Error removing dummy_classical_text.txt: {e}")
